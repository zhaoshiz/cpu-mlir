# ===- test_flash_attention.py ----------------------------------------------===
# 
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  SPDX-License-Identifier: BSD-3-Clause.
#  For more license information:
#    https://github.com/qualcomm/cpu-mlir/LICENSE.txt
# 
# ===------------------------------------------------------------------------===

import pytest

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

Z, H, N_CTX, D_HEAD = 1, 1, 1024, 64
BLOCK_M = 256
BLOCK_N = 64
BLOCK_DMODEL = 64
STAGE = 1
NUM_THREADS = 4


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    desc_k,
    desc_v,  #
    qk_scale,  #
    BLOCK_N: tl.constexpr,  #
    N_CTX: tl.constexpr,
):
    # loop over k, v and update accumulator
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([start_n, 0])
        qk = tl.dot(q, tl.trans(k))
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = desc_v.load([start_n, 0])
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def attention_fwd_kernel(
    pid1,  ## Set to 0
    Q,
    K,
    V,
    sm_scale,
    Out,  #
    stride_qz,  # Tensor.stride(0)
    stride_qh,  # Tensor.stride(1)
    stride_qm,  # Tensor.stride(2)
    stride_qk,  # Tensor.stride(3)
    stride_kn,  # Tensor.stride(2)
    stride_kk,  # Tensor.stride(3)
    stride_vk,  # Tensor.stride(2)
    stride_vn,  # Tensor.stride(3)
    stride_om,  # Tensor.stride(2)
    stride_on,  # Tensor.stride(3)
    H,  #
    N_CTX: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
):

    pid0 = tl.program_id(0)
    start_m = pid0
    off_hz = pid1
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    # tensor descriptors
    desc_q = tl.make_tensor_descriptor(
        base=Q + qvk_offset,
        shape=[N_CTX, BLOCK_DMODEL],
        strides=[stride_qm, stride_qk],
        block_shape=[BLOCK_M, BLOCK_DMODEL],
    )
    desc_v = tl.make_tensor_descriptor(
        base=V + qvk_offset,
        shape=[N_CTX, BLOCK_DMODEL],
        strides=[stride_vk, stride_vn],
        block_shape=[BLOCK_N, BLOCK_DMODEL],
    )
    desc_k = tl.make_tensor_descriptor(
        base=K + qvk_offset,
        shape=[N_CTX, BLOCK_DMODEL],
        strides=[stride_kn, stride_kk],
        block_shape=[BLOCK_N, BLOCK_DMODEL],
    )
    desc_o = tl.make_tensor_descriptor(
        base=Out + qvk_offset,
        shape=[N_CTX, BLOCK_DMODEL],
        strides=[stride_om, stride_on],
        block_shape=[BLOCK_M, BLOCK_DMODEL],
    )
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 99999999999999.0  # float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([start_m * BLOCK_M, 0])
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        desc_k,
        desc_v,
        qk_scale,
        BLOCK_N,
        N_CTX,
    )
    # epilogue
    acc = acc / l_i[:, None]
    desc_o.store([start_m * BLOCK_M, 0], acc.to(Out.type.element_ty))


def test_flash_attention():
    query = torch.rand(Z, H, N_CTX, D_HEAD)
    key = torch.rand(Z, H, N_CTX, D_HEAD)
    value = torch.rand(Z, H, N_CTX, D_HEAD)

    matrix_stride_0 = H * N_CTX * D_HEAD
    matrix_stride_1 = N_CTX * D_HEAD
    matrix_stride_2 = D_HEAD
    matrix_stride_3 = 1

    output = torch.zeros_like(query)
    SCALE = 0.5

    grid = (NUM_THREADS,)
    attention_fwd_kernel[grid](
        pid1=0,
        Q=query,
        K=key,
        V=value,
        sm_scale=SCALE,
        Out=output,
        stride_qz=matrix_stride_0,
        stride_qh=matrix_stride_1,
        stride_qm=matrix_stride_2,
        stride_qk=matrix_stride_3,
        stride_kn=matrix_stride_2,
        stride_kk=matrix_stride_3,
        stride_vk=matrix_stride_2,
        stride_vn=matrix_stride_3,
        stride_om=matrix_stride_2,
        stride_on=matrix_stride_3,
        H=H,  #
        N_CTX=N_CTX,  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_DMODEL=BLOCK_DMODEL,  #
        BLOCK_N=BLOCK_N,  #
        STAGE=STAGE,
    )

    reference = F.scaled_dot_product_attention(query, key, value, scale=SCALE)
    assert torch.allclose(output, reference)
