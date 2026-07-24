"""Testes do RMSNorm (nb 04) — identidade RMS, equivalência a LayerNorm, precisão BF16."""
from __future__ import annotations

import torch
import torch.nn as nn

from tucanoce.model.layers.rmsnorm import RMSNorm


def test_shape_preserved():
    norm = RMSNorm(16)
    x = torch.randn(4, 8, 16)
    assert norm(x).shape == x.shape


def test_equals_layernorm_when_mean_zero():
    # com gamma=1 e entrada de média 0, RMSNorm coincide com LayerNorm sem afim
    d = 64
    norm = RMSNorm(d)
    ln = nn.LayerNorm(d, elementwise_affine=False)
    x = torch.randn(4, d)
    x = x - x.mean(dim=-1, keepdim=True)          # força média 0
    assert torch.allclose(norm(x), ln(x), atol=1e-4)


def test_differs_from_layernorm_when_mean_nonzero():
    d = 64
    norm = RMSNorm(d)
    ln = nn.LayerNorm(d, elementwise_affine=False)
    x = torch.randn(4, d) + 10.0                  # média != 0
    assert (norm(x) - ln(x)).abs().max() > 0.1    # RMSNorm não centraliza


def test_normalizes_to_unit_rms():
    # com gamma=1, o RMS da saída deve ser ~1 (projeta na esfera de raio sqrt(d))
    d = 128
    norm = RMSNorm(d)
    x = torch.randn(4, d) * 5.0
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones(4), atol=1e-2)


def test_gamma_scales():
    d = 8
    norm = RMSNorm(d)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    x = torch.randn(2, d)
    norm.weight.data.fill_(1.0)
    out1 = norm(x)
    norm.weight.data.fill_(2.0)
    out2 = norm(x)
    assert torch.allclose(out2, 2.0 * out1, atol=1e-5)


def test_bf16_computes_rms_in_fp32():
    # ABLAÇÃO: com o módulo inteiro em BF16 (cenário real de treino), o RMS ainda é
    # calculado em FP32 (x.float() interno). A saída sai em BF16 mas permanece
    # numericamente próxima do cálculo FP32 puro — é o ganho de precisão do design.
    d = 512
    norm_fp32 = RMSNorm(d)
    norm_bf16 = RMSNorm(d).bfloat16()             # gamma também em BF16
    x = torch.randn(4, d)

    out_fp32 = norm_fp32(x.float())
    out_bf16 = norm_bf16(x.bfloat16())

    assert out_bf16.dtype == torch.bfloat16
    # próxima do fp32 apesar do dtype reduzido (tolerância BF16 ~1e-2)
    assert (out_bf16.float() - out_fp32).abs().max() < 5e-2
