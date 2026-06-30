"""Real bitstream codec utilities for GLC / GP-ResLC image evaluation.

This module implements an evaluation codec that mirrors the GLC inference
graph, but counts bytes from actual entropy-coded streams instead of summing
Gaussian likelihood estimates.

Payload design:
- z uses the published GLC design: fixed-length VQ codebook indices
  (14 bits/index for a 16384-entry codebook).
- y uses four arithmetic-coded streams following the same four-part spatial
  prior order as GLC.
- A compact binary header is counted in the payload size and is sufficient for
  decoding without access to the original image.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchac


MAGIC = b"GPRC1"
VERSION = 1
HEADER_STRUCT = struct.Struct("<5sBHIIIIHIIIIIIB")
STREAM_STRUCT = struct.Struct("<hhI")


@dataclass
class ArithmeticStream:
    lo: int
    hi: int
    data: bytes


@dataclass
class RealCodecPayload:
    q: int
    orig_hw: Tuple[int, int]
    padded_hw: Tuple[int, int]
    y_shape: Tuple[int, int, int]
    z_shape: Tuple[int, int]
    z_count: int
    z_data: bytes
    y_streams: List[ArithmeticStream]

    def to_bytes(self) -> bytes:
        y_c, y_h, y_w = self.y_shape
        z_h, z_w = self.z_shape
        header = HEADER_STRUCT.pack(
            MAGIC,
            VERSION,
            int(self.q),
            int(self.orig_hw[0]),
            int(self.orig_hw[1]),
            int(self.padded_hw[0]),
            int(self.padded_hw[1]),
            int(y_c),
            int(y_h),
            int(y_w),
            int(z_h),
            int(z_w),
            int(self.z_count),
            len(self.z_data),
            len(self.y_streams),
        )
        chunks = [header, self.z_data]
        for stream in self.y_streams:
            chunks.append(STREAM_STRUCT.pack(int(stream.lo), int(stream.hi), len(stream.data)))
            chunks.append(stream.data)
        return b"".join(chunks)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "RealCodecPayload":
        offset = 0
        header = payload[offset:offset + HEADER_STRUCT.size]
        offset += HEADER_STRUCT.size
        (
            magic,
            version,
            q,
            orig_h,
            orig_w,
            pad_h,
            pad_w,
            y_c,
            y_h,
            y_w,
            z_h,
            z_w,
            z_count,
            z_nbytes,
            stream_count,
        ) = HEADER_STRUCT.unpack(header)
        if magic != MAGIC:
            raise ValueError(f"invalid real-codec magic: {magic!r}")
        if version != VERSION:
            raise ValueError(f"unsupported real-codec version: {version}")
        z_data = payload[offset:offset + z_nbytes]
        offset += z_nbytes
        streams: List[ArithmeticStream] = []
        for _ in range(stream_count):
            lo, hi, nbytes = STREAM_STRUCT.unpack(payload[offset:offset + STREAM_STRUCT.size])
            offset += STREAM_STRUCT.size
            data = payload[offset:offset + nbytes]
            offset += nbytes
            streams.append(ArithmeticStream(lo=lo, hi=hi, data=data))
        if offset != len(payload):
            raise ValueError(f"trailing payload bytes: {len(payload) - offset}")
        return cls(
            q=q,
            orig_hw=(orig_h, orig_w),
            padded_hw=(pad_h, pad_w),
            y_shape=(y_c, y_h, y_w),
            z_shape=(z_h, z_w),
            z_count=z_count,
            z_data=z_data,
            y_streams=streams,
        )


def _pack_fixed_width(values: torch.Tensor, bit_width: int) -> bytes:
    vals = values.reshape(-1).to(device="cpu", dtype=torch.int64).tolist()
    acc = 0
    nbits = 0
    out = bytearray()
    mask = (1 << bit_width) - 1
    for v in vals:
        if v < 0 or v > mask:
            raise ValueError(f"value {v} cannot be represented with {bit_width} bits")
        acc = (acc << bit_width) | int(v)
        nbits += bit_width
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
    if nbits:
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out)


def _unpack_fixed_width(data: bytes, count: int, bit_width: int, device: torch.device) -> torch.Tensor:
    vals = []
    acc = 0
    nbits = 0
    mask = (1 << bit_width) - 1
    for byte in data:
        acc = (acc << 8) | byte
        nbits += 8
        while nbits >= bit_width and len(vals) < count:
            nbits -= bit_width
            vals.append((acc >> nbits) & mask)
    if len(vals) != count:
        raise ValueError(f"decoded {len(vals)} fixed-width values, expected {count}")
    return torch.tensor(vals, dtype=torch.long, device=device)


ENTROPY_FAMILIES = {"gaussian", "laplace", "logistic"}


def _normal_cdf(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / (sigma * math.sqrt(2.0))))


def _laplace_cdf(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # Match the Gaussian variance parameterization: var(Laplace(0,b)) = 2b^2.
    b = (sigma / math.sqrt(2.0)).clamp_min(1e-5)
    return torch.where(
        x < 0,
        0.5 * torch.exp(x / b),
        1.0 - 0.5 * torch.exp(-x / b),
    )


def _logistic_cdf(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # Match Gaussian variance: var(Logistic(0,s)) = pi^2 s^2 / 3.
    s = (sigma * math.sqrt(3.0) / math.pi).clamp_min(1e-5)
    return torch.sigmoid(x / s)


def _continuous_cdf(x: torch.Tensor, sigma: torch.Tensor, family: str) -> torch.Tensor:
    if family == "gaussian":
        return _normal_cdf(x, sigma)
    if family == "laplace":
        return _laplace_cdf(x, sigma)
    if family == "logistic":
        return _logistic_cdf(x, sigma)
    raise ValueError(f"unknown entropy family: {family}")


def _cdf_for_range(sigma: torch.Tensor, lo: int, hi: int, family: str = "gaussian") -> torch.Tensor:
    """Return CDF with explicit lower/upper tail symbols.

    Symbols:
      0        : lower tail (< lo)
      1..K     : integer values lo..hi
      K + 1    : upper tail (> hi)

    Actual encoded symbols should be in 1..K. The unused tail symbols preserve
    the untruncated Gaussian mass, so interval probabilities match the
    likelihood model for observed values.
    """
    if hi < lo:
        raise ValueError(f"invalid support [{lo}, {hi}]")
    if family not in ENTROPY_FAMILIES:
        raise ValueError(f"unknown entropy family: {family}")
    sigma = sigma.to(device="cpu", dtype=torch.float32).clamp_min_(1e-5)
    edges = torch.arange(lo, hi + 2, dtype=torch.float32, device=sigma.device) - 0.5
    cdf = torch.empty((sigma.numel(), (hi - lo + 1) + 3), dtype=torch.float32)
    cdf[:, 0] = 0.0
    cdf[:, 1:-1] = _continuous_cdf(edges.unsqueeze(0), sigma.reshape(-1, 1), family)
    cdf[:, -1] = 1.0
    return cdf.clamp_(0.0, 1.0)


def _gaussian_cdf_for_range(sigma: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    return _cdf_for_range(sigma, lo, hi, family="gaussian")


def _encode_continuous_symbols(
    values: torch.Tensor,
    sigma: torch.Tensor,
    family: str = "gaussian",
    scale_factor: float = 1.0,
) -> ArithmeticStream:
    values_cpu = values.reshape(-1).to(device="cpu", dtype=torch.int32)
    if values_cpu.numel() == 0:
        return ArithmeticStream(lo=0, hi=-1, data=b"")
    lo = int(values_cpu.min().item())
    hi = int(values_cpu.max().item())
    if lo < -32767 or hi > 32766:
        raise ValueError(f"symbol range [{lo}, {hi}] is outside int16-safe bounds")
    sigma = sigma.reshape(-1) * max(float(scale_factor), 1e-6)
    cdf = _cdf_for_range(sigma, lo, hi, family=family)
    sym = (values_cpu - lo + 1).to(dtype=torch.int16)
    data = torchac.encode_float_cdf(cdf, sym, needs_normalization=True, check_input_bounds=False)
    return ArithmeticStream(lo=lo, hi=hi, data=data)


def _decode_continuous_symbols(
    stream: ArithmeticStream,
    sigma: torch.Tensor,
    family: str = "gaussian",
    scale_factor: float = 1.0,
) -> torch.Tensor:
    sigma = sigma.reshape(-1) * max(float(scale_factor), 1e-6)
    if sigma.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=sigma.device)
    cdf = _cdf_for_range(sigma, stream.lo, stream.hi, family=family)
    sym = torchac.decode_float_cdf(cdf, stream.data, needs_normalization=True).to(torch.int32)
    tail_low = int((sym == 0).sum().item())
    tail_high = int((sym == (stream.hi - stream.lo + 2)).sum().item())
    if tail_low or tail_high:
        raise ValueError(f"decoded tail symbols: low={tail_low}, high={tail_high}")
    values = sym - 1 + int(stream.lo)
    return values.to(device=sigma.device, dtype=torch.float32)


def _encode_gaussian_symbols(values: torch.Tensor, sigma: torch.Tensor) -> ArithmeticStream:
    return _encode_continuous_symbols(values, sigma, family="gaussian")


def _decode_gaussian_symbols(stream: ArithmeticStream, sigma: torch.Tensor) -> torch.Tensor:
    return _decode_continuous_symbols(stream, sigma, family="gaussian")


def _bernoulli_cdf(num: int, prob_one: float, device: torch.device | str = "cpu") -> torch.Tensor:
    p1 = min(max(float(prob_one), 1e-5), 1.0 - 1e-5)
    p0 = 1.0 - p1
    cdf = torch.empty((int(num), 3), dtype=torch.float32, device=device)
    cdf[:, 0] = 0.0
    cdf[:, 1] = p0
    cdf[:, 2] = 1.0
    return cdf


def _encode_bernoulli_symbols(values: torch.Tensor, prob_one: float) -> ArithmeticStream:
    values_cpu = values.reshape(-1).to(device="cpu", dtype=torch.int16)
    if values_cpu.numel() == 0:
        return ArithmeticStream(lo=0, hi=1, data=b"")
    if int(values_cpu.min().item()) < 0 or int(values_cpu.max().item()) > 1:
        raise ValueError("bernoulli control symbols must be 0/1")
    cdf = _bernoulli_cdf(values_cpu.numel(), prob_one, device="cpu")
    data = torchac.encode_float_cdf(cdf, values_cpu, needs_normalization=True, check_input_bounds=False)
    return ArithmeticStream(lo=0, hi=1, data=data)


def _decode_bernoulli_symbols(
    stream: ArithmeticStream,
    count: int,
    prob_one: float,
    device: torch.device,
) -> torch.Tensor:
    cdf = _bernoulli_cdf(count, prob_one, device="cpu")
    sym = torchac.decode_float_cdf(cdf, stream.data, needs_normalization=True).to(torch.int64)
    if sym.numel() != count:
        raise ValueError(f"decoded {sym.numel()} control symbols, expected {count}")
    if int(sym.min().item()) < 0 or int(sym.max().item()) > 1:
        raise ValueError("decoded non-binary control symbol")
    return sym.to(device=device, dtype=torch.float32)


def _encode_control_streams(control_symbols: torch.Tensor, prob_one: float) -> List[ArithmeticStream]:
    if control_symbols.shape[0] != 1 or control_symbols.shape[1] != 4:
        raise ValueError(f"control_symbols must be 1x4xHzxWz, got {tuple(control_symbols.shape)}")
    hard = (control_symbols.detach() > 0.5).to(torch.int16)
    return [_encode_bernoulli_symbols(hard, prob_one)]


def _decode_control_streams(
    streams: Sequence[ArithmeticStream],
    z_shape: Tuple[int, int],
    prob_one: float,
    device: torch.device,
) -> torch.Tensor:
    if len(streams) != 1:
        raise ValueError(f"expected 1 packed control stream, got {len(streams)}")
    z_h, z_w = z_shape
    vals = _decode_bernoulli_symbols(streams[0], 4 * z_h * z_w, prob_one, device)
    return vals.reshape(1, 4, z_h, z_w)


def _ternary_cdf(num: int, prob_nonzero: float, device: torch.device | str = "cpu") -> torch.Tensor:
    p = min(max(float(prob_nonzero), 1e-5), 1.0 - 1e-5)
    p_side = 0.5 * p
    cdf = torch.empty((int(num), 4), dtype=torch.float32, device=device)
    cdf[:, 0] = 0.0
    cdf[:, 1] = p_side
    cdf[:, 2] = 1.0 - p_side
    cdf[:, 3] = 1.0
    return cdf


def _signed_integer_cdf(
    num: int,
    max_abs: int,
    prob_nonzero: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    max_abs = max(1, int(max_abs))
    p = min(max(float(prob_nonzero), 1e-6), 1.0 - 1e-6)
    cdf = torch.empty((int(num), 2 * max_abs + 2), dtype=torch.float32, device=device)
    cdf[:, 0] = 0.0
    step = p / float(2 * max_abs)
    total = 0.0
    for symbol_idx in range(2 * max_abs + 1):
        if symbol_idx == max_abs:
            total += 1.0 - p
        else:
            total += step
        cdf[:, symbol_idx + 1] = total
    cdf[:, -1] = 1.0
    return cdf


def _encode_ternary_symbols(values: torch.Tensor, prob_nonzero: float) -> ArithmeticStream:
    values_cpu = values.reshape(-1).to(device="cpu", dtype=torch.int16)
    if values_cpu.numel() == 0:
        return ArithmeticStream(lo=-1, hi=1, data=b"")
    lo = int(values_cpu.min().item())
    hi = int(values_cpu.max().item())
    if lo < -1 or hi > 1:
        raise ValueError("signed residual-control symbols must be -1/0/+1")
    # torchac symbols are 0, 1, 2 for -1, 0, +1.
    sym = (values_cpu + 1).to(dtype=torch.int16)
    cdf = _ternary_cdf(values_cpu.numel(), prob_nonzero, device="cpu")
    data = torchac.encode_float_cdf(cdf, sym, needs_normalization=True, check_input_bounds=False)
    return ArithmeticStream(lo=-1, hi=1, data=data)


def _decode_ternary_symbols(
    stream: ArithmeticStream,
    count: int,
    prob_nonzero: float,
    device: torch.device,
) -> torch.Tensor:
    cdf = _ternary_cdf(count, prob_nonzero, device="cpu")
    sym = torchac.decode_float_cdf(cdf, stream.data, needs_normalization=True).to(torch.int64)
    if sym.numel() != count:
        raise ValueError(f"decoded {sym.numel()} residual-control symbols, expected {count}")
    if int(sym.min().item()) < 0 or int(sym.max().item()) > 2:
        raise ValueError("decoded invalid residual-control symbol")
    return (sym - 1).to(device=device, dtype=torch.float32)


def _encode_signed_integer_symbols(
    values: torch.Tensor,
    max_abs: int,
    prob_nonzero: float,
) -> ArithmeticStream:
    max_abs = max(1, int(max_abs))
    values_cpu = values.reshape(-1).to(device="cpu", dtype=torch.int16)
    if values_cpu.numel() == 0:
        return ArithmeticStream(lo=-max_abs, hi=max_abs, data=b"")
    lo = int(values_cpu.min().item())
    hi = int(values_cpu.max().item())
    if lo < -max_abs or hi > max_abs:
        raise ValueError(f"signed control symbols must be in [{-max_abs},{max_abs}]")
    sym = (values_cpu + max_abs).to(dtype=torch.int16)
    cdf = _signed_integer_cdf(values_cpu.numel(), max_abs, prob_nonzero, device="cpu")
    data = torchac.encode_float_cdf(cdf, sym, needs_normalization=True, check_input_bounds=False)
    return ArithmeticStream(lo=-max_abs, hi=max_abs, data=data)


def _decode_signed_integer_symbols(
    stream: ArithmeticStream,
    count: int,
    max_abs: int,
    prob_nonzero: float,
    device: torch.device,
) -> torch.Tensor:
    max_abs = max(1, int(max_abs))
    if stream.lo != -max_abs or stream.hi != max_abs:
        raise ValueError(f"expected signed control range [{-max_abs},{max_abs}], got [{stream.lo},{stream.hi}]")
    cdf = _signed_integer_cdf(count, max_abs, prob_nonzero, device="cpu")
    sym = torchac.decode_float_cdf(cdf, stream.data, needs_normalization=True).to(torch.int64)
    if sym.numel() != count:
        raise ValueError(f"decoded {sym.numel()} signed control symbols, expected {count}")
    if int(sym.min().item()) < 0 or int(sym.max().item()) > 2 * max_abs:
        raise ValueError("decoded invalid signed control symbol")
    return (sym - max_abs).to(device=device, dtype=torch.float32)


def _encode_signed_control_streams(
    control_symbols: torch.Tensor,
    prob_nonzero: float,
    max_abs: int = 1,
) -> List[ArithmeticStream]:
    if control_symbols.ndim != 4 or control_symbols.shape[0] != 1:
        raise ValueError(f"control_symbols must be 1xCxHzxWz, got {tuple(control_symbols.shape)}")
    max_abs = max(1, int(max_abs))
    hard = control_symbols.detach().round().clamp(-max_abs, max_abs).to(torch.int16)
    if max_abs == 1:
        return [_encode_ternary_symbols(hard, prob_nonzero)]
    return [_encode_signed_integer_symbols(hard, max_abs, prob_nonzero)]


def _decode_signed_control_streams(
    streams: Sequence[ArithmeticStream],
    z_shape: Tuple[int, int],
    groups: int,
    prob_nonzero: float,
    device: torch.device,
    max_abs: int = 1,
) -> torch.Tensor:
    if len(streams) != 1:
        raise ValueError(f"expected 1 packed residual-control stream, got {len(streams)}")
    z_h, z_w = z_shape
    groups = int(groups)
    if groups <= 0:
        raise ValueError(f"residual_control_groups must be positive, got {groups}")
    max_abs = max(1, int(max_abs))
    if max_abs == 1:
        vals = _decode_ternary_symbols(streams[0], 4 * groups * z_h * z_w, prob_nonzero, device)
    else:
        vals = _decode_signed_integer_symbols(
            streams[0], 4 * groups * z_h * z_w, max_abs, prob_nonzero, device)
    return vals.reshape(1, 4 * groups, z_h, z_w)


def _apply_gp_reslc_params(
    net,
    z_hat: torch.Tensor,
    params: torch.Tensor,
    q: int,
    predictor_param_mode: str,
    predictor_delta_bound: float,
):
    latent_mean_scaled: Optional[torch.Tensor] = None
    q_shift = getattr(net, "q_embed", None)
    z_cond = z_hat if q_shift is None else z_hat + q_shift[q:q + 1]

    # Stage-aware modes do not use the global prior_predictor, but they still
    # must see the same perceptual gate as train_forward before the four-part
    # prior is evaluated. Skipping this made real-codec consistency checks
    # compare against a different inference graph.
    uses_stage_mode = predictor_param_mode in {
        "stage_quant_gate",
        "stage_latent_residual",
        "stage_residual_quant_gate",
        "stage_residual_entropy_quant_gate",
        "stage_residual_entropy_quant_gate_scale_calib",
        "stage_residual_entropy_quant_gate_residual_refiner",
        "stage_residual_quant_gate_control",
        "stage_residual_entropy_quant_gate_control",
        "stage_residual_entropy_quant_gate_residual_control",
        "stage_residual_entropy_quant_gate_latent_control",
    }
    if not uses_stage_mode:
        if not hasattr(net, "prior_predictor") or net.prior_predictor is None:
            if predictor_param_mode not in {"mean", "scale_mean", "all", "latent_residual"}:
                raise ValueError(f"unknown predictor_param_mode: {predictor_param_mode}")
        else:
            delta_params, _, _ = net.prior_predictor(z_cond)
            if predictor_delta_bound and predictor_delta_bound > 0:
                delta_params = predictor_delta_bound * torch.tanh(delta_params / predictor_delta_bound)

            if predictor_param_mode == "latent_residual":
                latent_mean_scaled = delta_params[:, 2 * net.N:]
            elif predictor_param_mode == "mean":
                masked = torch.zeros_like(delta_params)
                masked[:, 2 * net.N:] = delta_params[:, 2 * net.N:]
                params = params + masked
            elif predictor_param_mode == "scale_mean":
                masked = torch.zeros_like(delta_params)
                masked[:, net.N:] = delta_params[:, net.N:]
                params = params + masked
            elif predictor_param_mode == "all":
                params = params + delta_params
            else:
                raise ValueError(f"unknown predictor_param_mode: {predictor_param_mode}")

    gate = getattr(net, "perceptual_gate", None)
    if gate is not None:
        gate_rho, _ = gate(z_cond)
        params = torch.cat((params[:, :net.N] * gate_rho, params[:, net.N:]), dim=1)
    return params, latent_mean_scaled


def _stage_q_shift(net, q: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    q_embed = getattr(net, "q_embed", None)
    if q_embed is None:
        return None
    return q_embed[int(q):int(q) + 1].to(device=device, dtype=dtype)


def _apply_stage_q_condition(common_params: torch.Tensor, q_shift: Optional[torch.Tensor]) -> torch.Tensor:
    if q_shift is None:
        return common_params
    return common_params + q_shift.to(device=common_params.device, dtype=common_params.dtype)


def _bound_delta(delta: torch.Tensor, bound: float) -> torch.Tensor:
    if bound and bound > 0:
        return bound * torch.tanh(delta / bound)
    return delta


def _stage_delta_and_scales(
    stage_predictor,
    stage_idx: int,
    common_params: torch.Tensor,
    y_hat_so_far: Optional[torch.Tensor],
    base_scales: torch.Tensor,
    predictor_delta_bound: float,
    stage_scale_calibrator=None,
    stage_residual_refiner=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if stage_idx == 0:
        pred = stage_predictor.forward_stage(0, common_params)
    else:
        pred = stage_predictor.forward_stage(stage_idx, common_params, y_hat_so_far)
    if isinstance(pred, tuple):
        delta, scale_mul = pred
        scales = (base_scales * scale_mul).clamp_min(1e-5)
    else:
        delta = pred
        scales = base_scales
    if stage_residual_refiner is not None:
        if stage_idx == 0:
            ref_delta, ref_scale_mul = stage_residual_refiner.forward_stage(0, common_params)
        else:
            ref_delta, ref_scale_mul = stage_residual_refiner.forward_stage(
                stage_idx, common_params, y_hat_so_far)
        delta = delta + ref_delta
        scales = (scales * ref_scale_mul).clamp_min(1e-5)
    if stage_scale_calibrator is not None:
        if stage_idx == 0:
            calib_mul = stage_scale_calibrator.forward_stage(0, common_params)
        else:
            calib_mul = stage_scale_calibrator.forward_stage(stage_idx, common_params, y_hat_so_far)
        scales = (scales * calib_mul).clamp_min(1e-5)
    return _bound_delta(delta, predictor_delta_bound), scales


def _encode_y_four_part(
    net,
    y: torch.Tensor,
    params: torch.Tensor,
    latent_mean_scaled: Optional[torch.Tensor] = None,
    entropy_family: str = "gaussian",
    entropy_scale_factor: float = 1.0,
):
    q_enc, q_dec, scales, means = net.separate_prior(params)
    common_params = net.y_spatial_prior_reduction(params)
    dtype = y.dtype
    device = y.device
    b, c, h, w = y.size()
    masks = net.get_mask_four_parts(b, c, h, w, dtype, device)
    y_scaled = y * q_enc
    y_hat_so_far = None
    streams: List[ArithmeticStream] = []

    for part_idx, mask in enumerate(masks):
        if part_idx == 0:
            cur_scales, cur_means = scales, means
        else:
            prior_in = torch.cat((y_hat_so_far, common_params), dim=1)
            adaptor = (
                net.y_spatial_prior_adaptor_1 if part_idx == 1
                else net.y_spatial_prior_adaptor_2 if part_idx == 2
                else net.y_spatial_prior_adaptor_3
            )
            cur_scales, cur_means = net.y_spatial_prior(adaptor(prior_in)).chunk(2, 1)

        if latent_mean_scaled is None:
            cur_means_total = cur_means
        else:
            cur_means_total = cur_means + latent_mean_scaled
        means_hat = cur_means_total * mask
        y_res = (y_scaled - means_hat) * mask
        y_q = torch.round(y_res)
        active = mask > 0.5
        stream = _encode_continuous_symbols(
            y_q[active], cur_scales[active].clamp_min(1e-5),
            family=entropy_family, scale_factor=entropy_scale_factor)
        streams.append(stream)

        y_hat_part = y_q + means_hat
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part

    y_hat = y_hat_so_far * q_dec
    return streams, y_hat


def _encode_y_four_part_stage_residual(net, y: torch.Tensor, params: torch.Tensor,
                                      predictor_delta_bound: float = 0.0,
                                      q_shift: Optional[torch.Tensor] = None,
                                      entropy_family: str = "gaussian",
                                      entropy_scale_factor: float = 1.0):
    q_enc, q_dec, scales, means = net.separate_prior(params)
    common_params = _apply_stage_q_condition(net.y_spatial_prior_reduction(params), q_shift)
    dtype = y.dtype
    device = y.device
    b, c, h, w = y.size()
    masks = net.get_mask_four_parts(b, c, h, w, dtype, device)
    y_scaled = y * q_enc
    y_hat_so_far = None
    streams: List[ArithmeticStream] = []

    for part_idx, mask in enumerate(masks):
        if part_idx == 0:
            cur_scales, cur_means = scales, means
            delta = net.stage_residual_predictor.forward_stage(0, common_params)
        else:
            prior_in = torch.cat((y_hat_so_far, common_params), dim=1)
            adaptor = (
                net.y_spatial_prior_adaptor_1 if part_idx == 1
                else net.y_spatial_prior_adaptor_2 if part_idx == 2
                else net.y_spatial_prior_adaptor_3
            )
            cur_scales, cur_means = net.y_spatial_prior(adaptor(prior_in)).chunk(2, 1)
            delta = net.stage_residual_predictor.forward_stage(part_idx, common_params, y_hat_so_far)
        if predictor_delta_bound and predictor_delta_bound > 0:
            delta = predictor_delta_bound * torch.tanh(delta / predictor_delta_bound)

        means_hat = (cur_means + delta) * mask
        y_res = (y_scaled - means_hat) * mask
        y_q = torch.round(y_res)
        active = mask > 0.5
        streams.append(_encode_continuous_symbols(
            y_q[active], cur_scales[active].clamp_min(1e-5),
            family=entropy_family, scale_factor=entropy_scale_factor))
        y_hat_part = y_q + means_hat
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part

    return streams, y_hat_so_far * q_dec


def _encode_y_four_part_stage_quant_gate(
    net,
    y: torch.Tensor,
    params: torch.Tensor,
    q_shift: Optional[torch.Tensor] = None,
    entropy_family: str = "gaussian",
    entropy_scale_factor: float = 1.0,
):
    quant_step, scales, means = params.chunk(3, 1)
    quant_step = quant_step.clamp_min(0.5)
    common_params = _apply_stage_q_condition(net.y_spatial_prior_reduction(params), q_shift)
    dtype = y.dtype
    device = y.device
    b, c, h, w = y.size()
    masks = net.get_mask_four_parts(b, c, h, w, dtype, device)
    y_hat_so_far = None
    q_dec_map = torch.zeros_like(quant_step)
    streams: List[ArithmeticStream] = []

    for part_idx, mask in enumerate(masks):
        if part_idx == 0:
            cur_scales, cur_means = scales, means
            rho, _ = net.stage_quant_gate.forward_stage(0, common_params)
        else:
            prior_in = torch.cat((y_hat_so_far, common_params), dim=1)
            adaptor = (
                net.y_spatial_prior_adaptor_1 if part_idx == 1
                else net.y_spatial_prior_adaptor_2 if part_idx == 2
                else net.y_spatial_prior_adaptor_3
            )
            cur_scales, cur_means = net.y_spatial_prior(adaptor(prior_in)).chunk(2, 1)
            rho, _ = net.stage_quant_gate.forward_stage(part_idx, common_params, y_hat_so_far)

        q_enc = 1.0 / (quant_step * rho)
        q_dec_map = q_dec_map + quant_step * rho * mask
        means_hat = cur_means * mask
        y_res = (y * q_enc - means_hat) * mask
        y_q = torch.round(y_res)
        active = mask > 0.5
        streams.append(_encode_continuous_symbols(
            y_q[active], cur_scales[active].clamp_min(1e-5),
            family=entropy_family, scale_factor=entropy_scale_factor))
        y_hat_part = y_q + means_hat
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part

    return streams, y_hat_so_far * q_dec_map


def _channel_group_slices(channels: int, groups: int) -> List[slice]:
    groups = max(1, min(int(groups), int(channels)))
    base = channels // groups
    rem = channels % groups
    out: List[slice] = []
    start = 0
    for group_idx in range(groups):
        width = base + (1 if group_idx < rem else 0)
        out.append(slice(start, start + width))
        start += width
    return out


def _stage_signed_control_to_mean_map(
    stage_control: torch.Tensor,
    channels: int,
    height: int,
    width: int,
    groups: int,
    delta: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    if stage_control is None:
        raise ValueError("stage_control must not be None")
    if float(delta) == 0.0:
        return torch.zeros((1, channels, height, width), device=stage_control.device, dtype=dtype)
    up = F.interpolate(stage_control.to(dtype=dtype), size=(height, width), mode="nearest")
    pieces = []
    for group_idx, sl in enumerate(_channel_group_slices(channels, groups)):
        pieces.append(up[:, group_idx:group_idx + 1].expand(1, sl.stop - sl.start, height, width))
    return torch.cat(pieces, dim=1) * float(delta)


def _select_topk_signed_stage_controls(
    signed_score: torch.Tensor,
    topk_frac: float,
    max_abs: int = 1,
    value_step: float = 1.0,
) -> torch.Tensor:
    symbols = torch.zeros_like(signed_score)
    frac = float(topk_frac)
    if frac <= 0.0 or signed_score.numel() == 0:
        return symbols
    flat_score = signed_score.abs().reshape(-1)
    if float(flat_score.max().item()) <= 0.0:
        return symbols
    k = int(round(frac * flat_score.numel()))
    k = max(1, min(k, flat_score.numel()))
    threshold = torch.topk(flat_score, k, largest=True).values[-1]
    keep = signed_score.abs() >= threshold
    max_abs = max(1, int(max_abs))
    if max_abs == 1:
        symbols[keep] = signed_score.sign()[keep]
    else:
        step = max(float(value_step), 1e-8)
        quantized = torch.round(signed_score / step).clamp(-max_abs, max_abs)
        symbols[keep] = quantized[keep]
    return symbols.clamp(-max_abs, max_abs)


def _make_signed_stage_mean_control_maps(
    net,
    y: torch.Tensor,
    params: torch.Tensor,
    z_shape: Tuple[int, int],
    predictor_delta_bound: float,
    q_shift: Optional[torch.Tensor],
    topk_frac: float,
    groups: int,
    mean_control_delta: float,
    control_levels: int = 1,
) -> torch.Tensor:
    quant_step, scales, means = params.chunk(3, 1)
    quant_step = quant_step.clamp_min(0.5)
    common_params = _apply_stage_q_condition(net.y_spatial_prior_reduction(params), q_shift)
    dtype = y.dtype
    device = y.device
    b, c, h, w = y.size()
    if b != 1:
        raise ValueError("real codec currently expects batch size 1")
    groups = max(1, min(int(groups), int(c)))
    masks = net.get_mask_four_parts(b, c, h, w, dtype, device)
    y_hat_so_far = None
    control_maps = torch.zeros((1, 4 * groups, int(z_shape[0]), int(z_shape[1])), dtype=dtype, device=device)

    for part_idx, mask in enumerate(masks):
        if part_idx == 0:
            cur_scales, cur_means = scales, means
            rho, _ = net.stage_quant_gate.forward_stage(0, common_params)
            delta, cur_scales = _stage_delta_and_scales(
                net.stage_residual_predictor, 0, common_params, None, cur_scales,
                predictor_delta_bound, getattr(net, "stage_scale_calibrator", None),
                getattr(net, "stage_residual_refiner", None))
        else:
            prior_in = torch.cat((y_hat_so_far, common_params), dim=1)
            adaptor = (
                net.y_spatial_prior_adaptor_1 if part_idx == 1
                else net.y_spatial_prior_adaptor_2 if part_idx == 2
                else net.y_spatial_prior_adaptor_3
            )
            cur_scales, cur_means = net.y_spatial_prior(adaptor(prior_in)).chunk(2, 1)
            rho, _ = net.stage_quant_gate.forward_stage(part_idx, common_params, y_hat_so_far)
            delta, cur_scales = _stage_delta_and_scales(
                net.stage_residual_predictor, part_idx, common_params, y_hat_so_far,
                cur_scales, predictor_delta_bound, getattr(net, "stage_scale_calibrator", None),
                getattr(net, "stage_residual_refiner", None))

        q_enc = 1.0 / (quant_step * rho)
        pre_control_res = (y * q_enc - (cur_means + delta)) * mask
        stage_scores = []
        for sl in _channel_group_slices(c, groups):
            group_res = pre_control_res[:, sl].sum(dim=1, keepdim=True)
            group_weight = mask[:, sl].sum(dim=1, keepdim=True)
            pooled_res = F.adaptive_avg_pool2d(group_res, z_shape)
            pooled_weight = F.adaptive_avg_pool2d(group_weight, z_shape).clamp_min(1e-6)
            stage_scores.append(pooled_res / pooled_weight)
        stage_score = torch.cat(stage_scores, dim=1)
        stage_symbols = _select_topk_signed_stage_controls(
            stage_score, topk_frac, max_abs=control_levels, value_step=mean_control_delta)
        control_maps[:, part_idx * groups:(part_idx + 1) * groups] = stage_symbols

        mean_control = _stage_signed_control_to_mean_map(
            stage_symbols, c, h, w, groups, mean_control_delta, dtype)
        means_hat = (cur_means + delta + mean_control) * mask
        y_res = (y * q_enc - means_hat) * mask
        y_q = torch.round(y_res)
        y_hat_part = y_q + means_hat
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part

    return control_maps


def _make_signed_latent_control_maps(
    net,
    y: torch.Tensor,
    y_hat_base: torch.Tensor,
    z_shape: Tuple[int, int],
    topk_frac: float,
    groups: int,
    control_delta: float,
    control_levels: int = 1,
) -> torch.Tensor:
    dtype = y.dtype
    device = y.device
    b, c, h, w = y.size()
    if b != 1:
        raise ValueError("real codec currently expects batch size 1")
    groups = max(1, min(int(groups), int(c)))
    masks = net.get_mask_four_parts(b, c, h, w, dtype, device)
    residual = y - y_hat_base
    control_maps = torch.zeros((1, 4 * groups, int(z_shape[0]), int(z_shape[1])), dtype=dtype, device=device)

    for part_idx, mask in enumerate(masks):
        stage_scores = []
        stage_res = residual * mask
        for sl in _channel_group_slices(c, groups):
            group_res = stage_res[:, sl].sum(dim=1, keepdim=True)
            group_weight = mask[:, sl].sum(dim=1, keepdim=True)
            pooled_res = F.adaptive_avg_pool2d(group_res, z_shape)
            pooled_weight = F.adaptive_avg_pool2d(group_weight, z_shape).clamp_min(1e-6)
            stage_scores.append(pooled_res / pooled_weight)
        stage_score = torch.cat(stage_scores, dim=1)
        stage_symbols = _select_topk_signed_stage_controls(
            stage_score, topk_frac, max_abs=control_levels, value_step=control_delta)
        control_maps[:, part_idx * groups:(part_idx + 1) * groups] = stage_symbols

    return control_maps


def _apply_signed_latent_control(
    net,
    y_hat: torch.Tensor,
    control_maps: torch.Tensor,
    delta: float,
    groups: int,
) -> torch.Tensor:
    if float(delta) == 0.0:
        return y_hat
    b, c, h, w = y_hat.size()
    if b != 1:
        raise ValueError("real codec currently expects batch size 1")
    groups = max(1, min(int(groups), int(c)))
    masks = net.get_mask_four_parts(b, c, h, w, y_hat.dtype, y_hat.device)
    correction = torch.zeros_like(y_hat)
    for part_idx, mask in enumerate(masks):
        stage_control = control_maps[:, part_idx * groups:(part_idx + 1) * groups]
        stage_corr = _stage_signed_control_to_mean_map(
            stage_control, c, h, w, groups, delta, y_hat.dtype)
        correction = correction + stage_corr * mask
    return y_hat + correction


def _encode_y_four_part_stage_residual_quant_gate(
    net,
    y: torch.Tensor,
    params: torch.Tensor,
    predictor_delta_bound: float = 0.0,
    control_maps: Optional[torch.Tensor] = None,
    mean_control_maps: Optional[torch.Tensor] = None,
    mean_control_delta: float = 0.0,
    mean_control_groups: int = 1,
    q_shift: Optional[torch.Tensor] = None,
    entropy_family: str = "gaussian",
    entropy_scale_factor: float = 1.0,
):
    quant_step, scales, means = params.chunk(3, 1)
    quant_step = quant_step.clamp_min(0.5)
    common_params = _apply_stage_q_condition(net.y_spatial_prior_reduction(params), q_shift)
    dtype = y.dtype
    device = y.device
    b, c, h, w = y.size()
    masks = net.get_mask_four_parts(b, c, h, w, dtype, device)
    y_hat_so_far = None
    q_dec_map = torch.zeros_like(quant_step)
    streams: List[ArithmeticStream] = []

    def protect_rho(rho: torch.Tensor, part_idx: int) -> torch.Tensor:
        if control_maps is None:
            return rho
        ctrl = F.interpolate(control_maps[:, part_idx:part_idx + 1], size=(h, w), mode="nearest")
        ctrl = ctrl.to(device=rho.device, dtype=rho.dtype)
        return 1.0 + (rho - 1.0) * (1.0 - ctrl)

    for part_idx, mask in enumerate(masks):
        if part_idx == 0:
            cur_scales, cur_means = scales, means
            rho, _ = net.stage_quant_gate.forward_stage(0, common_params)
            delta, cur_scales = _stage_delta_and_scales(
                net.stage_residual_predictor, 0, common_params, None, cur_scales,
                predictor_delta_bound, getattr(net, "stage_scale_calibrator", None),
                getattr(net, "stage_residual_refiner", None))
        else:
            prior_in = torch.cat((y_hat_so_far, common_params), dim=1)
            adaptor = (
                net.y_spatial_prior_adaptor_1 if part_idx == 1
                else net.y_spatial_prior_adaptor_2 if part_idx == 2
                else net.y_spatial_prior_adaptor_3
            )
            cur_scales, cur_means = net.y_spatial_prior(adaptor(prior_in)).chunk(2, 1)
            rho, _ = net.stage_quant_gate.forward_stage(part_idx, common_params, y_hat_so_far)
            delta, cur_scales = _stage_delta_and_scales(
                net.stage_residual_predictor, part_idx, common_params, y_hat_so_far,
                cur_scales, predictor_delta_bound, getattr(net, "stage_scale_calibrator", None),
                getattr(net, "stage_residual_refiner", None))
        rho = protect_rho(rho, part_idx)

        q_enc = 1.0 / (quant_step * rho)
        q_dec_map = q_dec_map + quant_step * rho * mask
        if mean_control_maps is None:
            mean_control = 0.0
        else:
            stage_control = mean_control_maps[:, part_idx * mean_control_groups:(part_idx + 1) * mean_control_groups]
            mean_control = _stage_signed_control_to_mean_map(
                stage_control, c, h, w, mean_control_groups, mean_control_delta, dtype)
        means_hat = (cur_means + delta + mean_control) * mask
        y_res = (y * q_enc - means_hat) * mask
        y_q = torch.round(y_res)
        active = mask > 0.5
        streams.append(_encode_continuous_symbols(
            y_q[active], cur_scales[active].clamp_min(1e-5),
            family=entropy_family, scale_factor=entropy_scale_factor))
        y_hat_part = y_q + means_hat
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part

    return streams, y_hat_so_far * q_dec_map


def _decode_y_four_part_stage_residual(net, payload: RealCodecPayload, params: torch.Tensor,
                                      predictor_delta_bound: float = 0.0,
                                      q_shift: Optional[torch.Tensor] = None,
                                      entropy_family: str = "gaussian",
                                      entropy_scale_factor: float = 1.0):
    q_enc, q_dec, scales, means = net.separate_prior(params)
    del q_enc
    common_params = _apply_stage_q_condition(net.y_spatial_prior_reduction(params), q_shift)
    device = params.device
    dtype = params.dtype
    c, h, w = payload.y_shape
    masks = net.get_mask_four_parts(1, c, h, w, dtype, device)
    y_hat_so_far = None

    for part_idx, (mask, stream) in enumerate(zip(masks, payload.y_streams)):
        if part_idx == 0:
            cur_scales, cur_means = scales, means
            delta = net.stage_residual_predictor.forward_stage(0, common_params)
        else:
            prior_in = torch.cat((y_hat_so_far, common_params), dim=1)
            adaptor = (
                net.y_spatial_prior_adaptor_1 if part_idx == 1
                else net.y_spatial_prior_adaptor_2 if part_idx == 2
                else net.y_spatial_prior_adaptor_3
            )
            cur_scales, cur_means = net.y_spatial_prior(adaptor(prior_in)).chunk(2, 1)
            delta = net.stage_residual_predictor.forward_stage(part_idx, common_params, y_hat_so_far)
        if predictor_delta_bound and predictor_delta_bound > 0:
            delta = predictor_delta_bound * torch.tanh(delta / predictor_delta_bound)

        active = mask > 0.5
        y_q_part = torch.zeros((1, c, h, w), dtype=dtype, device=device)
        decoded = _decode_continuous_symbols(
            stream, cur_scales[active].clamp_min(1e-5),
            family=entropy_family, scale_factor=entropy_scale_factor)
        y_q_part[active] = decoded.to(device=device, dtype=dtype)
        y_hat_part = y_q_part + (cur_means + delta) * mask
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part

    return y_hat_so_far * q_dec


def _decode_y_four_part_stage_quant_gate(
    net,
    payload: RealCodecPayload,
    params: torch.Tensor,
    q_shift: Optional[torch.Tensor] = None,
    entropy_family: str = "gaussian",
    entropy_scale_factor: float = 1.0,
):
    quant_step, scales, means = params.chunk(3, 1)
    quant_step = quant_step.clamp_min(0.5)
    common_params = _apply_stage_q_condition(net.y_spatial_prior_reduction(params), q_shift)
    device = params.device
    dtype = params.dtype
    c, h, w = payload.y_shape
    masks = net.get_mask_four_parts(1, c, h, w, dtype, device)
    y_hat_so_far = None
    q_dec_map = torch.zeros((1, c, h, w), dtype=dtype, device=device)

    for part_idx, (mask, stream) in enumerate(zip(masks, payload.y_streams)):
        if part_idx == 0:
            cur_scales, cur_means = scales, means
            rho, _ = net.stage_quant_gate.forward_stage(0, common_params)
        else:
            prior_in = torch.cat((y_hat_so_far, common_params), dim=1)
            adaptor = (
                net.y_spatial_prior_adaptor_1 if part_idx == 1
                else net.y_spatial_prior_adaptor_2 if part_idx == 2
                else net.y_spatial_prior_adaptor_3
            )
            cur_scales, cur_means = net.y_spatial_prior(adaptor(prior_in)).chunk(2, 1)
            rho, _ = net.stage_quant_gate.forward_stage(part_idx, common_params, y_hat_so_far)

        active = mask > 0.5
        y_q_part = torch.zeros((1, c, h, w), dtype=dtype, device=device)
        decoded = _decode_continuous_symbols(
            stream, cur_scales[active].clamp_min(1e-5),
            family=entropy_family, scale_factor=entropy_scale_factor)
        y_q_part[active] = decoded.to(device=device, dtype=dtype)
        q_dec_map = q_dec_map + quant_step * rho * mask
        y_hat_part = y_q_part + cur_means * mask
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part

    return y_hat_so_far * q_dec_map


def _decode_y_four_part_stage_residual_quant_gate(
    net,
    payload: RealCodecPayload,
    params: torch.Tensor,
    predictor_delta_bound: float = 0.0,
    control_maps: Optional[torch.Tensor] = None,
    mean_control_maps: Optional[torch.Tensor] = None,
    mean_control_delta: float = 0.0,
    mean_control_groups: int = 1,
    streams: Optional[Sequence[ArithmeticStream]] = None,
    q_shift: Optional[torch.Tensor] = None,
    entropy_family: str = "gaussian",
    entropy_scale_factor: float = 1.0,
):
    quant_step, scales, means = params.chunk(3, 1)
    quant_step = quant_step.clamp_min(0.5)
    common_params = _apply_stage_q_condition(net.y_spatial_prior_reduction(params), q_shift)
    device = params.device
    dtype = params.dtype
    c, h, w = payload.y_shape
    masks = net.get_mask_four_parts(1, c, h, w, dtype, device)
    y_hat_so_far = None
    q_dec_map = torch.zeros((1, c, h, w), dtype=dtype, device=device)
    y_streams = list(payload.y_streams if streams is None else streams)

    def protect_rho(rho: torch.Tensor, part_idx: int) -> torch.Tensor:
        if control_maps is None:
            return rho
        ctrl = F.interpolate(control_maps[:, part_idx:part_idx + 1], size=(h, w), mode="nearest")
        ctrl = ctrl.to(device=rho.device, dtype=rho.dtype)
        return 1.0 + (rho - 1.0) * (1.0 - ctrl)

    for part_idx, (mask, stream) in enumerate(zip(masks, y_streams)):
        if part_idx == 0:
            cur_scales, cur_means = scales, means
            rho, _ = net.stage_quant_gate.forward_stage(0, common_params)
            delta, cur_scales = _stage_delta_and_scales(
                net.stage_residual_predictor, 0, common_params, None, cur_scales,
                predictor_delta_bound, getattr(net, "stage_scale_calibrator", None),
                getattr(net, "stage_residual_refiner", None))
        else:
            prior_in = torch.cat((y_hat_so_far, common_params), dim=1)
            adaptor = (
                net.y_spatial_prior_adaptor_1 if part_idx == 1
                else net.y_spatial_prior_adaptor_2 if part_idx == 2
                else net.y_spatial_prior_adaptor_3
            )
            cur_scales, cur_means = net.y_spatial_prior(adaptor(prior_in)).chunk(2, 1)
            rho, _ = net.stage_quant_gate.forward_stage(part_idx, common_params, y_hat_so_far)
            delta, cur_scales = _stage_delta_and_scales(
                net.stage_residual_predictor, part_idx, common_params, y_hat_so_far,
                cur_scales, predictor_delta_bound, getattr(net, "stage_scale_calibrator", None),
                getattr(net, "stage_residual_refiner", None))
        rho = protect_rho(rho, part_idx)

        active = mask > 0.5
        y_q_part = torch.zeros((1, c, h, w), dtype=dtype, device=device)
        decoded = _decode_continuous_symbols(
            stream, cur_scales[active].clamp_min(1e-5),
            family=entropy_family, scale_factor=entropy_scale_factor)
        y_q_part[active] = decoded.to(device=device, dtype=dtype)
        q_dec_map = q_dec_map + quant_step * rho * mask
        if mean_control_maps is None:
            mean_control = 0.0
        else:
            stage_control = mean_control_maps[:, part_idx * mean_control_groups:(part_idx + 1) * mean_control_groups]
            mean_control = _stage_signed_control_to_mean_map(
                stage_control, c, h, w, mean_control_groups, mean_control_delta, dtype)
        y_hat_part = y_q_part + (cur_means + delta + mean_control) * mask
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part

    return y_hat_so_far * q_dec_map


def _decode_y_four_part(
    net,
    payload: RealCodecPayload,
    params: torch.Tensor,
    latent_mean_scaled: Optional[torch.Tensor] = None,
    entropy_family: str = "gaussian",
    entropy_scale_factor: float = 1.0,
):
    q_enc, q_dec, scales, means = net.separate_prior(params)
    del q_enc
    common_params = net.y_spatial_prior_reduction(params)
    device = params.device
    dtype = params.dtype
    c, h, w = payload.y_shape
    masks = net.get_mask_four_parts(1, c, h, w, dtype, device)
    y_hat_so_far = None

    for part_idx, (mask, stream) in enumerate(zip(masks, payload.y_streams)):
        if part_idx == 0:
            cur_scales, cur_means = scales, means
        else:
            prior_in = torch.cat((y_hat_so_far, common_params), dim=1)
            adaptor = (
                net.y_spatial_prior_adaptor_1 if part_idx == 1
                else net.y_spatial_prior_adaptor_2 if part_idx == 2
                else net.y_spatial_prior_adaptor_3
            )
            cur_scales, cur_means = net.y_spatial_prior(adaptor(prior_in)).chunk(2, 1)

        active = mask > 0.5
        y_q_part = torch.zeros((1, c, h, w), dtype=dtype, device=device)
        decoded = _decode_continuous_symbols(
            stream, cur_scales[active].clamp_min(1e-5),
            family=entropy_family, scale_factor=entropy_scale_factor)
        y_q_part[active] = decoded.to(device=device, dtype=dtype)
        if latent_mean_scaled is None:
            cur_means_total = cur_means
        else:
            cur_means_total = cur_means + latent_mean_scaled
        y_hat_part = y_q_part + cur_means_total * mask
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part

    return y_hat_so_far * q_dec


@torch.no_grad()
def compress_to_real_bitstream(
    net,
    x_padded: torch.Tensor,
    q: int,
    orig_hw: Tuple[int, int],
    predictor_param_mode: str = "scale_mean",
    predictor_delta_bound: float = 0.0,
    entropy_family: str = "gaussian",
    entropy_scale_factor: float = 1.0,
    residual_control_topk_frac: float = 0.0,
    residual_control_prob_nonzero: float = 0.01,
    residual_control_delta: float = 0.25,
    residual_control_groups: int = 1,
    residual_control_levels: int = 1,
) -> Tuple[bytes, Dict[str, float]]:
    """Encode a padded image tensor and return serialized payload bytes."""
    if entropy_family not in ENTROPY_FAMILIES:
        raise ValueError(f"unknown entropy family: {entropy_family}")
    entropy_scale_factor = max(float(entropy_scale_factor), 1e-6)
    curr_q_enc = net.q_enc[q:q + 1]
    y_ori = net.vqgan.encoder(x_padded)
    y = net.enc(y_ori, curr_q_enc)
    z = net.hyper_enc(y)
    z_indices = net.z_vq.get_indices(z).reshape(-1)
    z_hat = net.z_vq.get_quan_feat(
        z_indices.reshape(-1, 1),
        (z.shape[0], z.shape[2], z.shape[3], z.shape[1]),
    )

    params = net.y_prior_fusion(net.hyper_dec(z_hat))
    params, latent_mean_scaled = _apply_gp_reslc_params(
        net, z_hat, params, q, predictor_param_mode, predictor_delta_bound)
    q_shift = _stage_q_shift(net, q, params.device, params.dtype)
    control_streams: List[ArithmeticStream] = []
    control_prob_one = getattr(net, "tiny_control_prob_one", 0.08)
    residual_control_levels = max(1, int(residual_control_levels))
    if predictor_param_mode == "stage_residual_entropy_quant_gate_residual_control":
        control_symbols = _make_signed_stage_mean_control_maps(
            net, y, params, (int(z.shape[2]), int(z.shape[3])),
            predictor_delta_bound=predictor_delta_bound,
            q_shift=q_shift,
            topk_frac=residual_control_topk_frac,
            groups=residual_control_groups,
            mean_control_delta=residual_control_delta,
            control_levels=residual_control_levels,
        )
        control_streams = _encode_signed_control_streams(
            control_symbols, residual_control_prob_nonzero, max_abs=residual_control_levels)
        y_streams, _ = _encode_y_four_part_stage_residual_quant_gate(
            net, y, params, predictor_delta_bound,
            mean_control_maps=control_symbols,
            mean_control_delta=residual_control_delta,
            mean_control_groups=residual_control_groups,
            q_shift=q_shift,
            entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor,
        )
        streams = control_streams + y_streams
    elif predictor_param_mode == "stage_residual_entropy_quant_gate_latent_control":
        y_streams, y_hat_base = _encode_y_four_part_stage_residual_quant_gate(
            net, y, params, predictor_delta_bound, q_shift=q_shift,
            entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor)
        if hasattr(net, "latent_control_encoder") and net.latent_control_encoder is not None:
            common_for_control = _apply_stage_q_condition(net.y_spatial_prior_reduction(params), q_shift)
            control_symbols, _, _ = net.latent_control_encoder(
                y, common_for_control, (int(z.shape[2]), int(z.shape[3])), int(q))
            residual_control_groups = int(getattr(net.latent_control_encoder, "groups", residual_control_groups))
            residual_control_prob_nonzero = float(
                getattr(net, "latent_control_prob_nonzero", residual_control_prob_nonzero))
            residual_control_delta = float(getattr(net, "latent_control_delta", residual_control_delta))
        else:
            control_symbols = _make_signed_latent_control_maps(
                net, y, y_hat_base, (int(z.shape[2]), int(z.shape[3])),
                topk_frac=residual_control_topk_frac,
                groups=residual_control_groups,
                control_delta=residual_control_delta,
                control_levels=residual_control_levels,
            )
        control_streams = _encode_signed_control_streams(
            control_symbols, residual_control_prob_nonzero, max_abs=residual_control_levels)
        streams = control_streams + y_streams
    elif predictor_param_mode in {"stage_residual_quant_gate_control", "stage_residual_entropy_quant_gate_control"}:
        if not hasattr(net, "tiny_control_encoder") or net.tiny_control_encoder is None:
            raise ValueError(f"{predictor_param_mode} requires net.tiny_control_encoder")
        common_for_control = _apply_stage_q_condition(net.y_spatial_prior_reduction(params), q_shift)
        control_symbols, _, _ = net.tiny_control_encoder(
            y, common_for_control, (int(z.shape[2]), int(z.shape[3])), int(q))
        control_symbols = (control_symbols > 0.5).to(dtype=y.dtype)
        control_streams = _encode_control_streams(control_symbols, control_prob_one)
        y_streams, _ = _encode_y_four_part_stage_residual_quant_gate(
            net, y, params, predictor_delta_bound, control_maps=control_symbols,
            q_shift=q_shift, entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor)
        streams = control_streams + y_streams
    elif predictor_param_mode in {
        "stage_residual_quant_gate",
        "stage_residual_entropy_quant_gate",
        "stage_residual_entropy_quant_gate_scale_calib",
        "stage_residual_entropy_quant_gate_residual_refiner",
    }:
        y_streams, _ = _encode_y_four_part_stage_residual_quant_gate(
            net, y, params, predictor_delta_bound, q_shift=q_shift,
            entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor)
        streams = y_streams
    elif predictor_param_mode == "stage_quant_gate":
        y_streams, _ = _encode_y_four_part_stage_quant_gate(
            net, y, params, q_shift=q_shift, entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor)
        streams = y_streams
    elif predictor_param_mode == "stage_latent_residual":
        y_streams, _ = _encode_y_four_part_stage_residual(
            net, y, params, predictor_delta_bound, q_shift=q_shift,
            entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor)
        streams = y_streams
    else:
        y_streams, _ = _encode_y_four_part(
            net, y, params, latent_mean_scaled, entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor)
        streams = y_streams

    z_bits = int(math.ceil(math.log2(net.codebook_size)))
    z_data = _pack_fixed_width(z_indices, z_bits)
    payload = RealCodecPayload(
        q=q,
        orig_hw=orig_hw,
        padded_hw=(int(x_padded.shape[2]), int(x_padded.shape[3])),
        y_shape=(int(y.shape[1]), int(y.shape[2]), int(y.shape[3])),
        z_shape=(int(z.shape[2]), int(z.shape[3])),
        z_count=int(z_indices.numel()),
        z_data=z_data,
        y_streams=streams,
    )
    payload_bytes = payload.to_bytes()
    control_bytes = sum(len(s.data) for s in control_streams)
    y_bytes = sum(len(s.data) for s in y_streams)
    stream_header_bytes = len(streams) * STREAM_STRUCT.size
    fixed_header_bytes = HEADER_STRUCT.size + stream_header_bytes
    stats = {
        "payload_bytes": float(len(payload_bytes)),
        "header_bytes": float(fixed_header_bytes),
        "z_bytes": float(len(z_data)),
        "y_bytes": float(y_bytes),
        "control_bytes": float(control_bytes),
        "y_stream_count": float(len(streams)),
        "entropy_family": entropy_family,
        "entropy_scale_factor": float(entropy_scale_factor),
        "residual_control_nonzero": float(
            (control_symbols != 0).sum().item()) if "control_symbols" in locals() else 0.0,
        "residual_control_symbol_count": float(
            control_symbols.numel()) if "control_symbols" in locals() else 0.0,
    }
    return payload_bytes, stats


@torch.no_grad()
def decompress_from_real_bitstream(
    net,
    payload_bytes: bytes,
    predictor_param_mode: str = "scale_mean",
    predictor_delta_bound: float = 0.0,
    entropy_family: str = "gaussian",
    entropy_scale_factor: float = 1.0,
    residual_control_prob_nonzero: float = 0.01,
    residual_control_delta: float = 0.25,
    residual_control_groups: int = 1,
    residual_control_levels: int = 1,
) -> torch.Tensor:
    """Decode a serialized payload and return padded reconstruction in [-1, 1]."""
    if entropy_family not in ENTROPY_FAMILIES:
        raise ValueError(f"unknown entropy family: {entropy_family}")
    entropy_scale_factor = max(float(entropy_scale_factor), 1e-6)
    payload = RealCodecPayload.from_bytes(payload_bytes)
    device = next(net.parameters()).device
    z_bits = int(math.ceil(math.log2(net.codebook_size)))
    z_indices = _unpack_fixed_width(payload.z_data, payload.z_count, z_bits, device=device)
    z_hat = net.z_vq.get_quan_feat(
        z_indices.reshape(-1, 1),
        (1, payload.z_shape[0], payload.z_shape[1], net.N),
    )

    params = net.y_prior_fusion(net.hyper_dec(z_hat))
    params, latent_mean_scaled = _apply_gp_reslc_params(
        net, z_hat, params, payload.q, predictor_param_mode, predictor_delta_bound)
    q_shift = _stage_q_shift(net, payload.q, params.device, params.dtype)
    control_prob_one = getattr(net, "tiny_control_prob_one", 0.08)
    residual_control_levels = max(1, int(residual_control_levels))
    if predictor_param_mode == "stage_residual_entropy_quant_gate_residual_control":
        if len(payload.y_streams) < 5:
            raise ValueError(
                f"{predictor_param_mode} expects 1 residual-control + 4 y streams, got {len(payload.y_streams)}")
        control_maps = _decode_signed_control_streams(
            payload.y_streams[:1], payload.z_shape, residual_control_groups,
            residual_control_prob_nonzero, device, max_abs=residual_control_levels)
        y_hat = _decode_y_four_part_stage_residual_quant_gate(
            net, payload, params, predictor_delta_bound,
            mean_control_maps=control_maps,
            mean_control_delta=residual_control_delta,
            mean_control_groups=residual_control_groups,
            streams=payload.y_streams[1:],
            q_shift=q_shift,
            entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor)
    elif predictor_param_mode == "stage_residual_entropy_quant_gate_latent_control":
        if len(payload.y_streams) < 5:
            raise ValueError(
                f"{predictor_param_mode} expects 1 latent-control + 4 y streams, got {len(payload.y_streams)}")
        if hasattr(net, "latent_control_encoder") and net.latent_control_encoder is not None:
            residual_control_groups = int(getattr(net.latent_control_encoder, "groups", residual_control_groups))
            residual_control_prob_nonzero = float(
                getattr(net, "latent_control_prob_nonzero", residual_control_prob_nonzero))
            residual_control_delta = float(getattr(net, "latent_control_delta", residual_control_delta))
        control_maps = _decode_signed_control_streams(
            payload.y_streams[:1], payload.z_shape, residual_control_groups,
            residual_control_prob_nonzero, device, max_abs=residual_control_levels)
        y_hat_base = _decode_y_four_part_stage_residual_quant_gate(
            net, payload, params, predictor_delta_bound,
            streams=payload.y_streams[1:],
            q_shift=q_shift,
            entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor)
        y_hat = _apply_signed_latent_control(
            net, y_hat_base, control_maps, residual_control_delta, residual_control_groups)
    elif predictor_param_mode in {"stage_residual_quant_gate_control", "stage_residual_entropy_quant_gate_control"}:
        if len(payload.y_streams) < 5:
            raise ValueError(
                f"{predictor_param_mode} expects 1 control + 4 y streams, got {len(payload.y_streams)}")
        control_maps = _decode_control_streams(
            payload.y_streams[:1], payload.z_shape, control_prob_one, device)
        y_hat = _decode_y_four_part_stage_residual_quant_gate(
            net, payload, params, predictor_delta_bound,
            control_maps=control_maps, streams=payload.y_streams[1:],
            q_shift=q_shift, entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor)
    elif predictor_param_mode in {
        "stage_residual_quant_gate",
        "stage_residual_entropy_quant_gate",
        "stage_residual_entropy_quant_gate_scale_calib",
        "stage_residual_entropy_quant_gate_residual_refiner",
    }:
        y_hat = _decode_y_four_part_stage_residual_quant_gate(
            net, payload, params, predictor_delta_bound, q_shift=q_shift,
            entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor)
    elif predictor_param_mode == "stage_quant_gate":
        y_hat = _decode_y_four_part_stage_quant_gate(
            net, payload, params, q_shift=q_shift, entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor)
    elif predictor_param_mode == "stage_latent_residual":
        y_hat = _decode_y_four_part_stage_residual(
            net, payload, params, predictor_delta_bound, q_shift=q_shift,
            entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor)
    else:
        y_hat = _decode_y_four_part(
            net, payload, params, latent_mean_scaled, entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor)
    curr_q_dec = net.q_dec[payload.q:payload.q + 1]
    latent = net.dec(y_hat, curr_q_dec)
    return net.vqgan.generator(latent)


def crop_to_original(x_hat_padded: torch.Tensor, orig_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = orig_hw
    return x_hat_padded[..., :h, :w]
