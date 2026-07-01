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
from pathlib import Path
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


Z_ENTROPY_MODES = {"fixed", "static", "auto"}
Z_STATIC_PREFIX = b"ZAC0"
Z_FIXED_PREFIX = b"ZFX0"
_Z_PROBS_CACHE: Dict[Tuple[str, int], torch.Tensor] = {}


def _load_z_entropy_probs(path: str, codebook_size: int) -> torch.Tensor:
    if not path:
        raise ValueError("z_entropy_cdf_path is required for static z entropy coding")
    key = (str(Path(path).resolve()), int(codebook_size))
    cached = _Z_PROBS_CACHE.get(key)
    if cached is not None:
        return cached
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        probs = obj.get("probs")
        if probs is None:
            counts = obj.get("counts")
            if counts is None:
                raise ValueError(f"{path} must contain 'probs' or 'counts'")
            probs = counts.to(dtype=torch.float64)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    else:
        probs = obj
    probs = torch.as_tensor(probs, dtype=torch.float64, device="cpu")
    if probs.ndim == 1:
        probs = probs.unsqueeze(0).repeat(4, 1)
    if probs.ndim != 2 or probs.shape[1] != int(codebook_size):
        raise ValueError(
            f"invalid z entropy probs shape {tuple(probs.shape)}, expected [Q,{codebook_size}]")
    probs = probs.clamp_min(1e-12)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    _Z_PROBS_CACHE[key] = probs.to(dtype=torch.float32)
    return _Z_PROBS_CACHE[key]


def _z_cdf_from_probs(probs: torch.Tensor, count: int) -> torch.Tensor:
    probs = probs.to(device="cpu", dtype=torch.float32)
    cdf_1d = torch.empty((probs.numel() + 1,), dtype=torch.float32)
    cdf_1d[0] = 0.0
    cdf_1d[1:] = torch.cumsum(probs, dim=0)
    cdf_1d[-1] = 1.0
    return cdf_1d.unsqueeze(0).expand(int(count), -1).contiguous()


def _encode_z_static(values: torch.Tensor, probs: torch.Tensor) -> bytes:
    values_cpu = values.reshape(-1).to(device="cpu", dtype=torch.int16)
    if values_cpu.numel() == 0:
        return Z_STATIC_PREFIX
    if int(values_cpu.min().item()) < 0 or int(values_cpu.max().item()) >= probs.numel():
        raise ValueError("z index outside static entropy codebook range")
    cdf = _z_cdf_from_probs(probs, values_cpu.numel())
    data = torchac.encode_float_cdf(cdf, values_cpu, needs_normalization=True, check_input_bounds=False)
    return Z_STATIC_PREFIX + data


def _decode_z_static(data: bytes, count: int, probs: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not data.startswith(Z_STATIC_PREFIX):
        raise ValueError("static z payload is missing ZAC0 prefix")
    body = data[len(Z_STATIC_PREFIX):]
    if int(count) == 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    cdf = _z_cdf_from_probs(probs, int(count))
    sym = torchac.decode_float_cdf(cdf, body, needs_normalization=True).to(torch.int64)
    if sym.numel() != int(count):
        raise ValueError(f"decoded {sym.numel()} z indices, expected {count}")
    if int(sym.min().item()) < 0 or int(sym.max().item()) >= probs.numel():
        raise ValueError("decoded invalid z index")
    return sym.to(device=device, dtype=torch.long)


def _encode_z_indices(
    values: torch.Tensor,
    codebook_size: int,
    mode: str,
    q: int,
    cdf_path: Optional[str],
) -> bytes:
    if mode not in Z_ENTROPY_MODES:
        raise ValueError(f"unknown z_entropy_mode: {mode}")
    z_bits = int(math.ceil(math.log2(codebook_size)))
    fixed = _pack_fixed_width(values, z_bits)
    if mode == "fixed":
        return fixed
    probs = _load_z_entropy_probs(str(cdf_path or ""), codebook_size)
    q_idx = min(max(int(q), 0), probs.shape[0] - 1)
    static = _encode_z_static(values, probs[q_idx])
    if mode == "static":
        return static
    fixed_tagged = Z_FIXED_PREFIX + fixed
    return static if len(static) < len(fixed_tagged) else fixed_tagged


def _decode_z_indices(
    data: bytes,
    count: int,
    codebook_size: int,
    q: int,
    device: torch.device,
    mode: str,
    cdf_path: Optional[str],
) -> torch.Tensor:
    z_bits = int(math.ceil(math.log2(codebook_size)))
    if data.startswith(Z_STATIC_PREFIX):
        probs = _load_z_entropy_probs(str(cdf_path or ""), codebook_size)
        q_idx = min(max(int(q), 0), probs.shape[0] - 1)
        return _decode_z_static(data, count, probs[q_idx], device)
    if data.startswith(Z_FIXED_PREFIX):
        return _unpack_fixed_width(data[len(Z_FIXED_PREFIX):], count, z_bits, device=device)
    if mode in {"fixed", "auto"}:
        return _unpack_fixed_width(data, count, z_bits, device=device)
    raise ValueError("static z entropy mode requested, but payload has no static z prefix")


ENTROPY_FAMILIES = {"gaussian", "laplace", "logistic"}
OMITTED_RESIDUAL_MODES = {
    "zero",
    "hash_gaussian",
    "hash_gaussian_clipped",
    "hash_rademacher",
    "learned_symbol",
    "learned_value",
}


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


def _encode_stage3_send_stream(send_map: torch.Tensor, prob_one: float) -> ArithmeticStream:
    if send_map.shape[0] != 1 or send_map.shape[1] != 1:
        raise ValueError(f"stage3 send map must be 1x1xHzxWz, got {tuple(send_map.shape)}")
    return _encode_bernoulli_symbols((send_map.detach() > 0.5).to(torch.int16), prob_one)


def _decode_stage3_send_stream(
    stream: ArithmeticStream,
    z_shape: Tuple[int, int],
    prob_one: float,
    device: torch.device,
) -> torch.Tensor:
    z_h, z_w = z_shape
    vals = _decode_bernoulli_symbols(stream, int(z_h) * int(z_w), prob_one, device)
    return vals.reshape(1, 1, int(z_h), int(z_w))


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
        "stage_residual_entropy_quant_gate_stage3_send_control",
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


def _normalize_stage_set(stages: Optional[Sequence[int]]) -> set[int]:
    if stages is None:
        return set()
    out = {int(s) for s in stages}
    invalid = sorted(s for s in out if s < 0 or s > 3)
    if invalid:
        raise ValueError(f"suppress_yq_stages must be in 0..3, got {invalid}")
    return out


def _deterministic_noise_like(ref: torch.Tensor, stage_idx: int, q: int) -> torch.Tensor:
    """Return decoder-computable pseudo-noise with no transmitted seed.

    This is for diagnostics, not a trained synthesis model.  It depends only on
    tensor coordinates, q, and stage index, so encoder and decoder generate the
    same values without side information.
    """
    flat = torch.arange(ref.numel(), device=ref.device, dtype=torch.float32).reshape(ref.shape)
    seed = float((int(stage_idx) + 1) * 1009 + (int(q) + 1) * 9176)
    raw = torch.sin(flat * 12.9898 + seed) * 43758.5453
    u = raw - torch.floor(raw)
    return (2.0 * u - 1.0).to(dtype=ref.dtype)


def _omitted_residual_values(
    scales: torch.Tensor,
    mask: torch.Tensor,
    stage_idx: int,
    q: int,
    mode: str,
    amplitude: float,
    clip: float,
) -> torch.Tensor:
    if mode not in OMITTED_RESIDUAL_MODES:
        raise ValueError(f"unknown omitted residual mode: {mode}")
    if mode in {"learned_symbol", "learned_value"}:
        raise ValueError(f"{mode} requires decoder context; call _learned_or_omitted_residual_values")
    if mode == "zero" or float(amplitude) == 0.0:
        return torch.zeros_like(scales) * mask
    eps = _deterministic_noise_like(scales, stage_idx, q)
    if mode in {"hash_gaussian", "hash_gaussian_clipped"}:
        # Convert deterministic uniform-like values to an approximately normal
        # signal.  Clamp before erfinv to avoid infinities.
        eps = eps.clamp(-0.999, 0.999)
        eps = torch.erfinv(eps) * math.sqrt(2.0)
        if mode == "hash_gaussian_clipped":
            eps = eps.clamp(-abs(float(clip)), abs(float(clip)))
    elif mode == "hash_rademacher":
        eps = torch.where(eps >= 0, torch.ones_like(eps), -torch.ones_like(eps))
    return float(amplitude) * scales.to(dtype=eps.dtype) * eps * mask


def _learned_or_omitted_residual_values(
    net,
    common_params: torch.Tensor,
    y_hat_so_far: Optional[torch.Tensor],
    scales: torch.Tensor,
    mask: torch.Tensor,
    stage_idx: int,
    q: int,
    mode: str,
    amplitude: float,
    clip: float,
) -> torch.Tensor:
    if mode == "learned_symbol":
        synth = getattr(net, "stage_residual_symbol_synthesizer", None)
        if synth is None:
            raise ValueError("omitted_residual_mode=learned_symbol requires net.stage_residual_symbol_synthesizer")
        return synth.predict_symbols(stage_idx, common_params, y_hat_so_far).to(dtype=scales.dtype) * mask
    if mode == "learned_value":
        synth = getattr(net, "stage_residual_value_synthesizer", None)
        if synth is None:
            raise ValueError("omitted_residual_mode=learned_value requires net.stage_residual_value_synthesizer")
        return synth.predict_values(stage_idx, common_params, y_hat_so_far).to(dtype=scales.dtype) * mask
    return _omitted_residual_values(scales, mask, stage_idx, q, mode, amplitude, clip)


def _learned_value_addition(
    net,
    common_params: torch.Tensor,
    y_hat_so_far: Optional[torch.Tensor],
    mask: torch.Tensor,
    stage_idx: int,
    value_scale: float,
) -> torch.Tensor:
    if stage_idx != 3:
        raise ValueError("decoder-side value addition is currently restricted to stage 3")
    synth = getattr(net, "stage_residual_value_synthesizer", None)
    if synth is None:
        raise ValueError("synth value addition requires net.stage_residual_value_synthesizer")
    return float(value_scale) * synth.predict_values(stage_idx, common_params, y_hat_so_far).to(dtype=mask.dtype) * mask


def _append_stage_stream_or_omit(
    streams: List[ArithmeticStream],
    y_q: torch.Tensor,
    scales: torch.Tensor,
    mask: torch.Tensor,
    part_idx: int,
    suppressed_stages: set[int],
    entropy_family: str,
    entropy_scale_factor: float,
) -> None:
    if part_idx in suppressed_stages:
        streams.append(ArithmeticStream(lo=0, hi=-1, data=b""))
        return
    active = mask > 0.5
    streams.append(_encode_continuous_symbols(
        y_q[active], scales[active].clamp_min(1e-5),
        family=entropy_family, scale_factor=entropy_scale_factor))


def _stage_suppression_mask(
    mask: torch.Tensor,
    rho: Optional[torch.Tensor],
    part_idx: int,
    suppressed_stages: set[int],
    rho_threshold: float,
) -> torch.Tensor:
    active = mask > 0.5
    if float(rho_threshold) > 0.0:
        if suppressed_stages and part_idx not in suppressed_stages:
            return torch.zeros_like(mask, dtype=torch.bool)
        if rho is None:
            raise ValueError("rho_threshold suppression requires a stage rho map")
        return active & (rho >= float(rho_threshold))
    if part_idx in suppressed_stages:
        return active
    return torch.zeros_like(mask, dtype=torch.bool)


def _append_symbols_with_suppression(
    streams: List[ArithmeticStream],
    y_q: torch.Tensor,
    scales: torch.Tensor,
    mask: torch.Tensor,
    suppress_mask: torch.Tensor,
    entropy_family: str,
    entropy_scale_factor: float,
) -> None:
    active = (mask > 0.5) & (~suppress_mask)
    streams.append(_encode_continuous_symbols(
        y_q[active], scales[active].clamp_min(1e-5),
        family=entropy_family, scale_factor=entropy_scale_factor))


def _encode_y_four_part(
    net,
    y: torch.Tensor,
    params: torch.Tensor,
    latent_mean_scaled: Optional[torch.Tensor] = None,
    entropy_family: str = "gaussian",
    entropy_scale_factor: float = 1.0,
    suppress_yq_stages: Optional[Sequence[int]] = None,
    omitted_residual_mode: str = "zero",
    omitted_residual_scale: float = 0.0,
    omitted_residual_clip: float = 2.0,
    q: int = 0,
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
    suppressed_stages = _normalize_stage_set(suppress_yq_stages)

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
        if part_idx in suppressed_stages:
            y_q = _learned_or_omitted_residual_values(
                net, common_params, y_hat_so_far, cur_scales, mask, part_idx, q,
                omitted_residual_mode, omitted_residual_scale, omitted_residual_clip)
        else:
            y_q = torch.round(y_res)
        _append_stage_stream_or_omit(
            streams, y_q, cur_scales, mask, part_idx, suppressed_stages,
            entropy_family, entropy_scale_factor)

        y_hat_part = y_q + means_hat
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part

    y_hat = y_hat_so_far * q_dec
    return streams, y_hat


def _encode_y_four_part_stage_residual(net, y: torch.Tensor, params: torch.Tensor,
                                      predictor_delta_bound: float = 0.0,
                                      q_shift: Optional[torch.Tensor] = None,
                                      entropy_family: str = "gaussian",
                                      entropy_scale_factor: float = 1.0,
                                      suppress_yq_stages: Optional[Sequence[int]] = None,
                                      omitted_residual_mode: str = "zero",
                                      omitted_residual_scale: float = 0.0,
                                      omitted_residual_clip: float = 2.0,
                                      q: int = 0):
    q_enc, q_dec, scales, means = net.separate_prior(params)
    common_params = _apply_stage_q_condition(net.y_spatial_prior_reduction(params), q_shift)
    dtype = y.dtype
    device = y.device
    b, c, h, w = y.size()
    masks = net.get_mask_four_parts(b, c, h, w, dtype, device)
    y_scaled = y * q_enc
    y_hat_so_far = None
    streams: List[ArithmeticStream] = []
    suppressed_stages = _normalize_stage_set(suppress_yq_stages)

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
        if part_idx in suppressed_stages:
            y_q = _learned_or_omitted_residual_values(
                net, common_params, y_hat_so_far, cur_scales, mask, part_idx, q,
                omitted_residual_mode, omitted_residual_scale, omitted_residual_clip)
        else:
            y_q = torch.round(y_res)
        _append_stage_stream_or_omit(
            streams, y_q, cur_scales, mask, part_idx, suppressed_stages,
            entropy_family, entropy_scale_factor)
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
    suppress_yq_stages: Optional[Sequence[int]] = None,
    omitted_residual_mode: str = "zero",
    omitted_residual_scale: float = 0.0,
    omitted_residual_clip: float = 2.0,
    suppress_rho_threshold: float = 0.0,
    synth_yq_stages: Optional[Sequence[int]] = None,
    synth_rho_threshold: float = 0.0,
    synth_value_scale: float = 1.0,
    q: int = 0,
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
    suppressed_stages = _normalize_stage_set(suppress_yq_stages)
    synth_stages = _normalize_stage_set(synth_yq_stages)

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
        suppress_mask = _stage_suppression_mask(
            mask, rho, part_idx, suppressed_stages, suppress_rho_threshold)
        y_q = torch.round(y_res)
        if suppress_mask.any():
            omitted = _learned_or_omitted_residual_values(
                net, common_params, y_hat_so_far, cur_scales, suppress_mask.to(dtype=mask.dtype),
                part_idx, q, omitted_residual_mode, omitted_residual_scale, omitted_residual_clip)
            y_q = torch.where(suppress_mask, omitted.to(dtype=y_q.dtype), y_q)
        _append_symbols_with_suppression(
            streams, y_q, cur_scales, mask, suppress_mask,
            entropy_family, entropy_scale_factor)
        synth_mask = _stage_suppression_mask(
            mask, rho, part_idx, synth_stages, synth_rho_threshold)
        synth_add = 0.0
        if synth_mask.any():
            synth_add = _learned_value_addition(
                net, common_params, y_hat_so_far, synth_mask.to(dtype=mask.dtype),
                part_idx, float(synth_value_scale))
        y_hat_part = y_q + means_hat + synth_add
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


def _make_rdo_protection_control_maps(
    net,
    y: torch.Tensor,
    params: torch.Tensor,
    z_shape: Tuple[int, int],
    predictor_delta_bound: float,
    q_shift: Optional[torch.Tensor],
    topk_frac: float,
) -> torch.Tensor:
    """Build encoder-side protection tokens from local residual RDO benefit.

    The binary token means "protect this stage/cell from decoder-only
    coarsening".  It is not decoder-computable, so it must be entropy-coded and
    counted.  Scores compare the local latent error under the current rho
    against the error when rho is forced back to 1.0, using only the source-side
    latent and the same four-part context order used by the codec.
    """
    frac = float(topk_frac)
    quant_step, scales, means = params.chunk(3, 1)
    quant_step = quant_step.clamp_min(0.5)
    common_params = _apply_stage_q_condition(net.y_spatial_prior_reduction(params), q_shift)
    dtype = y.dtype
    device = y.device
    b, c, h, w = y.size()
    if b != 1:
        raise ValueError("real codec currently expects batch size 1")
    masks = net.get_mask_four_parts(b, c, h, w, dtype, device)
    control_maps = torch.zeros((1, 4, int(z_shape[0]), int(z_shape[1])), dtype=dtype, device=device)
    y_hat_so_far = None

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

        means_hat = (cur_means + delta) * mask
        q_enc_base = 1.0 / (quant_step * rho)
        q_enc_protect = 1.0 / quant_step
        y_res_base = (y * q_enc_base - means_hat) * mask
        y_res_protect = (y * q_enc_protect - means_hat) * mask
        y_q_base = torch.round(y_res_base)
        y_q_protect = torch.round(y_res_protect)
        rec_base = (y_q_base + means_hat) * (quant_step * rho) * mask
        rec_protect = (y_q_protect + means_hat) * quant_step * mask
        benefit = ((rec_base - y).square() - (rec_protect - y).square()).clamp_min(0.0) * mask
        benefit = benefit.mean(dim=1, keepdim=True)
        pooled = F.adaptive_avg_pool2d(benefit, z_shape)
        symbols = torch.zeros_like(pooled)
        if frac > 0.0 and float(pooled.max().item()) > 0.0:
            flat = pooled.flatten(1)
            k = int(round(max(0.0, min(frac, 1.0)) * flat.shape[1]))
            if k > 0:
                k = min(k, flat.shape[1])
                idx = torch.topk(flat, k=k, dim=1).indices
                hard = torch.zeros_like(flat)
                hard.scatter_(1, idx, 1.0)
                symbols = hard.reshape_as(pooled)
        control_maps[:, part_idx:part_idx + 1] = symbols

        ctrl_y = F.interpolate(symbols, size=(h, w), mode="nearest").to(dtype=dtype)
        rho_eff = 1.0 + (rho - 1.0) * (1.0 - ctrl_y)
        y_res = (y / (quant_step * rho_eff) - means_hat) * mask
        y_q = torch.round(y_res)
        y_hat_part = y_q + means_hat
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part

    return control_maps


def _make_stage3_send_control_map(
    net,
    x_padded: torch.Tensor,
    y: torch.Tensor,
    params: torch.Tensor,
    z_shape: Tuple[int, int],
    predictor_delta_bound: float,
    q_shift: Optional[torch.Tensor],
    send_frac: float,
    score_mode: str = "latent_mse",
    grad_max_side: int = 256,
) -> torch.Tensor:
    """Select stage-3 residual cells to transmit under a counted send mask."""
    frac = float(send_frac)
    score_mode = str(score_mode)
    quant_step, scales, means = params.chunk(3, 1)
    quant_step = quant_step.clamp_min(0.5)
    common_params = _apply_stage_q_condition(net.y_spatial_prior_reduction(params), q_shift)
    dtype = y.dtype
    device = y.device
    b, c, h, w = y.size()
    if b != 1:
        raise ValueError("real codec currently expects batch size 1")
    masks = net.get_mask_four_parts(b, c, h, w, dtype, device)
    y_hat_so_far = None
    q_dec_map_so_far = torch.zeros_like(quant_step)

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

        q_dec = quant_step * rho
        means_hat = (cur_means + delta) * mask
        y_res = (y / q_dec - means_hat) * mask
        y_q = torch.round(y_res)

        if part_idx == 3:
            if score_mode == "latent_mse":
                rec_send = (y_q + means_hat) * q_dec * mask
                rec_drop = means_hat * q_dec * mask
                benefit = ((rec_drop - y).square() - (rec_send - y).square()).clamp_min(0.0) * mask
                benefit = benefit.mean(dim=1, keepdim=True)
                pooled = F.adaptive_avg_pool2d(benefit, z_shape)
            elif score_mode in {"image_mse_grad", "image_l1_grad", "image_mse_grad_abs"}:
                # Encoder-side teacher: rank residual symbols by first-order
                # image-loss change from the omitted-residual baseline.  The
                # selected binary mask is transmitted, so using source pixels
                # at encode time does not introduce hidden side information.
                q_dec_map = q_dec_map_so_far + q_dec * mask
                y_prefix = torch.zeros_like(y) if y_hat_so_far is None else y_hat_so_far.detach()
                y_q_candidate = y_q.detach()
                means_stage = means_hat.detach()
                mask_stage = mask.detach()
                x_ref = x_padded.detach()
                with torch.enable_grad():
                    probe = torch.zeros_like(y_q_candidate, requires_grad=True)
                    y_hat_stage = probe * mask_stage + means_stage
                    latent = (y_prefix + y_hat_stage) * q_dec_map.detach()
                    x_hat = net.vqgan.generator(latent)
                    max_side = int(grad_max_side)
                    if max_side > 0:
                        _, _, hh, ww = x_hat.shape
                        side = max(hh, ww)
                        if side > max_side:
                            scale = float(max_side) / float(side)
                            out_hw = (max(1, int(round(hh * scale))), max(1, int(round(ww * scale))))
                            x_hat_loss = F.interpolate(x_hat, size=out_hw, mode="bilinear", align_corners=False)
                            x_ref_loss = F.interpolate(x_ref, size=out_hw, mode="bilinear", align_corners=False)
                        else:
                            x_hat_loss = x_hat
                            x_ref_loss = x_ref
                    else:
                        x_hat_loss = x_hat
                        x_ref_loss = x_ref
                    if score_mode == "image_l1_grad":
                        loss = F.l1_loss(x_hat_loss, x_ref_loss)
                    else:
                        loss = F.mse_loss(x_hat_loss, x_ref_loss)
                    grad = torch.autograd.grad(loss, probe, retain_graph=False, create_graph=False)[0]
                first_order_delta = grad.detach() * y_q_candidate
                if score_mode == "image_mse_grad_abs":
                    benefit = first_order_delta.abs() * mask_stage
                else:
                    benefit = (-first_order_delta).clamp_min(0.0) * mask_stage
                benefit = benefit.mean(dim=1, keepdim=True)
                pooled = F.adaptive_avg_pool2d(benefit, z_shape)
            else:
                raise ValueError(f"unknown stage3 send score mode: {score_mode}")
            send_map = torch.zeros_like(pooled)
            if frac > 0.0 and float(pooled.max().item()) > 0.0:
                flat = pooled.flatten(1)
                k = int(round(max(0.0, min(frac, 1.0)) * flat.shape[1]))
                if k > 0:
                    k = min(k, flat.shape[1])
                    idx = torch.topk(flat, k=k, dim=1).indices
                    hard = torch.zeros_like(flat)
                    hard.scatter_(1, idx, 1.0)
                    send_map = hard.reshape_as(pooled)
            return send_map.to(dtype=dtype)

        y_hat_part = y_q + means_hat
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part
        q_dec_map_so_far = q_dec_map_so_far + q_dec * mask

    raise RuntimeError("stage 3 was not reached")


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
    suppress_yq_stages: Optional[Sequence[int]] = None,
    omitted_residual_mode: str = "zero",
    omitted_residual_scale: float = 0.0,
    omitted_residual_clip: float = 2.0,
    suppress_rho_threshold: float = 0.0,
    synth_yq_stages: Optional[Sequence[int]] = None,
    synth_rho_threshold: float = 0.0,
    synth_value_scale: float = 1.0,
    send_stage3_map: Optional[torch.Tensor] = None,
    q: int = 0,
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
    suppressed_stages = _normalize_stage_set(suppress_yq_stages)
    synth_stages = _normalize_stage_set(synth_yq_stages)

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
        suppress_mask = _stage_suppression_mask(
            mask, rho, part_idx, suppressed_stages, suppress_rho_threshold)
        if send_stage3_map is not None and part_idx == 3:
            send_y = F.interpolate(
                send_stage3_map.to(device=device, dtype=mask.dtype),
                size=(h, w),
                mode="nearest",
            ) > 0.5
            suppress_mask = suppress_mask | ((mask > 0.5) & (~send_y))
        y_q = torch.round(y_res)
        if suppress_mask.any():
            omitted = _learned_or_omitted_residual_values(
                net, common_params, y_hat_so_far, cur_scales, suppress_mask.to(dtype=mask.dtype),
                part_idx, q, omitted_residual_mode, omitted_residual_scale, omitted_residual_clip)
            y_q = torch.where(suppress_mask, omitted.to(dtype=y_q.dtype), y_q)
        _append_symbols_with_suppression(
            streams, y_q, cur_scales, mask, suppress_mask,
            entropy_family, entropy_scale_factor)
        synth_mask = _stage_suppression_mask(
            mask, rho, part_idx, synth_stages, synth_rho_threshold)
        synth_add = 0.0
        if synth_mask.any():
            synth_add = _learned_value_addition(
                net, common_params, y_hat_so_far, synth_mask.to(dtype=mask.dtype),
                part_idx, float(synth_value_scale))
        y_hat_part = y_q + means_hat + synth_add
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part

    return streams, y_hat_so_far * q_dec_map


def _decode_y_four_part_stage_residual(net, payload: RealCodecPayload, params: torch.Tensor,
                                      predictor_delta_bound: float = 0.0,
                                      q_shift: Optional[torch.Tensor] = None,
                                      entropy_family: str = "gaussian",
                                      entropy_scale_factor: float = 1.0,
                                      suppress_yq_stages: Optional[Sequence[int]] = None,
                                      omitted_residual_mode: str = "zero",
                                      omitted_residual_scale: float = 0.0,
                                      omitted_residual_clip: float = 2.0):
    q_enc, q_dec, scales, means = net.separate_prior(params)
    del q_enc
    common_params = _apply_stage_q_condition(net.y_spatial_prior_reduction(params), q_shift)
    device = params.device
    dtype = params.dtype
    c, h, w = payload.y_shape
    masks = net.get_mask_four_parts(1, c, h, w, dtype, device)
    y_hat_so_far = None
    suppressed_stages = _normalize_stage_set(suppress_yq_stages)

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

        suppress_mask = _stage_suppression_mask(
            mask, rho, part_idx, suppressed_stages, suppress_rho_threshold)
        active = (mask > 0.5) & (~suppress_mask)
        y_q_part = torch.zeros((1, c, h, w), dtype=dtype, device=device)
        if suppress_mask.any():
            y_q_part = _learned_or_omitted_residual_values(
                net, common_params, y_hat_so_far, cur_scales, suppress_mask.to(dtype=mask.dtype),
                part_idx, payload.q, omitted_residual_mode, omitted_residual_scale, omitted_residual_clip
            ).to(device=device, dtype=dtype)
        if active.any():
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
    suppress_yq_stages: Optional[Sequence[int]] = None,
    omitted_residual_mode: str = "zero",
    omitted_residual_scale: float = 0.0,
    omitted_residual_clip: float = 2.0,
    suppress_rho_threshold: float = 0.0,
    synth_yq_stages: Optional[Sequence[int]] = None,
    synth_rho_threshold: float = 0.0,
    synth_value_scale: float = 1.0,
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
    suppressed_stages = _normalize_stage_set(suppress_yq_stages)

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

        suppress_mask = _stage_suppression_mask(
            mask, rho, part_idx, suppressed_stages, suppress_rho_threshold)
        active = (mask > 0.5) & (~suppress_mask)
        y_q_part = torch.zeros((1, c, h, w), dtype=dtype, device=device)
        if suppress_mask.any():
            y_q_part = _learned_or_omitted_residual_values(
                net, common_params, y_hat_so_far, cur_scales, suppress_mask.to(dtype=mask.dtype),
                part_idx, payload.q, omitted_residual_mode, omitted_residual_scale, omitted_residual_clip
            ).to(device=device, dtype=dtype)
        if active.any():
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
    suppress_yq_stages: Optional[Sequence[int]] = None,
    omitted_residual_mode: str = "zero",
    omitted_residual_scale: float = 0.0,
    omitted_residual_clip: float = 2.0,
    suppress_rho_threshold: float = 0.0,
    synth_yq_stages: Optional[Sequence[int]] = None,
    synth_rho_threshold: float = 0.0,
    synth_value_scale: float = 1.0,
    send_stage3_map: Optional[torch.Tensor] = None,
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
    suppressed_stages = _normalize_stage_set(suppress_yq_stages)
    synth_stages = _normalize_stage_set(synth_yq_stages)

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

        suppress_mask = _stage_suppression_mask(
            mask, rho, part_idx, suppressed_stages, suppress_rho_threshold)
        if send_stage3_map is not None and part_idx == 3:
            send_y = F.interpolate(
                send_stage3_map.to(device=device, dtype=mask.dtype),
                size=(h, w),
                mode="nearest",
            ) > 0.5
            suppress_mask = suppress_mask | ((mask > 0.5) & (~send_y))
        active = (mask > 0.5) & (~suppress_mask)
        y_q_part = torch.zeros((1, c, h, w), dtype=dtype, device=device)
        if suppress_mask.any():
            y_q_part = _learned_or_omitted_residual_values(
                net, common_params, y_hat_so_far, cur_scales, suppress_mask.to(dtype=mask.dtype),
                part_idx, payload.q, omitted_residual_mode, omitted_residual_scale, omitted_residual_clip
            ).to(device=device, dtype=dtype)
        if active.any():
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
        synth_mask = _stage_suppression_mask(
            mask, rho, part_idx, synth_stages, synth_rho_threshold)
        synth_add = 0.0
        if synth_mask.any():
            synth_add = _learned_value_addition(
                net, common_params, y_hat_so_far, synth_mask.to(dtype=mask.dtype),
                part_idx, float(synth_value_scale))
        y_hat_part = y_q_part + (cur_means + delta + mean_control) * mask + synth_add
        y_hat_so_far = y_hat_part if y_hat_so_far is None else y_hat_so_far + y_hat_part

    return y_hat_so_far * q_dec_map


def _decode_y_four_part(
    net,
    payload: RealCodecPayload,
    params: torch.Tensor,
    latent_mean_scaled: Optional[torch.Tensor] = None,
    entropy_family: str = "gaussian",
    entropy_scale_factor: float = 1.0,
    suppress_yq_stages: Optional[Sequence[int]] = None,
    omitted_residual_mode: str = "zero",
    omitted_residual_scale: float = 0.0,
    omitted_residual_clip: float = 2.0,
):
    q_enc, q_dec, scales, means = net.separate_prior(params)
    del q_enc
    common_params = net.y_spatial_prior_reduction(params)
    device = params.device
    dtype = params.dtype
    c, h, w = payload.y_shape
    masks = net.get_mask_four_parts(1, c, h, w, dtype, device)
    y_hat_so_far = None
    suppressed_stages = _normalize_stage_set(suppress_yq_stages)

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
        if part_idx in suppressed_stages:
            y_q_part = _learned_or_omitted_residual_values(
                net, common_params, y_hat_so_far, cur_scales, mask, part_idx, payload.q,
                omitted_residual_mode, omitted_residual_scale, omitted_residual_clip
            ).to(device=device, dtype=dtype)
        else:
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
    suppress_yq_stages: Optional[Sequence[int]] = None,
    omitted_residual_mode: str = "zero",
    omitted_residual_scale: float = 0.0,
    omitted_residual_clip: float = 2.0,
    suppress_rho_threshold: float = 0.0,
    synth_yq_stages: Optional[Sequence[int]] = None,
    synth_rho_threshold: float = 0.0,
    synth_value_scale: float = 1.0,
    stage3_send_score_mode: str = "latent_mse",
    stage3_send_grad_max_side: int = 256,
    z_entropy_mode: str = "fixed",
    z_entropy_cdf_path: Optional[str] = None,
) -> Tuple[bytes, Dict[str, float]]:
    """Encode a padded image tensor and return serialized payload bytes."""
    if entropy_family not in ENTROPY_FAMILIES:
        raise ValueError(f"unknown entropy family: {entropy_family}")
    if omitted_residual_mode not in OMITTED_RESIDUAL_MODES:
        raise ValueError(f"unknown omitted residual mode: {omitted_residual_mode}")
    if z_entropy_mode not in Z_ENTROPY_MODES:
        raise ValueError(f"unknown z_entropy_mode: {z_entropy_mode}")
    entropy_scale_factor = max(float(entropy_scale_factor), 1e-6)
    suppressed_stages = _normalize_stage_set(suppress_yq_stages)
    synth_stages = _normalize_stage_set(synth_yq_stages)
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
            suppress_yq_stages=suppressed_stages,
            omitted_residual_mode=omitted_residual_mode,
            omitted_residual_scale=omitted_residual_scale,
            omitted_residual_clip=omitted_residual_clip,
            suppress_rho_threshold=suppress_rho_threshold,
            q=q,
        )
        streams = control_streams + y_streams
    elif predictor_param_mode == "stage_residual_entropy_quant_gate_stage3_send_control":
        send_map = _make_stage3_send_control_map(
            net, x_padded, y, params, (int(z.shape[2]), int(z.shape[3])),
            predictor_delta_bound=predictor_delta_bound,
            q_shift=q_shift,
            send_frac=residual_control_topk_frac,
            score_mode=stage3_send_score_mode,
            grad_max_side=stage3_send_grad_max_side,
        )
        control_streams = [_encode_stage3_send_stream(send_map, residual_control_prob_nonzero)]
        y_streams, _ = _encode_y_four_part_stage_residual_quant_gate(
            net, y, params, predictor_delta_bound,
            q_shift=q_shift,
            entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor,
            suppress_yq_stages=suppressed_stages,
            omitted_residual_mode=omitted_residual_mode,
            omitted_residual_scale=omitted_residual_scale,
            omitted_residual_clip=omitted_residual_clip,
            suppress_rho_threshold=suppress_rho_threshold,
            synth_yq_stages=synth_stages,
            synth_rho_threshold=synth_rho_threshold,
            synth_value_scale=synth_value_scale,
            send_stage3_map=send_map,
            q=q,
        )
        streams = control_streams + y_streams
    elif predictor_param_mode == "stage_residual_entropy_quant_gate_latent_control":
        y_streams, y_hat_base = _encode_y_four_part_stage_residual_quant_gate(
            net, y, params, predictor_delta_bound, q_shift=q_shift,
            entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor,
            suppress_yq_stages=suppressed_stages,
            omitted_residual_mode=omitted_residual_mode,
            omitted_residual_scale=omitted_residual_scale,
            omitted_residual_clip=omitted_residual_clip,
            suppress_rho_threshold=suppress_rho_threshold,
            synth_yq_stages=synth_stages,
            synth_rho_threshold=synth_rho_threshold,
            synth_value_scale=synth_value_scale,
            q=q)
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
        common_for_control = _apply_stage_q_condition(net.y_spatial_prior_reduction(params), q_shift)
        if hasattr(net, "tiny_control_encoder") and net.tiny_control_encoder is not None:
            control_symbols, _, _ = net.tiny_control_encoder(
                y, common_for_control, (int(z.shape[2]), int(z.shape[3])), int(q))
            control_symbols = (control_symbols > 0.5).to(dtype=y.dtype)
        elif float(residual_control_topk_frac) > 0.0:
            control_symbols = _make_rdo_protection_control_maps(
                net, y, params, (int(z.shape[2]), int(z.shape[3])),
                predictor_delta_bound=predictor_delta_bound,
                q_shift=q_shift,
                topk_frac=residual_control_topk_frac,
            )
            control_prob_one = float(residual_control_prob_nonzero)
        else:
            raise ValueError(
                f"{predictor_param_mode} requires net.tiny_control_encoder or "
                "--residual_control_topk_frac > 0 for source-side RDO control")
        control_streams = _encode_control_streams(control_symbols, control_prob_one)
        y_streams, _ = _encode_y_four_part_stage_residual_quant_gate(
            net, y, params, predictor_delta_bound, control_maps=control_symbols,
            q_shift=q_shift, entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor,
            suppress_yq_stages=suppressed_stages,
            omitted_residual_mode=omitted_residual_mode,
            omitted_residual_scale=omitted_residual_scale,
            omitted_residual_clip=omitted_residual_clip,
            suppress_rho_threshold=suppress_rho_threshold,
            synth_yq_stages=synth_stages,
            synth_rho_threshold=synth_rho_threshold,
            synth_value_scale=synth_value_scale,
            q=q)
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
            entropy_scale_factor=entropy_scale_factor,
            suppress_yq_stages=suppressed_stages,
            omitted_residual_mode=omitted_residual_mode,
            omitted_residual_scale=omitted_residual_scale,
            omitted_residual_clip=omitted_residual_clip,
            suppress_rho_threshold=suppress_rho_threshold,
            synth_yq_stages=synth_stages,
            synth_rho_threshold=synth_rho_threshold,
            synth_value_scale=synth_value_scale,
            q=q)
        streams = y_streams
    elif predictor_param_mode == "stage_quant_gate":
        y_streams, _ = _encode_y_four_part_stage_quant_gate(
            net, y, params, q_shift=q_shift, entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor,
            suppress_yq_stages=suppressed_stages,
            omitted_residual_mode=omitted_residual_mode,
            omitted_residual_scale=omitted_residual_scale,
            omitted_residual_clip=omitted_residual_clip,
            suppress_rho_threshold=suppress_rho_threshold,
            q=q)
        streams = y_streams
    elif predictor_param_mode == "stage_latent_residual":
        y_streams, _ = _encode_y_four_part_stage_residual(
            net, y, params, predictor_delta_bound, q_shift=q_shift,
            entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor,
            suppress_yq_stages=suppressed_stages,
            omitted_residual_mode=omitted_residual_mode,
            omitted_residual_scale=omitted_residual_scale,
            omitted_residual_clip=omitted_residual_clip,
            q=q)
        streams = y_streams
    else:
        y_streams, _ = _encode_y_four_part(
            net, y, params, latent_mean_scaled, entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor,
            suppress_yq_stages=suppressed_stages,
            omitted_residual_mode=omitted_residual_mode,
            omitted_residual_scale=omitted_residual_scale,
            omitted_residual_clip=omitted_residual_clip,
            q=q)
        streams = y_streams

    z_data = _encode_z_indices(
        z_indices, net.codebook_size, z_entropy_mode, q, z_entropy_cdf_path)
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
        "y_stage_bytes": [float(len(s.data)) for s in y_streams],
        "control_bytes": float(control_bytes),
        "y_stream_count": float(len(streams)),
        "entropy_family": entropy_family,
        "entropy_scale_factor": float(entropy_scale_factor),
        "suppress_yq_stages": sorted(suppressed_stages),
        "omitted_residual_mode": omitted_residual_mode,
        "omitted_residual_scale": float(omitted_residual_scale),
        "omitted_residual_clip": float(omitted_residual_clip),
        "suppress_rho_threshold": float(suppress_rho_threshold),
        "synth_yq_stages": sorted(synth_stages),
        "synth_rho_threshold": float(synth_rho_threshold),
        "synth_value_scale": float(synth_value_scale),
        "stage3_send_score_mode": str(stage3_send_score_mode),
        "stage3_send_grad_max_side": float(stage3_send_grad_max_side),
        "residual_control_nonzero": float(
            (control_symbols != 0).sum().item()) if "control_symbols" in locals() else 0.0,
        "residual_control_symbol_count": float(
            control_symbols.numel()) if "control_symbols" in locals() else 0.0,
        "z_entropy_mode": z_entropy_mode,
        "z_entropy_cdf_path": str(z_entropy_cdf_path or ""),
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
    suppress_yq_stages: Optional[Sequence[int]] = None,
    omitted_residual_mode: str = "zero",
    omitted_residual_scale: float = 0.0,
    omitted_residual_clip: float = 2.0,
    suppress_rho_threshold: float = 0.0,
    synth_yq_stages: Optional[Sequence[int]] = None,
    synth_rho_threshold: float = 0.0,
    synth_value_scale: float = 1.0,
    z_entropy_mode: str = "fixed",
    z_entropy_cdf_path: Optional[str] = None,
) -> torch.Tensor:
    """Decode a serialized payload and return padded reconstruction in [-1, 1]."""
    if entropy_family not in ENTROPY_FAMILIES:
        raise ValueError(f"unknown entropy family: {entropy_family}")
    if omitted_residual_mode not in OMITTED_RESIDUAL_MODES:
        raise ValueError(f"unknown omitted residual mode: {omitted_residual_mode}")
    if z_entropy_mode not in Z_ENTROPY_MODES:
        raise ValueError(f"unknown z_entropy_mode: {z_entropy_mode}")
    entropy_scale_factor = max(float(entropy_scale_factor), 1e-6)
    suppressed_stages = _normalize_stage_set(suppress_yq_stages)
    synth_stages = _normalize_stage_set(synth_yq_stages)
    payload = RealCodecPayload.from_bytes(payload_bytes)
    device = next(net.parameters()).device
    z_indices = _decode_z_indices(
        payload.z_data, payload.z_count, net.codebook_size, payload.q, device,
        z_entropy_mode, z_entropy_cdf_path)
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
            entropy_scale_factor=entropy_scale_factor,
            suppress_yq_stages=suppressed_stages,
            omitted_residual_mode=omitted_residual_mode,
            omitted_residual_scale=omitted_residual_scale,
            omitted_residual_clip=omitted_residual_clip,
            suppress_rho_threshold=suppress_rho_threshold,
            synth_yq_stages=synth_stages,
            synth_rho_threshold=synth_rho_threshold,
            synth_value_scale=synth_value_scale)
    elif predictor_param_mode == "stage_residual_entropy_quant_gate_stage3_send_control":
        if len(payload.y_streams) < 5:
            raise ValueError(
                f"{predictor_param_mode} expects 1 stage3-send control + 4 y streams, got {len(payload.y_streams)}")
        send_map = _decode_stage3_send_stream(
            payload.y_streams[0], payload.z_shape, residual_control_prob_nonzero, device)
        y_hat = _decode_y_four_part_stage_residual_quant_gate(
            net, payload, params, predictor_delta_bound,
            streams=payload.y_streams[1:],
            q_shift=q_shift,
            entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor,
            suppress_yq_stages=suppressed_stages,
            omitted_residual_mode=omitted_residual_mode,
            omitted_residual_scale=omitted_residual_scale,
            omitted_residual_clip=omitted_residual_clip,
            suppress_rho_threshold=suppress_rho_threshold,
            synth_yq_stages=synth_stages,
            synth_rho_threshold=synth_rho_threshold,
            synth_value_scale=synth_value_scale,
            send_stage3_map=send_map)
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
            entropy_scale_factor=entropy_scale_factor,
            suppress_yq_stages=suppressed_stages,
            omitted_residual_mode=omitted_residual_mode,
            omitted_residual_scale=omitted_residual_scale,
            omitted_residual_clip=omitted_residual_clip,
            suppress_rho_threshold=suppress_rho_threshold,
            synth_yq_stages=synth_stages,
            synth_rho_threshold=synth_rho_threshold,
            synth_value_scale=synth_value_scale)
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
            entropy_scale_factor=entropy_scale_factor,
            suppress_yq_stages=suppressed_stages,
            omitted_residual_mode=omitted_residual_mode,
            omitted_residual_scale=omitted_residual_scale,
            omitted_residual_clip=omitted_residual_clip,
            suppress_rho_threshold=suppress_rho_threshold,
            synth_yq_stages=synth_stages,
            synth_rho_threshold=synth_rho_threshold,
            synth_value_scale=synth_value_scale)
    elif predictor_param_mode in {
        "stage_residual_quant_gate",
        "stage_residual_entropy_quant_gate",
        "stage_residual_entropy_quant_gate_scale_calib",
        "stage_residual_entropy_quant_gate_residual_refiner",
    }:
        y_hat = _decode_y_four_part_stage_residual_quant_gate(
            net, payload, params, predictor_delta_bound, q_shift=q_shift,
            entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor,
            suppress_yq_stages=suppressed_stages,
            omitted_residual_mode=omitted_residual_mode,
            omitted_residual_scale=omitted_residual_scale,
            omitted_residual_clip=omitted_residual_clip,
            suppress_rho_threshold=suppress_rho_threshold,
            synth_yq_stages=synth_stages,
            synth_rho_threshold=synth_rho_threshold,
            synth_value_scale=synth_value_scale)
    elif predictor_param_mode == "stage_quant_gate":
        y_hat = _decode_y_four_part_stage_quant_gate(
            net, payload, params, q_shift=q_shift, entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor,
            suppress_yq_stages=suppressed_stages,
            omitted_residual_mode=omitted_residual_mode,
            omitted_residual_scale=omitted_residual_scale,
            omitted_residual_clip=omitted_residual_clip,
            suppress_rho_threshold=suppress_rho_threshold)
    elif predictor_param_mode == "stage_latent_residual":
        y_hat = _decode_y_four_part_stage_residual(
            net, payload, params, predictor_delta_bound, q_shift=q_shift,
            entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor,
            suppress_yq_stages=suppressed_stages,
            omitted_residual_mode=omitted_residual_mode,
            omitted_residual_scale=omitted_residual_scale,
            omitted_residual_clip=omitted_residual_clip)
    else:
        y_hat = _decode_y_four_part(
            net, payload, params, latent_mean_scaled, entropy_family=entropy_family,
            entropy_scale_factor=entropy_scale_factor,
            suppress_yq_stages=suppressed_stages,
            omitted_residual_mode=omitted_residual_mode,
            omitted_residual_scale=omitted_residual_scale,
            omitted_residual_clip=omitted_residual_clip)
    curr_q_dec = net.q_dec[payload.q:payload.q + 1]
    latent = net.dec(y_hat, curr_q_dec)
    return net.vqgan.generator(latent)


def crop_to_original(x_hat_padded: torch.Tensor, orig_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = orig_hw
    return x_hat_padded[..., :h, :w]
