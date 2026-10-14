"""Independent CPU/Torch reference for padded spectral correlation and pair PWS.

Explicitly selected by cc_backend=cpu or torch. Never called by cpp_cuda.
Preprocessing and whitening are evaluated in float64; shared spectra use
complex64 and CCFs float32, as in the native implementation. No CLI entry point.
"""

from __future__ import annotations

import numpy as np
from scipy import fft
from scipy.fftpack import next_fast_len
from scipy.ndimage import uniform_filter1d


def fir_taps(factor):
    k = np.arange(-20 * factor, 20 * factor + 1, dtype=np.float64)
    taps = (
        (0.8 / factor)
        * np.sinc(0.8 * k / factor)
        * np.i0(8.6 * np.sqrt(np.maximum(0, 1 - (k / (20 * factor)) ** 2)))
        / np.i0(8.6)
    )
    return (taps / taps.sum()).astype(np.float32)


class Ops:
    def __init__(self, backend, device):
        self.backend, self.device = backend, device
        self.torch = None
        if backend == "torch":
            import torch

            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("Torch CUDA requested but unavailable")
            self.torch = torch

    def array(self, value, kind="float32"):
        if self.torch:
            return self.torch.as_tensor(
                value, device=self.device, dtype=getattr(self.torch, kind)
            )
        return np.asarray(value, dtype=kind)

    def cast(self, value, kind):
        return value.to(getattr(self.torch, kind)) if self.torch else value.astype(kind)

    def host(self, value):
        return value.detach().cpu().numpy() if self.torch else np.asarray(value)

    def rfft(self, x, n):
        return self.torch.fft.rfft(x, n=n) if self.torch else fft.rfft(x, n=n)

    def irfft(self, x, n):
        return self.torch.fft.irfft(x, n=n) if self.torch else fft.irfft(x, n=n)

    def ifft(self, x, n):
        return self.torch.fft.ifft(x, n=n) if self.torch else fft.ifft(x, n=n)

    def average(self, x, width):
        width = min(width, x.shape[-1])
        if not self.torch:
            return uniform_filter1d(x, size=width, axis=-1, mode="nearest")
        t = self.torch
        indices = t.arange(x.shape[-1], device=self.device)
        return sum(
            x[..., (indices + k - width // 2).clamp(0, x.shape[-1] - 1)] / width
            for k in range(width)
        )

    def where(self, mask, a, b):
        return self.torch.where(mask, a, b) if self.torch else np.where(mask, a, b)

    def maximum(self, x, axis=-1):
        return (
            self.torch.amax(x, dim=axis, keepdim=True)
            if self.torch
            else np.max(x, axis=axis, keepdims=True)
        )

    def mean(self, x):
        return (
            x.mean(dim=-1, keepdim=True)
            if self.torch
            else x.mean(axis=-1, keepdims=True)
        )

    def all(self, x):
        return x.all(dim=-1) if self.torch else x.all(axis=-1)


def preprocess(windows, p, ops):
    """Return zero-padded whitened spectra and window×station quality mask."""
    x = ops.cast(windows, "float64")
    n = x.shape[-1]
    pre_fft = next_fast_len(n)
    grid = np.arange(n) / n - (n - 1) / (2 * n)
    centered = ops.array(grid, "float64")
    x = x - ops.mean(x)
    slope = ops.mean(x * centered) / np.mean(grid**2)
    x = x - slope * centered
    edge = min(20, int(np.floor(n * 0.05 + 0.5)))
    taper = np.ones(n)
    if edge:
        taper[:edge] = 0.5 - 0.5 * np.cos(np.pi * np.arange(edge) / edge)
        taper[-edge:] = taper[:edge][::-1]
    x = x * ops.array(taper, "float64")
    fs = p["samp_freq"] / p["time_downsample"]
    freq = np.fft.rfftfreq(pre_fft, 1 / fs)
    lo, hi = p["freqmin"], p["freqmax"]
    shoulder = np.zeros_like(freq)
    shoulder[(freq >= lo) & (freq <= hi)] = 1
    if lo:
        selected = (freq >= 0.5 * lo) & (freq < lo)
        shoulder[selected] = 0.5 * (
            1 - np.cos(np.pi * (freq[selected] - 0.5 * lo) / (0.5 * lo))
        )
    selected = (freq > hi) & (freq <= 1.1 * hi)
    shoulder[selected] = 0.5 * (1 + np.cos(np.pi * (freq[selected] - hi) / (0.1 * hi)))
    x = ops.irfft(ops.rfft(x, pre_fft) * ops.array(shoulder, "float64"), pre_fft)[
        ..., :n
    ]
    variance = ops.mean(x * x) - ops.mean(x) ** 2
    if ops.torch:
        std = ops.torch.sqrt(variance.clamp(min=0))
    else:
        std = np.sqrt(np.maximum(variance, 0))
    ratio = ops.maximum(abs(x)) / ops.where(
        std > np.finfo(float).eps, std, ops.array(np.finfo(float).eps, "float64")
    )
    quality = ((ratio > 0) & (ratio < p["max_over_std"]))[..., 0]
    if p["time_norm"] == "one_bit":
        x = ops.torch.sign(x) if ops.torch else np.sign(x)
    elif p["time_norm"] == "rma":
        amplitude = ops.average(abs(x), p["smooth_N"])
        x = x / ops.where(amplitude == 0, ops.array(1, "float64"), amplitude)
    correlation_fft = next_fast_len(2 * n - 1)
    freq = np.fft.rfftfreq(correlation_fft, 1 / fs)
    spectrum = ops.rfft(x, correlation_fft)
    if p["freq_norm"] != "no":
        # Match the reused native RFFT whitening operator explicitly.
        nf = correlation_fft // 2 + 1
        a = min(nf - 1, int(np.floor(lo * correlation_fft / fs + 0.5)))
        b = min(nf - 1, int(np.floor(hi * correlation_fft / fs + 0.5)))
        start, stop = max(0, a - 100), min(nf, b + 100)
        band = spectrum[..., start:stop]
        amplitude = abs(band)
        if p["freq_norm"] == "rma":
            amplitude = ops.average(amplitude, p["smoothspect_N"])
        normalized = band / ops.where(
            amplitude == 0, ops.array(1, "float64"), amplitude
        )
        if p["freq_norm"] == "phase_only":
            normalized = ops.where(
                amplitude == 0, ops.array(1, "complex128"), normalized
            )
        weights = np.ones(stop - start)
        if a > start:
            weights[: a - start] = (
                np.cos(
                    np.pi / 2 + np.pi / 2 * np.arange(a - start) / max(a - start - 1, 1)
                )
                ** 2
            )
        if stop > b:
            weights[b - start :] = (
                np.cos(np.pi / 2 * np.arange(stop - b) / max(stop - b - 1, 1)) ** 2
            )
        support = (freq[start:stop] >= 0.5 * lo) & (freq[start:stop] <= 1.1 * hi)
        spectrum = spectrum * 0
        spectrum[..., start:stop] = normalized * ops.array(weights * support, "float64")
    return ops.cast(spectrum, "complex64"), quality


def linear_pairs(spectrum, source, receiver, radius, ops, length):
    cross = spectrum[:, source, :].conj() * spectrum[:, receiver, :]
    corr = ops.irfft(cross, length)
    indices = np.arange(-radius, radius + 1) % length
    return corr[..., indices]


class PairAccumulator:
    def __init__(self, pairs, lags, ops, method):
        self.ops, self.method, self.lags = ops, method, lags
        self.sum = ops.array(np.zeros((pairs, lags)), "float64")
        self.phase = ops.array(np.zeros((pairs, lags)), "complex128")
        self.count = ops.array(np.zeros(pairs), "int64")

    def add(self, data, valid):
        ops = self.ops
        if data.ndim == 2:
            data, valid = data[None], valid[None]
        self.sum += (ops.cast(data, "float64") * valid[..., None]).sum(0)
        self.count += valid.sum(0)
        if self.method == "pws":
            n = next_fast_len(self.lags)
            half = ops.rfft(data, n)
            full = ops.array(np.zeros((*data.shape[:-1], n)), "complex64")
            full[..., 0] = half[..., 0]
            upper = (n + 1) // 2
            full[..., 1:upper] = 2 * half[..., 1:upper]
            if n % 2 == 0:
                full[..., n // 2] = half[..., n // 2]
            analytic = ops.ifft(full, n)[..., : self.lags]
            amplitude = abs(analytic)
            phase = analytic / ops.where(amplitude == 0, ops.array(1), amplitude)
            phase = ops.where(amplitude == 0, ops.array(0, "complex64"), phase)
            self.phase += (ops.cast(phase, "complex128") * valid[..., None]).sum(0)

    def finish(self):
        ops = self.ops
        count = ops.where(self.count > 0, self.count, ops.array(1, "int64"))[:, None]
        values = self.sum / count
        if self.method == "pws":
            values = values * (abs(self.phase) / count) ** 2
        peak = ops.maximum(abs(values))
        values = values / ops.where(peak == 0, ops.array(1, "float64"), peak)
        return ops.cast(values - ops.mean(values), "float32")


class ReferenceSession:
    def __init__(self, params, stations):
        self.p, self.stations = params, stations
        self.ops = Ops(params["cc_backend"], params["cc_device"])
        self.source, self.receiver = np.triu_indices(stations, 1)
        self.points = round(
            params["cc_len"] * params["samp_freq"] / params["time_downsample"]
        )
        self.radius = round(
            params["maxlag"] * params["samp_freq"] / params["time_downsample"]
        )
        self.minute_id = None
        self.raw_mode = self.restack = False
        self.clear_group()

    def _new(self, minute=False):
        return PairAccumulator(
            len(self.source),
            2 * self.radius + 1,
            self.ops,
            self.p.get("file_stacking_method", self.p["stacking_method"])
            if minute
            else self.p["stacking_method"],
        )

    def clear_group(self):
        self.minute, self.group = self._new(minute=True), self._new()
        self.short_count = np.zeros(len(self.source), dtype=np.uint64)
        self.minute_id = None
        self.raw_mode = self.restack = False

    def _flush(self):
        if self.minute_id is not None:
            self.group.add(self.minute.finish(), self.minute.count > 0)
            self.short_count += self.ops.host(self.minute.count).astype(np.uint64)
            self.minute = self._new(minute=True)
            self.minute_id = None

    def add_block(self, data, valid, starts, minute_ids):
        if self.restack:
            raise ValueError("Cannot mix restack and raw input")
        self.raw_mode = True
        factor = self.p["time_downsample"]
        taps = fir_taps(factor) if factor > 1 else np.ones(1, dtype=np.float32)
        halo = len(taps) // 2
        for start, minute_id in zip(starts, minute_ids, strict=True):
            if self.minute_id is not None and minute_id != self.minute_id:
                if minute_id < self.minute_id:
                    raise ValueError("Minute IDs must be nondecreasing")
                self._flush()
            self.minute_id = int(minute_id)
            indices = int(start) + np.arange(self.points) * factor
            x = self.ops.array(np.zeros((self.points, self.stations)))
            mask = np.ones((self.points, self.stations), dtype=bool)
            for k, tap in enumerate(taps):
                selected = np.clip(indices + k - halo, 0, len(data) - 1)
                raw = data[selected]
                mask &= valid[selected] & np.isfinite(raw)
                x += self.ops.array(np.where(np.isfinite(raw), raw, 0)) * float(tap)
            x = x.transpose(0, 1) if self.ops.torch else x.T
            traces, quality = preprocess(x[None], self.p, self.ops)
            channel_valid = self.ops.array(mask.all(0), "bool")
            channel_valid = (
                channel_valid & (self.ops.maximum(abs(x))[..., 0] > 0) & quality[0]
            )
            pair_valid = channel_valid[self.source] & channel_valid[self.receiver]
            corr = linear_pairs(
                traces,
                self.source,
                self.receiver,
                self.radius,
                self.ops,
                next_fast_len(2 * self.points - 1),
            )
            self.minute.add(corr, pair_valid[None])

    def stack_pairs(self, data, valid):
        if self.raw_mode:
            raise ValueError("Cannot mix raw input and restack")
        self.restack = True
        self.group.add(self.ops.array(data), self.ops.array(valid, "bool"))

    def finish_group(self):
        self._flush()
        return {
            "pairs": self.ops.host(self.group.finish()),
            "valid_count": self.ops.host(self.group.count).astype(np.uint64),
            "short_count": self.short_count.copy(),
        }

    def diagnostics(self):
        return {"backend": self.p["cc_backend"], "device": self.p["cc_device"]}

    def cancel(self):
        raise RuntimeError("Reference node calculation cancelled")
