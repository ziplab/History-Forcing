# Copyright 2024-2025 The Alibaba Wan Team Authors.
# Copyright 2025 Stability AI, Katherine Crowson and The HuggingFace Team.
# Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Euler and UniPC solvers, checkpoint noise grids and flow-training preparation.

Euler follows Diffusers v0.37.1; UniPC follows minWM / Wan2.1.
No external scheduler classes, configuration mixins or Hub loading are used.
Noise is caller-owned for reproducible paired rollouts. Solver arithmetic is
FP32; updates return to the latent dtype. UniPC keeps BH2 and final-order reduction.
AnyFlow, SGF and LingBot configure the same Euler solver rather than subclassing it.
"""

import math
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import numpy as np
import torch

from utils import block_lengths

NUM_TIMESTEPS = 1000
TIMESTEP_SHIFT = 5.0
DEFAULT_FEW_STEP_TIMESTEPS = (1000, 750, 500, 250)

_sampling_observer = ContextVar("sampling_observer", default=None)


def sampling_schedule_manifest(scheduler):
    """Snapshot the bound native grid and arithmetic, without constructing a solver."""
    stochastic = scheduler.stochastic_sampling
    unipc = isinstance(scheduler, FlowUniPCMultistepScheduler)
    result = {
        "solver": "unipc" if unipc else "euler",
        "update": "clean_re_noise" if stochastic else "unipc_bh2" if unipc else "euler",
        "sampling_mode": "sde" if stochastic else "ode",
        "num_train_timesteps": scheduler.num_train_timesteps,
    }
    for name in ("timesteps", "unshifted_timesteps", "sigmas", "noise_sigmas", "model_targets", "ode_deltas"):
        value = getattr(scheduler, name, None)
        result[name] = None if value is None else value.detach().cpu().tolist()
        if value is not None:
            result[name + "_dtype"] = str(value.dtype)
    result['arithmetic_dtype'] = 'torch.float32'
    for name in ("solver_order",):
        value = getattr(scheduler, name, None)
        result[name] = str(value) if isinstance(value, torch.dtype) else value
    return result


@contextmanager
def observe_sampling():
    """Collect only schedulers actually used by sampling_step; never draw noise."""
    records = []
    observer = (records, WeakKeyDictionary(), set())
    token = _sampling_observer.set(observer)
    try:
        yield records
    finally:
        _sampling_observer.reset(token)


def shift_sigmas(sigmas, shift):
    """Shared flow time warp, preserving the caller's tensor/array precision."""
    shift = float(shift)
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("Timestep shift must be finite and positive")
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def flowmap_schedule(num_inference_steps, device, *, shift=5.0, num_train_timesteps=1000):
    """SolarWM Stage 1 adjacent (t,r) schedule, in official FP32 operation order.

    From SolarWM runtime/inference.py::_flowmap_schedule, Apache-2.0.
    A flow map conditions on both endpoints; this is not a velocity-only solver.
    """
    if type(num_inference_steps) is not int or num_inference_steps < 1 or shift <= 0 or num_train_timesteps < 1:
        raise ValueError("Flow-map inference schedule values must be positive")
    base = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
    shifted = shift_sigmas(base, shift)
    shifted[-1] = 0.0
    raw = shifted * float(num_train_timesteps)
    return raw[:-1].contiguous(), raw[1:].contiguous()


def flow_noise(clean, noise, sigma):
    """Flow corruption in FP32; the caller owns noise and the output dtype."""
    clean, noise, sigma = clean.float(), noise.float(), sigma.to(device=clean.device, dtype=torch.float32)
    return (1 - sigma) * clean + sigma * noise


def _timestep_index(timesteps, timestep):
    indices = (timesteps == timestep).nonzero()
    if not len(indices):
        raise ValueError(f"Timestep {timestep} is not in the sampling schedule")
    # Wan and Diffusers select the second match when timesteps alias.
    return indices[1 if len(indices) > 1 else 0].item()


def _schedule_noise(scheduler, sample, noise, timesteps):
    timesteps = torch.as_tensor(timesteps, device=scheduler.timesteps.device).reshape(-1)
    if scheduler.begin_index is None:
        indices = [scheduler.index_for_timestep(t) for t in timesteps]
    else:
        index = scheduler.begin_index if scheduler.step_index is None else scheduler.step_index
        indices = [index] * len(timesteps)
    sigma = scheduler.sigmas.to(device=sample.device, dtype=torch.float32)[indices]
    sigma = sigma.reshape(-1, *([1] * (sample.ndim - 1)))
    return flow_noise(sample, noise, sigma).to(sample.dtype)


def sample_block_noises(shape, steps, generator, device, dtype, *, stochastic_sampling=True):
    """Draw one block's initial noise followed by its paired SDE transitions.

    A tuple of generators keeps each sample's stream independent of batching.
    ODE draws only the initial tensor, preserving the native RNG consumption.
    """
    steps = steps if stochastic_sampling else 1
    if isinstance(generator, tuple):
        if len(generator) != shape[0]:
            raise ValueError('Noise generators must match the sample batch')
        noises = tuple(torch.stack([torch.randn(shape[1:], generator=item, device=device, dtype=dtype)
                                    for item in generator]) for _ in range(steps))
    else:
        noises = tuple(torch.randn(shape, generator=generator, device=device, dtype=dtype)
                       for _ in range(steps))
    return noises[0], noises[1:]


class FlowMatchEulerDiscreteScheduler:
    """Flow-matching time table with first-order or re-noising updates."""

    def __init__(
        self, num_train_timesteps=1000, *, shift=1.0, stochastic_sampling=False,
    ):
        if num_train_timesteps <= 0 or not math.isfinite(shift) or shift <= 0:
            raise ValueError("Invalid timestep count or shift")
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.stochastic_sampling = stochastic_sampling
        self.model_targets = self.noise_sigmas = self.ode_deltas = None
        sigmas = torch.from_numpy(
            np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        ) / num_train_timesteps
        self.unshifted_timesteps = sigmas * num_train_timesteps
        self.sigmas = shift_sigmas(sigmas, shift)
        self.timesteps = self.sigmas * num_train_timesteps
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        self.num_inference_steps = None
        self.reset()

    def set_schedule(self, timesteps, sigmas, *, noise_sigmas=None,
                     model_targets=None, ode_deltas=None, unshifted_timesteps=None):
        """Bind a checkpoint's precomputed grid without altering its rounding."""
        if timesteps.ndim != 1 or not len(timesteps) or sigmas.shape != (len(timesteps) + 1,):
            raise ValueError('Sampling grid needs N timesteps and N+1 sigmas')
        unshifted_timesteps = timesteps if unshifted_timesteps is None else unshifted_timesteps
        if unshifted_timesteps.shape != timesteps.shape:
            raise ValueError('Unshifted and model timestep grids must be aligned')
        self.timesteps, self.sigmas, self.unshifted_timesteps = timesteps, sigmas, unshifted_timesteps
        self.num_inference_steps = len(timesteps)
        self.targets = torch.cat((timesteps[1:], timesteps.new_zeros(1)))
        self.noise_sigmas, self.model_targets, self.ode_deltas = noise_sigmas, model_targets, ode_deltas
        self.reset()

    def reset(self):
        self.step_index = None
        self.begin_index = None

    def set_begin_index(self, begin_index=0):
        self.begin_index = int(begin_index)

    def set_timesteps(self, num_inference_steps=None, device=None, *, sigmas=None):
        if sigmas is None:
            if (
                type(num_inference_steps) is not int
                or not 1 <= num_inference_steps <= self.num_train_timesteps
            ):
                raise ValueError("Invalid inference step count")
            sigmas = np.linspace(
                self.sigma_max * self.num_train_timesteps,
                self.sigma_min * self.num_train_timesteps,
                num_inference_steps,
            ) / self.num_train_timesteps
        else:
            sigmas = np.asarray(sigmas, dtype=np.float32)
        if sigmas.ndim != 1 or not len(sigmas) or not np.isfinite(sigmas).all():
            raise ValueError("sigmas must be a finite nonempty vector")
        if (sigmas <= 0).any() or (sigmas > 1).any() or (np.diff(sigmas) >= 0).any():
            raise ValueError("sigmas must decrease strictly within (0, 1]")
        unshifted_timesteps = torch.from_numpy(sigmas.copy()).to(device=device, dtype=torch.float32) * self.num_train_timesteps
        sigmas = shift_sigmas(sigmas, self.shift)
        sigmas = torch.from_numpy(sigmas).to(device=device, dtype=torch.float32)
        self.set_schedule(sigmas * self.num_train_timesteps, torch.cat((sigmas, sigmas.new_zeros(1))),
                          unshifted_timesteps=unshifted_timesteps)

    def index_for_timestep(self, timestep):
        return _timestep_index(self.timesteps, timestep)

    def scale_noise(self, sample, timestep, noise):
        """Corrupt in FP32 and return to the latent dtype."""
        return _schedule_noise(self, sample, noise, timestep)

    def convert_model_output(self, model_output, *, sample, index=None):
        """Predict x0 in FP32 without intermediate BF16 rounding."""
        index = self.step_index if index is None else index
        sigma = self.sigmas[index].to(device=sample.device, dtype=torch.float32)
        return sample.float() - sigma * model_output.float()

    def step(self, model_output, timestep, sample, *, noise=None, index=None):
        if self.num_inference_steps is None:
            raise ValueError("Call set_timesteps before step")
        if index is not None:
            self.step_index = index
        elif self.step_index is None:
            self.step_index = (self.index_for_timestep(timestep) if self.begin_index is None
                               else self.begin_index)
        sigma, sigma_next = self.sigmas[self.step_index : self.step_index + 2]
        if self.stochastic_sampling:
            clean = self.convert_model_output(model_output, sample=sample)
            if self.step_index + 1 == len(self.timesteps):
                result = clean
            else:
                if noise is None:
                    raise ValueError(
                        "Stochastic sampling requires caller-supplied transition noise"
                    )
                if self.noise_sigmas is not None:
                    sigma_next = self.noise_sigmas[self.step_index + 1]
                result = flow_noise(clean, noise, sigma_next)
        else:
            delta = sigma_next - sigma if self.ode_deltas is None else self.ode_deltas[self.step_index]
            result = sample.float() + delta.to(device=sample.device, dtype=torch.float32) * model_output.float()
        self.step_index += 1
        return result.to(sample.dtype)


class FlowUniPCMultistepScheduler:
    """Stateful flow-UniPC solver; create a separate instance per AR trajectory."""

    def __init__(self, num_train_timesteps=1000, *, shift=1.0, solver_order=2):
        if num_train_timesteps <= 0 or solver_order not in (1, 2, 3):
            raise ValueError("Invalid timestep count or UniPC order")
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("Timestep shift must be finite and positive")
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.solver_order = solver_order
        self.stochastic_sampling = False
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[
            ::-1
        ].copy()
        sigmas = torch.from_numpy(1.0 - alphas).float()
        self.unshifted_timesteps = sigmas * num_train_timesteps
        self.sigmas = shift_sigmas(sigmas, shift)
        self.timesteps = self.sigmas * num_train_timesteps
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        self.num_inference_steps = None
        self.reset()

    def reset(self):
        self.step_index = None
        self.model_outputs = [None] * self.solver_order
        self.begin_index = None
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = 0

    def set_begin_index(self, begin_index=0):
        self.begin_index = int(begin_index)

    def set_timesteps(self, num_inference_steps, device=None, *, shift=None):
        if (
            type(num_inference_steps) is not int
            or not 1 <= num_inference_steps <= self.num_train_timesteps
        ):
            raise ValueError("Invalid inference step count")
        shift = self.shift if shift is None else float(shift)
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("Timestep shift must be finite and positive")
        sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1)[
            :-1
        ]
        self.unshifted_timesteps = torch.from_numpy(sigmas.copy() * self.num_train_timesteps).to(device=device, dtype=torch.float32)
        sigmas = shift_sigmas(sigmas, shift)
        self.timesteps = torch.from_numpy(sigmas * self.num_train_timesteps).to(
            device=device, dtype=torch.int64
        )
        # Keep the CPU FP32 coefficients and NumPy FP64 construction used by Wan.
        self.sigmas = torch.from_numpy(
            np.concatenate([sigmas, [0.0]]).astype(np.float32)
        )
        self.num_inference_steps = num_inference_steps
        self.reset()

    @staticmethod
    def _sigma_to_alpha_sigma_t(sigma):
        return 1 - sigma, sigma

    def index_for_timestep(self, timestep):
        return _timestep_index(self.timesteps, timestep)

    def add_noise(self, original_samples, noise, timesteps):
        return _schedule_noise(self, original_samples, noise, timesteps)

    def convert_model_output(self, model_output, *, sample, index=None):
        sigma = self.sigmas[self.step_index if index is None else index].to(device=sample.device, dtype=torch.float32)
        return sample.float() - sigma * model_output.float()

    def step(self, model_output, timestep, sample, *, noise=None, index=None):
        if self.num_inference_steps is None:
            raise ValueError("Call set_timesteps before step")
        if noise is not None:
            raise ValueError('UniPC uses deterministic updates without transition noise')
        output_dtype = sample.dtype
        sample, model_output = sample.float(), model_output.float()
        if self.step_index is None:
            # Integer model timesteps may repeat; indices identify the actual sigma.
            self.step_index = index if index is not None else self.begin_index
            if self.step_index is None:
                self.step_index = self.index_for_timestep(timestep)
        elif index is not None and index != self.step_index:
            raise ValueError('UniPC step indices must be sequential; reset the solver before a new trajectory')
        converted = self.convert_model_output(model_output, sample=sample)
        if self.step_index > 0 and self.last_sample is not None:
            sample = self.multistep_uni_c_bh_update(
                converted, self.last_sample, sample, self.this_order
            )
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = converted
        self.this_order = min(
            self.solver_order,
            len(self.timesteps) - self.step_index,
            self.lower_order_nums + 1,
        )
        self.last_sample = sample
        result = self.multistep_uni_p_bh_update(sample, self.this_order)
        self.lower_order_nums = min(self.lower_order_nums + 1, self.solver_order)
        self.step_index += 1
        return result.to(output_dtype)

    def multistep_uni_p_bh_update(self, sample, order):
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = sample
        sigma_t, sigma_s0 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
        )
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        device = sample.device
        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        rks.append(1.0)
        rks = torch.tensor(rks, device=device)
        R = []
        b = []
        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        B_h = torch.expm1(hh)
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        R = torch.stack(R)
        b = torch.tensor(b, device=device)
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)
        else:
            D1s = None
        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - alpha_t * B_h * pred_res
        x_t = x_t.to(x.dtype)
        return x_t

    def multistep_uni_c_bh_update(
        self, this_model_output, last_sample, this_sample, order
    ):
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output
        sigma_t, sigma_s0 = (
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        device = this_sample.device
        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        rks.append(1.0)
        rks = torch.tensor(rks, device=device)
        R = []
        b = []
        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        B_h = torch.expm1(hh)
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        R = torch.stack(R)
        b = torch.tensor(b, device=device)
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)
        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
        else:
            corr_res = 0
        D1_t = model_t - m0
        x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        x_t = x_t.to(x.dtype)
        return x_t


def _bell_weights(timestep):
    """Return minWM's normalized bell weighting for warped timesteps."""

    weight = torch.exp(-2.0 * ((timestep - NUM_TIMESTEPS / 2) / NUM_TIMESTEPS).square())
    weight = weight - weight.min()
    return weight * (NUM_TIMESTEPS / weight.sum())


def _integer_sequence(values, name):
    try:
        values = tuple(values)
        if any(isinstance(item, bool) or not isinstance(item, (int, str, np.integer)) for item in values):
            raise ValueError('non-integer timestep')
        return tuple(int(item) for item in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} sequence must contain integers') from error


def parse_timesteps(value):
    """Parse the canonical denoising-timestep sequence."""
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise TypeError(
            "sampling_steps must be an integer, sequence, or comma-separated list"
        )
    steps = _integer_sequence(values, 'sampling_steps')
    if not steps:
        raise ValueError("sampling_steps sequence cannot be empty")
    if any(step <= 0 or step > NUM_TIMESTEPS for step in steps):
        raise ValueError(f"sampling_steps timesteps must be in [1,{NUM_TIMESTEPS}]")
    if any(left <= right for left, right in zip(steps, steps[1:])):
        raise ValueError("sampling_steps timesteps must be strictly decreasing")
    return steps


def parse_sampling_steps(value):
    """Parse one public sampler field.

    An integer selects a step count for the backend's sampler. A sequence
    (or comma-separated string) selects explicit decreasing timesteps.
    """
    if type(value) is int:
        steps = value
    elif isinstance(value, str) and "," not in value:
        try:
            steps = int(value)
        except ValueError as error:
            raise ValueError(
                "sampling_steps must be an integer or timestep sequence"
            ) from error
    else:
        return parse_timesteps(value)
    if not 1 <= steps <= NUM_TIMESTEPS:
        raise ValueError(f"sampling_steps must be in [1,{NUM_TIMESTEPS}]")
    return steps


def make_flow_match_scheduler(
    device,
    timesteps=None,
    *,
    timestep_shift=TIMESTEP_SHIFT,
    sampling_mode="sde",
):
    """Native few-step ODE/SDE on an explicit or uniformly spaced shifted grid.

    This explicit grid differs from Diffusers' default endpoint-inclusive grid.
    All factories share FP32 clean conversion and re-noising.
    """

    if sampling_mode not in ("ode", "sde"):
        raise ValueError("sampling_mode must be ode or sde")

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=NUM_TIMESTEPS,
        shift=float(timestep_shift),
        stochastic_sampling=sampling_mode == "sde",
    )
    if type(timesteps) is int:
        count = parse_sampling_steps(timesteps)
        base = torch.linspace(1.0, 0.0, count + 1, device=device, dtype=torch.float32)
        sigmas = shift_sigmas(base, timestep_shift)
        scheduler.set_schedule(sigmas[:-1] * NUM_TIMESTEPS, sigmas,
                               unshifted_timesteps=base[:-1] * NUM_TIMESTEPS)
        return scheduler
    sigmas = torch.linspace(1.0, 0.0, NUM_TIMESTEPS + 1, dtype=torch.float32)[:-1]
    if timesteps is not None:
        indices = NUM_TIMESTEPS - torch.tensor(parse_timesteps(timesteps))
        sigmas = sigmas[indices]
    scheduler.set_timesteps(device=device, sigmas=sigmas.numpy())
    return scheduler


def make_unipc_scheduler(
    device,
    sampling_steps=50,
    *,
    timestep_shift=TIMESTEP_SHIFT,
    sampling_mode="ode",
    num_train_timesteps=NUM_TIMESTEPS,
):
    """Independent flow-UniPC solver with the official Wan timestep table."""

    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=num_train_timesteps)
    scheduler.set_timesteps(sampling_steps, device=device, shift=timestep_shift)
    if sampling_mode == "sde":
        # Retain Wan's native table, including its integer model timesteps.
        stochastic = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps, stochastic_sampling=True,
        )
        stochastic.set_schedule(scheduler.timesteps, scheduler.sigmas.to(device),
                                unshifted_timesteps=scheduler.unshifted_timesteps)
        return stochastic
    if sampling_mode != "ode":
        raise ValueError("sampling_mode must be ode or sde")
    return scheduler


def make_sampling_scheduler(
    device,
    sampling_steps,
    *,
    solver=None,
    timestep_shift=TIMESTEP_SHIFT,
    sampling_mode="ode",
    num_train_timesteps=NUM_TIMESTEPS,
):
    """Construct a solver from checkpoint metadata and the requested noise grid.

    Without an explicit solver, step counts use Wan UniPC and timestep lists
    use the native few-step flow grid. ``euler`` retains Diffusers' default
    endpoint-inclusive grid, which differs from that explicit few-step grid.
    """
    steps = parse_sampling_steps(sampling_steps)
    if solver is None:
        solver = "unipc" if type(steps) is int else "flow_match"
    if solver == "flow_match":
        return make_flow_match_scheduler(
            device, steps, timestep_shift=timestep_shift, sampling_mode=sampling_mode,
        )
    if solver not in ("unipc", "euler"):
        raise ValueError(f"Unknown sampling solver: {solver}")
    if type(steps) is not int:
        raise ValueError(f"{solver} requires an integer sampling step count")
    if solver == "unipc":
        return make_unipc_scheduler(
            device, steps, timestep_shift=timestep_shift, sampling_mode=sampling_mode,
            num_train_timesteps=num_train_timesteps,
        )
    if sampling_mode != "ode":
        raise ValueError("Full-context Flow Euler sampling requires ODE")
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=num_train_timesteps, shift=timestep_shift,
    )
    scheduler.set_timesteps(steps, device=device)
    return scheduler


@dataclass(frozen=True)
class FlowTrainingSchedule:
    """Aligned noise levels and loss weights, independent of sampler state."""

    timesteps: torch.Tensor
    sigmas: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self):
        if (self.timesteps.ndim != 1 or not len(self.timesteps)
                or self.sigmas.shape != self.timesteps.shape or self.weights.shape != self.timesteps.shape):
            raise ValueError('Training timesteps, sigmas and weights must be aligned nonempty vectors')

    def select(self, indices):
        return type(self)(self.timesteps[indices], self.sigmas[indices], self.weights[indices])

    def checkpoint_identity(self):
        """The bound training grid, including corruption/loss arithmetic precision."""
        identity = {}
        for name in ("timesteps", "sigmas", "weights"):
            value = getattr(self, name)
            identity[name] = {"values": value.detach().cpu().tolist(), "dtype": str(value.dtype)}
        return identity


def make_flow_training_schedule(device, timestep_shift=TIMESTEP_SHIFT):
    """Return the local noise table and precomputed SFT loss weights."""

    scheduler = make_flow_match_scheduler(device, None, timestep_shift=timestep_shift)
    weights = _bell_weights(scheduler.timesteps.cpu()).to(device)
    return FlowTrainingSchedule(scheduler.timesteps, scheduler.sigmas[:len(weights)].to(device), weights)


def sample_flow_training(clean_latent, frame_blocks, schedule):
    """Sample one noise level per sample/block.

    ``frame_blocks`` is a uniform block size or an explicit partition of the
    latent timeline. The backend supplies its layout, including any first frame.
    """

    if clean_latent.ndim != 5:
        raise ValueError("Clean latent must have shape [B,F,C,H,W]")
    batch_size, frames = clean_latent.shape[:2]
    lengths = (block_lengths(frames, frame_blocks) if type(frame_blocks) is int
               else tuple(frame_blocks))
    if not lengths or any(type(length) is not int or length <= 0 for length in lengths) or sum(lengths) != frames:
        raise ValueError("Training blocks must be positive lengths covering the latent timeline")
    repeats = (lengths[0] if all(length == lengths[0] for length in lengths)
               else torch.tensor(lengths, device=clean_latent.device))
    noise = torch.randn_like(clean_latent)
    indices = torch.randint(
        0,
        len(schedule.timesteps),
        (batch_size, len(lengths)),
        device=clean_latent.device,
    ).repeat_interleave(repeats, dim=1, output_size=frames)
    timestep = schedule.timesteps[indices]
    # Keep the sampled indices: timestep lookup would introduce CPU/GPU sync
    # and reduced-precision timestep aliases. Batch corruption stays vectorized.
    sigma = schedule.sigmas[indices][..., None, None, None]
    noisy = flow_noise(clean_latent, noise, sigma).to(clean_latent.dtype)
    return (
        noisy,
        noise.float() - clean_latent.float(),
        timestep,
        schedule.weights[indices][..., None, None, None],
    )


def timestep_indices(value):
    """Accept a YAML sequence or the equivalent comma-separated CLI value."""
    if isinstance(value, str):
        value = value.split(",")
    indices = _integer_sequence(value, 'LingBot timestep_indices')
    if (
        not indices
        or indices != tuple(sorted(set(indices)))
        or min(indices) < 0
        or max(indices) >= 1000
    ):
        raise ValueError(
            "LingBot timestep_indices must be increasing unique indices in [0,999]"
        )
    return indices


def _euler_schedule(device, timesteps, sigmas, *, sampling_mode='ode', unshifted_timesteps=None, **options):
    if sampling_mode not in ('ode', 'sde'):
        raise ValueError('sampling_mode must be ode or sde')
    scheduler = FlowMatchEulerDiscreteScheduler(stochastic_sampling=sampling_mode == 'sde', **options)
    scheduler.set_schedule(timesteps.to(device), sigmas.to(device), unshifted_timesteps=unshifted_timesteps)
    return scheduler


def make_flow_map_scheduler(device, steps, *, shift=5.0):
    """AnyFlow supplies (t,r) to the model and uses the shared Euler update."""
    times, targets = flowmap_schedule(steps, device, shift=shift)
    sigmas = torch.cat((times / NUM_TIMESTEPS, times.new_zeros(1)))
    scheduler = _euler_schedule(device, times, sigmas,
                               unshifted_timesteps=torch.linspace(float(NUM_TIMESTEPS), 0., steps + 1,
                                                                  device=device)[:-1])
    scheduler.model_targets = targets
    # Preserve subtraction before division in the native endpoint update.
    scheduler.ode_deltas = -(times - targets) / NUM_TIMESTEPS
    return scheduler


def make_lingbot_scheduler(device, indices, shift, *, sampling_mode='sde'):
    """Native UniPC index grid; first-match clean sigma and second-match noise sigma."""
    indices = list(timestep_indices(indices))
    native = make_unipc_scheduler(device, NUM_TIMESTEPS, timestep_shift=shift)
    times = native.timesteps[indices]
    nearest = (native.timesteps[:, None] - times).abs().argmin(dim=0)
    sigmas = native.sigmas[nearest.cpu()].to(device=device)
    noise_sigmas = native.sigmas[[native.index_for_timestep(t) for t in times]].to(device)
    scheduler = _euler_schedule(
        device, times, torch.cat((sigmas, sigmas.new_zeros(1))), sampling_mode=sampling_mode,
        unshifted_timesteps=native.unshifted_timesteps[indices],
    )
    scheduler.noise_sigmas = torch.cat((noise_sigmas, noise_sigmas.new_zeros(1)))
    return scheduler


def make_solar_sgf_scheduler(device, *, shift=5.0, sampling_mode='sde'):
    """Released NFE4 grid with the shared FP32 update."""
    table = shift_sigmas(torch.linspace(1., 0., NUM_TIMESTEPS + 1, dtype=torch.float32)[:-1], shift).to(device)
    indices = torch.tensor([NUM_TIMESTEPS - t for t in DEFAULT_FEW_STEP_TIMESTEPS], device=device)
    times = table[indices] * NUM_TIMESTEPS
    sigmas = torch.cat((times / NUM_TIMESTEPS, times.new_zeros(1)))
    return _euler_schedule(
        device, times, sigmas, sampling_mode=sampling_mode,
        unshifted_timesteps=torch.tensor(DEFAULT_FEW_STEP_TIMESTEPS, device=device, dtype=torch.float32),
    )


def sampling_step(scheduler, prediction, sample, index, *, noise=None):
    """One update contract for Euler and UniPC, including explicit paired-rollout indices."""
    output = scheduler.step(prediction, scheduler.timesteps[index], sample, index=index, noise=noise)
    observer = _sampling_observer.get()
    if observer is not None:
        records, seen, fingerprints = observer
        precision = (sample.dtype, prediction.dtype, output.dtype)
        previous = seen.setdefault(scheduler, set())
        if precision not in previous:
            previous.add(precision)
            record = sampling_schedule_manifest(scheduler)
            record.update(sample_dtype=str(sample.dtype), prediction_dtype=str(prediction.dtype),
                          output_dtype=str(output.dtype),
                          provided_transition_noise_dtype=None if noise is None else str(noise.dtype))
            fingerprint = json.dumps(record, sort_keys=True)
            if fingerprint not in fingerprints:
                fingerprints.add(fingerprint)
                records.append(record)
    return output
