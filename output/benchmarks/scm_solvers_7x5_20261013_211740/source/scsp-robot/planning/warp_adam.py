"""Warp Adam + bookkeeping kernels (DexForge-style, no ForceAware import)."""

from types import SimpleNamespace

_KERNELS = None
PI = 3.141592653589793
LOSS_GRADIENT_SCALE = 1.0e-6
DEXFORGE_GRAD_CLIP = 50.0


def ensure_warp():
    import warp as wp

    wp.init()
    return wp


def ensure_kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    wp = ensure_warp()

    @wp.struct
    class AdamState:
        raw: wp.array2d(dtype=float)
        gradient: wp.array2d(dtype=float)
        first_moment: wp.array2d(dtype=float)
        second_moment: wp.array2d(dtype=float)
        gradient_norm: wp.array(dtype=float)
        valid: wp.array(dtype=int)
        iteration: wp.array(dtype=int)
        iterations: int
        learning_rate: float
        final_learning_rate: float
        beta1: float
        beta2: float
        epsilon: float
        clip: float
        gradient_scale: float

    @wp.kernel(enable_backward=False)
    def adam(state: AdamState):
        world, index = wp.tid()
        if state.valid[world] == 0:
            return
        iteration = state.iteration[0]
        if iteration >= state.iterations:
            return
        fraction = float(iteration) / float(wp.max(state.iterations - 1, 1))
        alpha = state.final_learning_rate / wp.max(state.learning_rate, 1.0e-12)
        decay = alpha + (1.0 - alpha) * 0.5 * (1.0 + wp.cos(PI * fraction))
        scaled_norm = state.gradient_norm[world] * state.gradient_scale
        scale = float(1.0)
        if state.clip > 0.0:
            scale = wp.min(1.0, state.clip / wp.max(scaled_norm, 1.0e-12))
        gradient = state.gradient[world, index] * state.gradient_scale * scale
        first = state.beta1 * state.first_moment[world, index] + (1.0 - state.beta1) * gradient
        second = (
            state.beta2 * state.second_moment[world, index]
            + (1.0 - state.beta2) * gradient * gradient
        )
        state.first_moment[world, index] = first
        state.second_moment[world, index] = second
        correction1 = 1.0 - wp.pow(state.beta1, float(iteration + 1))
        correction2 = 1.0 - wp.pow(state.beta2, float(iteration + 1))
        update = (first / correction1) / (wp.sqrt(second / correction2) + state.epsilon)
        state.raw[world, index] = state.raw[world, index] - state.learning_rate * decay * update

    @wp.kernel(enable_backward=False)
    def advance_iteration(iteration: wp.array(dtype=int)):
        iteration[0] = iteration[0] + 1

    @wp.kernel(enable_backward=False)
    def gradient_norm(gradient: wp.array2d(dtype=float), output: wp.array(dtype=float)):
        world = wp.tid()
        max_abs = float(0.0)
        for index in range(gradient.shape[1]):
            max_abs = wp.max(max_abs, wp.abs(gradient[world, index]))
        if max_abs == 0.0:
            output[world] = 0.0
            return
        squared = float(0.0)
        for index in range(gradient.shape[1]):
            normalized = gradient[world, index] / max_abs
            squared += normalized * normalized
        output[world] = max_abs * wp.sqrt(squared)

    @wp.kernel(enable_backward=False)
    def clamp_raw(raw: wp.array2d(dtype=float), limit: float):
        world, index = wp.tid()
        raw[world, index] = wp.clamp(raw[world, index], -limit, limit)

    @wp.kernel(enable_backward=False)
    def copy_if_better(
        loss: wp.array(dtype=float),
        raw: wp.array2d(dtype=float),
        best_loss: wp.array(dtype=float),
        best_raw: wp.array2d(dtype=float),
        best_available: wp.array(dtype=int),
    ):
        world, index = wp.tid()
        value = loss[world]
        if wp.isfinite(value) and value <= best_loss[world]:
            best_raw[world, index] = raw[world, index]
            if index == 0:
                best_loss[world] = value
                best_available[world] = 1

    @wp.kernel(enable_backward=False)
    def restore_best(
        raw: wp.array2d(dtype=float),
        best_raw: wp.array2d(dtype=float),
        best_available: wp.array(dtype=int),
    ):
        world, index = wp.tid()
        if best_available[world] == 1:
            raw[world, index] = best_raw[world, index]

    @wp.kernel(enable_backward=False)
    def add_float2d(source: wp.array2d(dtype=float), target: wp.array2d(dtype=float)):
        world, index = wp.tid()
        target[world, index] = target[world, index] + source[world, index]

    @wp.kernel(enable_backward=False)
    def zero_float2d(target: wp.array2d(dtype=float)):
        world, index = wp.tid()
        target[world, index] = 0.0

    @wp.kernel(enable_backward=False)
    def zero_float1d(target: wp.array(dtype=float)):
        target[wp.tid()] = 0.0

    _KERNELS = SimpleNamespace(
        wp=wp,
        AdamState=AdamState,
        adam=adam,
        advance_iteration=advance_iteration,
        gradient_norm=gradient_norm,
        clamp_raw=clamp_raw,
        copy_if_better=copy_if_better,
        restore_best=restore_best,
        add_float2d=add_float2d,
        zero_float2d=zero_float2d,
        zero_float1d=zero_float1d,
    )
    return _KERNELS


def launch_adam(
    raw,
    gradient,
    first_moment,
    second_moment,
    gradient_norm_arr,
    valid,
    iteration,
    *,
    iterations,
    learning_rate,
    final_learning_rate=None,
    beta1=0.9,
    beta2=0.999,
    epsilon=1.0e-8,
    clip=0.0,
    gradient_scale=1.0,
):
    k = ensure_kernels()
    if final_learning_rate is None:
        final_learning_rate = learning_rate
    state = k.AdamState()
    state.raw = raw
    state.gradient = gradient
    state.first_moment = first_moment
    state.second_moment = second_moment
    state.gradient_norm = gradient_norm_arr
    state.valid = valid
    state.iteration = iteration
    state.iterations = int(iterations)
    state.learning_rate = float(learning_rate)
    state.final_learning_rate = float(final_learning_rate)
    state.beta1 = float(beta1)
    state.beta2 = float(beta2)
    state.epsilon = float(epsilon)
    state.clip = float(clip)
    state.gradient_scale = float(gradient_scale)
    k.wp.launch(k.gradient_norm, dim=raw.shape[0], inputs=[gradient, gradient_norm_arr])
    k.wp.launch(k.adam, dim=raw.shape, inputs=[state])
    k.wp.launch(k.clamp_raw, dim=raw.shape, inputs=[raw, 8.0])
    k.wp.launch(k.advance_iteration, dim=1, inputs=[iteration])
