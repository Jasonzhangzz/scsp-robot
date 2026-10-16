"""Fuse independent Warp tape gradient clears without changing derivatives."""
import warp as wp
from functools import cache
from types import FunctionType,MethodType


@cache
def copy_kernel(dtype):
    @wp.kernel(enable_backward=False,module='unique')
    def copy(source:wp.array(dtype=dtype),target:wp.array(dtype=dtype)):
        i=wp.tid();target[i]=source[i]
    return copy


def device_copy(target,source,*args,**kwargs):
    if (not args and not kwargs and target.device.is_cuda and target.device==source.device
            and target.dtype==source.dtype and target.size==source.size and target.is_contiguous and source.is_contiguous):
        if target.size and target.ptr!=source.ptr:
            wp.launch(copy_kernel(source.dtype),target.size,inputs=[source.flatten(),target.flatten()],device=source.device)
        return
    return wp.copy(target,source,*args,**kwargs)


class KernelCopyWarp:
    """Private module view: existing native helpers keep all other operations."""
    copy=staticmethod(device_copy)
    def __getattr__(self,name):return getattr(wp,name)


def private_function(function,**overrides):
    # Isolate the few Python dispatch functions that use wp.copy. Their native
    # kernels and mathematics are unchanged; no shared module is monkeypatched.
    namespace=dict(function.__globals__);namespace.update(overrides)
    result=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result


def fuse_tape_seeds(tape):
    original=tape.backward
    def backward(loss=None,grads=None):
        if grads:
            for primal,seed in grads.items():
                if primal.grad is None:primal.grad=seed
                else:device_copy(primal.grad,seed)
        return original(loss=loss)
    tape.backward=backward


def fuse_tape_zero(tape):
    import torch
    original=tape.zero
    cache={}
    def zero():
        arrays=[]
        for primal,gradient in tape.gradients.items():
            if wp._src.types.is_struct(primal):
                for name,field in gradient._cls.vars.items():
                    if wp._src.types.matches_array_class(field.type,wp.array) and field.requires_grad:
                        arrays.append(getattr(gradient,name))
            else:arrays.append(gradient)
        # The same storage appears in multiple struct/array entries.
        unique={(a.ptr,a.shape,str(a.dtype)):a for a in arrays if a.ptr and a.size}
        signature=tuple(unique)
        if not signature:return
        if signature not in cache:
            cache[signature]=[wp.to_torch(a) for a in unique.values()]
        torch._foreach_zero_(cache[signature])
    tape.zero=zero
    return original


def install(backend):
    """Only replace tapes owned by this optimizer, never global Warp methods."""
    import torch
    stream=torch.cuda.ExternalStream(wp.get_stream(backend.device).cuda_stream,device=str(backend.device))
    for runtime in backend.steps:
        recorded=runtime.recorded
        for name in ('pose_tape','contact_tape','coefficient_tape','integration_tape'):
            fuse_tape_zero(getattr(recorded,name))
            fuse_tape_seeds(getattr(recorded,name))
    for workspace in (*backend.loss.immediate,backend.loss.terminal):fuse_tape_seeds(workspace.tape)
    # Observation tapes also supply small GPU seed arrays.
    observation=backend.terminal_observation.recorded
    for name in ('pose_tape','collision_tape'):
        if hasattr(observation,name):fuse_tape_seeds(getattr(observation,name))
    from forceaware import optimizer as opt
    from comfree_warp.native_adjoint import fast_step_adjoint as native
    proxy=KernelCopyWarp()
    prepare=private_function(native._backward_prepare,wp=proxy)
    step=private_function(native.backward,_backward_prepare=prepare)
    context=private_function(opt._copy_context_cotangent,wp=proxy)
    loss=private_function(opt._copy_loss_cotangent,wp=proxy,_copy_context_cotangent=context)
    observation_copy=private_function(opt._copy_observation_cotangent,wp=proxy)
    function=private_function(backend._backward.__func__,step_backward=step,
        _copy_loss_cotangent=loss,_copy_observation_cotangent=observation_copy)
    original=MethodType(function,backend)
    def backward(*args,**kwargs):
        with torch.cuda.stream(stream),torch.no_grad():return original(*args,**kwargs)
    backend._backward=backward
