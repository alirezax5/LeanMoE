from __future__ import annotations
import ctypes, math
from dataclasses import dataclass
from .native import LeanMoENativeError, NativeBridge

class _LMSamplerConfigV2(ctypes.Structure):
    _fields_=[("temperature",ctypes.c_float),("top_k",ctypes.c_int32),("top_p",ctypes.c_float),
    ("min_p",ctypes.c_float),("seed",ctypes.c_uint32),("greedy",ctypes.c_uint8),
    ("reserved0",ctypes.c_uint8*3),("penalty_last_n",ctypes.c_int32),("penalty_repeat",ctypes.c_float),
    ("penalty_freq",ctypes.c_float),("penalty_present",ctypes.c_float)]

@dataclass(frozen=True)
class SamplingConfig:
    temperature: float=.8
    top_k: int=40
    top_p: float=.95
    min_p: float=.05
    seed: int=12345
    greedy: bool=False
    penalty_last_n: int=64
    repeat_penalty: float=1.0
    frequency_penalty: float=0.0
    presence_penalty: float=0.0
    def __post_init__(self):
        if not isinstance(self.seed,int) or isinstance(self.seed,bool) or not 0<=self.seed<=0xFFFFFFFF: raise ValueError("seed must be uint32")
        if not isinstance(self.penalty_last_n,int) or isinstance(self.penalty_last_n,bool) or self.penalty_last_n<0: raise ValueError("penalty_last_n must be >= 0")
        vals=(self.temperature,self.top_p,self.min_p,self.repeat_penalty,self.frequency_penalty,self.presence_penalty)
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(float(x)) for x in vals): raise ValueError("sampling floats must be finite")
        if not self.greedy:
            if self.temperature<=0: raise ValueError("temperature must be > 0")
            if not 0<self.top_p<=1: raise ValueError("top_p must be in (0,1]")
            if not 0<=self.min_p<=1: raise ValueError("min_p must be in [0,1]")
            if self.repeat_penalty<=0: raise ValueError("repeat_penalty must be > 0")

class NativeSampler:
    def __init__(self,bridge:NativeBridge,model,config:SamplingConfig):
        self._bridge=bridge; self._dll=bridge.dll; self._handle=ctypes.c_void_p()
        self._dll.lm_sampler_create_v2.argtypes=[ctypes.c_void_p,ctypes.POINTER(_LMSamplerConfigV2),ctypes.POINTER(ctypes.c_void_p)]
        self._dll.lm_sampler_create_v2.restype=ctypes.c_int
        self._dll.lm_sampler_free.argtypes=[ctypes.c_void_p]; self._dll.lm_sampler_free.restype=None
        self._dll.lm_sampler_sample.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int32)]; self._dll.lm_sampler_sample.restype=ctypes.c_int
        self._dll.lm_sampler_reset.argtypes=[ctypes.c_void_p]; self._dll.lm_sampler_reset.restype=ctypes.c_int
        c=_LMSamplerConfigV2(config.temperature,config.top_k,config.top_p,config.min_p,config.seed,int(config.greedy),(ctypes.c_uint8*3)(0,0,0),config.penalty_last_n,config.repeat_penalty,config.frequency_penalty,config.presence_penalty)
        bridge.check(self._dll.lm_sampler_create_v2(model,ctypes.byref(c),ctypes.byref(self._handle)),"lm_sampler_create_v2()")
        if not self._handle.value: raise LeanMoENativeError("null sampler")
    def sample(self,context):
        t=ctypes.c_int32(); self._bridge.check(self._dll.lm_sampler_sample(self._handle,context,ctypes.byref(t)),"lm_sampler_sample()"); return int(t.value)
    def reset(self): self._bridge.check(self._dll.lm_sampler_reset(self._handle),"lm_sampler_reset()")
    def close(self):
        if self._handle.value: self._dll.lm_sampler_free(self._handle); self._handle=ctypes.c_void_p()
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
