"""
PyTorch implementations of adaptive activation functions from Section 4 of:

V. Kunc & J. Kléma, "Three Decades of Activations:
A Comprehensive Survey of 400 Activation Functions for Neural Networks" (2024).

Scope
-----
This module targets every numbered activation/function definition in Section 4
(§4.1 through §4.57), including formulas embedded in subsections when they are
needed to reproduce the activation exactly.

Conventions
-----------
* Trainable parameters are nn.Parameter objects.
* Parameters described as fixed hyperparameters stay ordinary Python floats
  unless `trainable=True` is explicitly exposed.
* Generic families accept a callable/nn.Module for the base function.
* Stochastic functions follow the survey's stated training/test behavior where
  stated. When the paper does not specify inference behavior, this is documented
  next to the implementation.
* Infinite series / fractional derivatives are represented by finite truncations
  controlled by a `terms` hyperparameter; this is the only practical finite
  PyTorch realization of the printed infinite sums.
* Some survey formulas are internally inconsistent or under-specified. Those
  cases are preserved as literally as possible and documented in
  SURVEY_AMBIGUITIES rather than silently replaced by a different formula.

The module is designed to be autograd-compatible.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _p(v: float) -> nn.Parameter:
    return nn.Parameter(torch.tensor(float(v)))

def _const_like(x: Tensor, v: float) -> Tensor:
    return torch.as_tensor(v, dtype=x.dtype, device=x.device)

def _phi(x: Tensor) -> Tensor:
    """Standard normal PDF."""
    return torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _Phi(x: Tensor) -> Tensor:
    """Standard normal CDF."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def _hard_tanh(x: Tensor, lo: float = -1.0, hi: float = 1.0) -> Tensor:
    return torch.clamp(x, lo, hi)

def _elu_survey(x: Tensor, a: Tensor | float = 1.0) -> Tensor:
    """Survey Section 3 parameterization: negative branch (exp(x)-1)/a."""
    return torch.where(x >= 0, x, torch.expm1(x) / a)

def _prelu_survey(x: Tensor, a: Tensor) -> Tensor:
    """Survey PReLU Eq. 257 uses x/a on the negative side."""
    return torch.where(x >= 0, x, x / a)

def _msrf(x: Tensor, eps: Tensor | float) -> Tensor:
    return torch.sqrt(x * x + eps)

def _poly_eval(coeffs: Tensor, x: Tensor) -> Tensor:
    y = torch.zeros_like(x)
    p = torch.ones_like(x)
    for c in coeffs:
        y = y + c * p
        p = p * x
    return y

def _gamma(x: Tensor) -> Tensor:
    """Real Gamma including sign on negative non-integers.
    torch.lgamma returns log|Gamma|, so the sign must be restored.
    """
    mag = torch.exp(torch.lgamma(x))
    sign = torch.where(x > 0, torch.ones_like(x), torch.sign(torch.sin(math.pi*x)))
    return mag * sign

def _identity(x: Tensor) -> Tensor:
    return x

def _as_callable(fn: Callable[[Tensor], Tensor] | nn.Module) -> Callable[[Tensor], Tensor]:
    return fn


# ---------------------------------------------------------------------------
# 4.1 Transformative adaptive activation function (TAAF), Eqs. 255-256
# ---------------------------------------------------------------------------

class TAAF(nn.Module):
    def __init__(self, base: Callable[[Tensor], Tensor] | nn.Module = torch.tanh,
                 alpha=1.0, beta=1.0, gamma=0.0, delta=0.0):
        super().__init__()
        self.base = base
        self.alpha, self.beta, self.gamma, self.delta = map(_p, (alpha, beta, gamma, delta))
    def forward(self, z: Tensor) -> Tensor:
        return self.alpha * self.base(self.beta * z + self.gamma) + self.delta


# ---------------------------------------------------------------------------
# 4.2 ReLU-based adaptive family
# ---------------------------------------------------------------------------

class PReLU(nn.Module):  # 4.2.1 Eq.257
    def __init__(self, a=100.0): super().__init__(); self.a=_p(a)
    def forward(self,z): return torch.where(z>=0,z,z/self.a)

class PositivePReLU(nn.Module):  # 4.2.2 Eq.258
    def __init__(self,a=1.0): super().__init__(); self.a=_p(a)
    def forward(self,z): return torch.where(z>=0,self.a*z,torch.zeros_like(z))

class MarginReLU(nn.Module):  # 4.2.3 Eq.259
    """a can be supplied or computed from negative responses externally."""
    def __init__(self,a=-0.1,trainable=False):
        super().__init__(); self.a=_p(a) if trainable else float(a)
    def forward(self,z): return torch.maximum(z, torch.as_tensor(self.a,dtype=z.dtype,device=z.device))

class FunnelReLU(nn.Module):  # 4.2.4 Eqs.260-261
    """Depth-wise 3x3 spatial-context implementation of FunReLU.
    FunPReLU is represented by FunnelPReLU below.
    """
    def __init__(self, channels:int, kernel_size:int=3):
        super().__init__()
        if kernel_size % 2 != 1: raise ValueError("kernel_size must be odd")
        self.kernel_size=kernel_size
        self.p=nn.Parameter(torch.zeros(channels,1,kernel_size,kernel_size))
    def context(self,z):
        return F.conv2d(z,self.p,padding=self.kernel_size//2,groups=z.shape[1])
    def forward(self,z): return torch.maximum(z,self.context(z))

class FunnelPReLU(FunnelReLU):
    """Natural PReLU analogue stated by the paper: compare to context, leak below it."""
    def __init__(self, channels:int, kernel_size:int=3, a=100.0):
        super().__init__(channels,kernel_size); self.a=_p(a)
    def forward(self,z):
        t=self.context(z); return torch.where(z>=t,z,t+(z-t)/self.a)

class RPReLU(nn.Module):  # 4.2.5 Eq.262
    def __init__(self,a=0.0,b=0.0,c=0.01): super().__init__(); self.a,self.b,self.c=map(_p,(a,b,c))
    def forward(self,z): return torch.where(z>=self.a,z-self.a+self.b,self.c*(z-self.a)+self.b)

class SAU(nn.Module):  # 4.2.6 Eq.264
    def __init__(self,a=100.0,b=1.0): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z):
        term1=(1/(2*self.b))*math.sqrt(2/math.pi)*torch.exp(-0.5*self.b*self.b*z*z)
        term2=0.5*(1+1/self.a)*z
        term3=0.5*(1-1/self.a)*z*torch.erf(self.b*z/math.sqrt(2))
        return term1+term2+term3

class SMU(nn.Module):  # 4.2.7 Eq.265
    def __init__(self,a=0.1,b=1.0): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z):
        return ((1+self.a)*z + (1-self.a)*z*torch.erf(self.b*(1-self.a)*z))/2

class LeLeLU(nn.Module):  # 4.2.8 Eq.266
    def __init__(self,a=1.0): super().__init__(); self.a=_p(a)
    def forward(self,z): return torch.where(z>=0,self.a*z,0.01*self.a*z)

class PREU(nn.Module):  # 4.2.9 Eq.267
    def __init__(self,a=1.0,b=1.0): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return torch.where(z>=0,self.a*z,self.a*z*torch.exp(self.b*z))

class RTPReLU(nn.Module):  # 4.2.10 Eq.268
    def __init__(self,a=100.0,sigma=0.75): super().__init__(); self.a=_p(a); self.sigma=float(sigma)
    def forward(self,z):
        b=torch.randn_like(z)*self.sigma if self.training else torch.zeros_like(z)
        return torch.where(z+b>=0,z,z/self.a)

class ProbAct(nn.Module):  # 4.2.11 Eqs.269-270
    def __init__(self,base:Callable[[Tensor],Tensor]|nn.Module=F.relu,sigma=0.1,trainable_sigma=True):
        super().__init__(); self.base=base; self.sigma=_p(sigma) if trainable_sigma else float(sigma)
    def forward(self,z):
        return self.base(z) + torch.as_tensor(self.sigma,dtype=z.dtype,device=z.device)*torch.randn_like(z)

class AOAF(nn.Module):  # 4.2.12 Eq.271
    def __init__(self,b=0.17,c=0.17,dim=-1): super().__init__(); self.b,self.c,self.dim=float(b),float(c),dim
    def forward(self,z):
        a=z.mean(dim=self.dim,keepdim=True)
        return F.relu(z-self.b*a)+self.c*a

class DLReLU(nn.Module):  # 4.2.13 Eq.272
    def __init__(self,a=0.1,b_t=1.0): super().__init__(); self.a=float(a); self.b_t=float(b_t)
    def set_previous_mse(self,mse:float): self.b_t=float(mse)
    def forward(self,z): return torch.where(z>=0,z,self.a*self.b_t*z)

class ExpDLReLU(nn.Module):  # Eq.273
    def __init__(self,a=0.1,b_t=1.0): super().__init__(); self.a=float(a); self.b_t=float(b_t)
    def set_previous_mse(self,mse:float): self.b_t=float(mse)
    def forward(self,z):
        c=math.exp(-self.b_t)
        return torch.where(z>=0,z,self.a*c*z)

class DynamicReLU(nn.Module):  # 4.2.14 Eq.274
    def __init__(self,dim=-1): super().__init__(); self.dim=dim
    def forward(self,z):
        a=(z.amin(dim=self.dim,keepdim=True)+z.amax(dim=self.dim,keepdim=True))/2
        return torch.where(z>=a,z,a)

class FReLU(nn.Module):  # 4.2.15 Eqs.275-276 final form
    def __init__(self,b=0.0): super().__init__(); self.b=_p(b)
    def forward(self,z): return F.relu(z)+self.b


class FlexibleReLUFull(nn.Module):  # 4.2.15 Eq.275, before the simplified Eq.276
    def __init__(self,a=0.0,b=0.0):
        super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return F.relu(z+self.a)+self.b

class ShiLU(nn.Module):  # 4.2.16 Eq.277
    def __init__(self,a=1.0,b=0.0): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return self.a*F.relu(z)+self.b

class StarReLU(nn.Module):  # 4.2.17 Eq.278
    def __init__(self,a=0.8944,b=-0.4472): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return self.a*F.relu(z).pow(2)+self.b

class AdaptiveHardTanh(nn.Module):  # 4.2.18 Eq.279
    def __init__(self,b=0.0,a=1.0): super().__init__(); self.b=_p(b); self.a=float(a)
    def set_epoch_scale(self,a_t:float): self.a=float(a_t)
    def forward(self,z): return _hard_tanh(self.a*(z-self.b))

class AReLU(nn.Module):  # 4.2.19 Eq.280
    def __init__(self,a=0.9,b=2.0): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z):
        return torch.where(z>=0,(1+torch.sigmoid(self.b))*z,torch.clamp(self.a,0.01,0.99)*z)

class DPReLU(nn.Module):  # 4.2.20 Eq.281
    def __init__(self,a=1.0,b=0.01): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return torch.where(z>=0,self.a*z,self.b*z)

class DualLine(nn.Module):  # 4.2.21 Eq.282
    def __init__(self,a=1.0,b=0.01,m=-0.22): super().__init__(); self.a,self.b,self.m=map(_p,(a,b,m))
    def forward(self,z): return torch.where(z>=0,self.a*z+self.m,self.b*z+self.m)

class PiLU(nn.Module):  # 4.2.22 Eq.283
    def __init__(self,a=1.0,b=0.01,c=0.0): super().__init__(); self.a,self.b,self.c=map(_p,(a,b,c))
    def forward(self,z):
        return torch.where(z>=self.c,self.a*z+self.c*(1-self.a),self.b*z+self.c*(1-self.b))

class DPAF(nn.Module):  # 4.2.23 Eq.284
    def __init__(self,g:Callable[[Tensor],Tensor]|nn.Module=torch.tanh,a=1.0,m=0.0):
        super().__init__(); self.g=g; self.a,self.m=map(_p,(a,m))
    def forward(self,z):
        gz=self.g(z); return torch.where(z>=0,self.a*gz+self.m,gz+self.m)

class FPAF(nn.Module):  # 4.2.24 Eq.285
    def __init__(self,g1:Callable[[Tensor],Tensor]|nn.Module=F.relu,g2:Callable[[Tensor],Tensor]|nn.Module=torch.tanh,a=1,b=1):
        super().__init__(); self.g1,self.g2=g1,g2; self.a,self.b=map(_p,(a,b))
    def forward(self,z): return torch.where(z>=0,self.a*self.g1(z),self.b*self.g2(z))

class EPReLU(nn.Module):  # 4.2.25 Eq.286
    def __init__(self,a=100.0,alpha=0.1): super().__init__(); self.a=_p(a); self.alpha=float(alpha)
    def forward(self,z):
        k=torch.empty_like(z).uniform_(1-self.alpha,1+self.alpha) if self.training else torch.ones_like(z)
        return torch.where(z>=0,k*z,z/self.a)

class PairedReLU(nn.Module):  # 4.2.26 Eq.287
    def __init__(self,a=.5,b=0,c=-.5,d=0,dim=-1):
        super().__init__(); self.a,self.b,self.c,self.d=map(_p,(a,b,c,d)); self.dim=dim
    def forward(self,z): return torch.cat([F.relu(self.a*z-self.b),F.relu(self.c*z-self.d)],dim=self.dim)

class Tent(nn.Module):  # 4.2.27 Eq.288
    def __init__(self,a=1.0): super().__init__(); self.a=_p(a)
    def forward(self,z): return F.relu(self.a-z.abs())

class Hat(nn.Module):  # 4.2.28 Eq.289
    """Literal Eq.289 from the survey.
    Both middle branches are printed as z, so algebraically this is z on [0,a]
    and zero outside. We intentionally do not replace the second z by (a-z).
    """
    def __init__(self,a=2.0): super().__init__(); self.a=_p(a)
    def forward(self,z):
        return torch.where((z>=0)&(z<=self.a),z,torch.zeros_like(z))

class RMAF(nn.Module):  # 4.2.29 Eq.290
    def __init__(self,a=1.0,b=1.0,c=1.0): super().__init__(); self.a=_p(a); self.b,self.c=float(b),float(c)
    def forward(self,z):
        base=self.b/(0.25*(1+torch.exp(-z))+0.75)**self.c
        return base*self.a*z

class PTELU(nn.Module):  # 4.2.30 Eq.291
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return torch.where(z>=0,z,self.a*torch.tanh(self.b*z))

class TaLU(nn.Module):  # 4.2.31 Eq.292
    def __init__(self,a=-0.75): super().__init__(); self.a=_p(a)
    def forward(self,z):
        return torch.where(z>=0,z,torch.where(z>self.a,torch.tanh(z),torch.tanh(self.a)))

class PTaLU(nn.Module):  # 4.2.32 Eq.293
    def __init__(self,a=-0.75,b=1.0): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z):
        return torch.where(z>=self.b,z,torch.where(z>self.a,torch.tanh(z),torch.tanh(self.a)))

class TanhLU(nn.Module):  # 4.2.33 Eq.294
    def __init__(self,a=1,b=1,c=0): super().__init__(); self.a,self.b,self.c=map(_p,(a,b,c))
    def forward(self,z): return self.a*torch.tanh(self.b*z)+self.c*z

class TeLU(nn.Module):  # 4.2.34 Eq.295
    def __init__(self,a=1): super().__init__(); self.a=_p(a)
    def forward(self,z): return z*torch.tanh(_elu_survey(self.a*z))

class TReLU(nn.Module):  # 4.2.35 Eq.296
    def __init__(self,b=1): super().__init__(); self.b=_p(b)
    def forward(self,z): return torch.where(z>=0,z,torch.tanh(self.b*z))

class TReLU2(nn.Module):  # Eq.297
    def __init__(self,a=1.0,trainable=False): super().__init__(); self.a=_p(a) if trainable else float(a)
    def forward(self,z): return torch.where(z>=0,z,torch.as_tensor(self.a,dtype=z.dtype,device=z.device)*torch.tanh(z))

class ReLTanh(nn.Module):  # 4.2.36 Eqs.298-299
    def __init__(self,a=-1.5,b=0.0): super().__init__(); self.a,self.b=map(_p,(a,b))
    @staticmethod
    def dtanh(x): return 1-torch.tanh(x).pow(2)
    def forward(self,z):
        lo=self.dtanh(self.a)*(z-self.a)+torch.tanh(self.a)
        hi=self.dtanh(self.b)*(z-self.b)+torch.tanh(self.b)
        return torch.where(z<=self.a,lo,torch.where(z>=self.b,hi,torch.tanh(z)))

class BLU(nn.Module):  # 4.2.37 Eq.300
    def __init__(self,a=0): super().__init__(); self.a=_p(a)
    def forward(self,z): return self.a*(torch.sqrt(z*z+1)-1)+z

class ReBLU(nn.Module):  # 4.2.38 Eq.301
    def __init__(self,a=0): super().__init__(); self.a=_p(a)
    def forward(self,z):
        y=self.a*(torch.sqrt(z*z+1)-1)+z
        return torch.where(z>0,y,torch.zeros_like(z))

class DELU(nn.Module):  # 4.2.39 Eq.302
    def __init__(self,a=0.5): super().__init__(); self.a=_p(a)
    def forward(self,z):
        return torch.where(z>=0,(self.a+0.5)*z+torch.abs(torch.exp(-z)-1),z*torch.sigmoid(z))

class SCMish(nn.Module):  # 4.2.40 Eq.303 (adaptive SCL-mish by default)
    def __init__(self,a=.25,trainable=True): super().__init__(); self.a=_p(a) if trainable else float(a)
    def forward(self,z):
        a=torch.as_tensor(self.a,dtype=z.dtype,device=z.device)
        return F.relu(z*torch.tanh(F.softplus(a*z)))

class SCSwish(nn.Module):  # 4.2.41 Eq.304
    def forward(self,z): return F.relu(z*torch.sigmoid(z))

class ParametricSwish(nn.Module):  # 4.2.42 Eq.305
    def __init__(self,a=1,b=1,c=0): super().__init__(); self.a,self.b,self.c=map(_p,(a,b,c))
    def forward(self,z): return torch.where(z<=self.c,self.a*z*torch.sigmoid(self.b*z),z)

class PELU(nn.Module):  # 4.2.43 Eq.307
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z):
        return torch.where(z>=0,(self.a/self.b)*z,self.a*torch.expm1(z/self.b))


class PELUFull(nn.Module):  # 4.2.43 Eq.306, unconstrained 3-parameter form
    def __init__(self,a=1,b=1,c=1):
        super().__init__(); self.a,self.b,self.c=map(_p,(a,b,c))
    def forward(self,z):
        return torch.where(z>=0,self.c*z,self.a*torch.expm1(z/self.b))

class EDELU(nn.Module):  # 4.2.44 Eqs.308-309
    def __init__(self,a=1,b=1,c=0): super().__init__(); self.a,self.b,self.c=map(_p,(a,b,c))
    def forward(self,z): return torch.where(z>=self.c,z,torch.expm1(self.a*z)/self.b)
    def constraint_residual(self): return self.b*self.c-torch.expm1(self.a*self.c)

class MixLReLUElu(nn.Module):  # 4.2.45 Eq.310
    def __init__(self,a=.5,leak=.01,elu_a=1): super().__init__(); self.a=_p(a); self.leak=leak; self.elu_a=elu_a
    def forward(self,z):
        lr=torch.where(z>=0,z,self.leak*z); el=_elu_survey(z,self.elu_a)
        return self.a*lr+(1-self.a)*el

class GatedPReLUPELU(nn.Module):  # Eq.311
    def __init__(self,gate=1,prelu_a=100,pelu_a=1,pelu_b=1):
        super().__init__(); self.gate=_p(gate); self.prelu=PReLU(prelu_a); self.pelu=PELU(pelu_a,pelu_b)
    def forward(self,z):
        s=torch.sigmoid(self.gate*z); return s*self.prelu(z)+(1-s)*self.pelu(z)

class FELU(nn.Module):  # 4.2.46 Eq.312
    def __init__(self,a=1): super().__init__(); self.a=_p(a)
    def forward(self,z):
        neg=self.a*(torch.pow(_const_like(z,2.0),z/math.log(2.0))-1)
        return torch.where(z>=0,z,neg)

class PPlusFELU(nn.Module):  # 4.2.47 Eq.313
    def __init__(self,a=1,b=0): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z):
        neg=self.a*(torch.pow(_const_like(z,2.0),z/math.log(2.0))-1)+self.b
        return torch.where(z>=0,z+self.b,neg)

class MPELU(nn.Module):  # 4.2.48 Eq.314
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return torch.where(z>=0,z,self.a*torch.expm1(self.b*z))

class PE2ReLU(nn.Module):  # 4.2.49 Eq.315
    def __init__(self,a=.4,b=.3): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z):
        return self.a*F.relu(z)+self.b*_elu_survey(z)+(1-self.a-self.b)*(-_elu_survey(-z))

class PE2Id(nn.Module):  # Eq.316
    def __init__(self,a=.5): super().__init__(); self.a=_p(a)
    def forward(self,z): return self.a*z+(1-self.a)*(_elu_survey(z)-_elu_survey(-z))

class PE2ReLU1(nn.Module):  # Eq.317
    def __init__(self,a=.5): super().__init__(); self.a=_p(a)
    def forward(self,z): return self.a*F.relu(z)+(1-self.a)*(_elu_survey(z)-_elu_survey(-z))

class SoftExponential(nn.Module):  # 4.2.50 Eq.318
    def __init__(self,a=0): super().__init__(); self.a=_p(a)
    def forward(self,z):
        a=self.a
        # Eq.318 is continuously differentiable at a=0. Direct evaluation gives
        # 0/0 there, so use its first-order expansion exactly at/very near zero:
        # f(z,a)=z + a*(1 + z^2/2) + O(a^2).
        if abs(float(a.detach())) < 1e-7:
            return z + a*(1.0 + 0.5*z*z)
        if float(a.detach()) > 0:
            return torch.expm1(a*z)/a + a
        return -torch.log1p(-a*(z+a))/a

class CELU(nn.Module):  # 4.2.51 Eq.319
    def __init__(self,a=1): super().__init__(); self.a=_p(a)
    def forward(self,z): return torch.where(z>=0,z,self.a*torch.expm1(z/self.a))

class ErfReLU(nn.Module):  # 4.2.52 Eq.320
    def __init__(self,a=1): super().__init__(); self.a=_p(a)
    def forward(self,z): return torch.where(z>=0,z,self.a*torch.erf(z))

class PSELU(nn.Module):  # 4.2.53 Eq.321
    def __init__(self,a=1.05078,b=1.6733): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return torch.where(z>=0,self.a*z,self.a*self.b*torch.expm1(z))

class LPSELU(nn.Module):  # 4.2.54 Eq.322
    def __init__(self,a=1.05,b=1.67,c=.01): super().__init__(); self.a,self.b,self.c=map(_p,(a,b,c))
    def forward(self,z): return torch.where(z>=0,self.a*z,self.a*self.b*torch.expm1(z)+self.c*z)

class LPSELURP(nn.Module):  # 4.2.55 Eq.323
    def __init__(self,a=1.05,b=1.67,c=.01,m=0): super().__init__(); self.a,self.b,self.c,self.m=map(_p,(a,b,c,m))
    def forward(self,z): return torch.where(z>=0,self.a*z+self.m,self.a*self.b*torch.expm1(z)+self.c*z+self.m)

class ShELU(nn.Module):  # 4.2.56 Eq.324
    def __init__(self,a=1,b=0): super().__init__(); self.a=float(a); self.b=float(b)
    def forward(self,z):
        x=z+self.b; return torch.where(x>=0,x,self.a*torch.expm1(x))

class SvELU(nn.Module):  # Eq.325
    def __init__(self,a=1,b=0): super().__init__(); self.a=float(a); self.b=float(b)
    def forward(self,z): return torch.where(z>=0,z+self.b,self.a*torch.expm1(z)+self.b)

class PShELU(nn.Module):  # Eq.326
    def __init__(self,a=1,b=1,c=0): super().__init__(); self.a,self.b,self.c=map(_p,(a,b,c))
    def forward(self,z):
        x=z+self.c; return torch.where(x>=0,(self.a/self.b)*x,self.a*torch.expm1(x/self.b))

class PSvELU(nn.Module):  # Eq.327, proposed by survey authors
    def __init__(self,a=1,b=1,c=0): super().__init__(); self.a,self.b,self.c=map(_p,(a,b,c))
    def forward(self,z):
        return torch.where(z>=0,(self.a/self.b)*z+self.c,self.a*torch.expm1(z/self.b)+self.c)

class TSwish(nn.Module):  # 4.2.57 Eq.328
    def __init__(self,a=1,b=1,c=0): super().__init__(); self.a,self.b,self.c=map(_p,(a,b,c))
    def forward(self,z): return torch.where(z>=self.c,z,self.a*z*torch.sigmoid(self.b*z))

class RePSU(nn.Module):  # 4.2.58 Eqs.329-331
    def __init__(self,a=.5,b=0,c=0,d=1,e=1):
        super().__init__(); self.a,self.b,self.c,self.d,self.e=map(_p,(a,b,c,d,e))
    def rePSKU(self,z):
        expo=-torch.sign(z-self.c)*(torch.abs(z-self.c)/self.d).pow(self.e)
        core=(z-self.b)/(1+torch.exp(expo))
        return torch.where(z>=self.b,core,torch.zeros_like(z))
    def rePSHU(self,z):
        ku=self.rePSKU(z); return torch.where(z>=self.b,2*z-ku,torch.zeros_like(z))
    def forward(self,z): return self.a*self.rePSKU(z)+(1-self.a)*self.rePSHU(z)

class PDELU(nn.Module):  # 4.2.59 Eq.332
    def __init__(self,a=1,b=.9): super().__init__(); self.a=_p(a); self.b=float(b)
    def forward(self,z):
        neg=self.a*((1+(1-self.b)*z).pow(1/(1-self.b))-1)
        return torch.where(z>=0,z,neg)

class EELU(nn.Module):  # 4.2.60 Eqs.333-336
    def __init__(self,a=1,b=1,eps=.1): super().__init__(); self.a,self.b=map(_p,(a,b)); self.eps=float(eps)
    def forward(self,z):
        if self.training:
            sigma=torch.rand((),device=z.device,dtype=z.dtype)*self.eps
            s=1+sigma*torch.randn_like(z)
            k=torch.clamp(s,0,2)
        else: k=torch.ones_like(z)
        return torch.where(z>=0,k*z,self.a*torch.expm1(self.b*z))

class PFPLUS(nn.Module):  # 4.2.61 Eqs.337-338
    def __init__(self,a=.2,b=10): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z):
        # H(z)-1 is 0 for z>=0 and -1 for z<0
        return torch.where(z>=0,self.a*z,self.a*z/(1-self.b*z))

class PVLU(nn.Module):  # 4.2.62 Eq.339
    def __init__(self,a=.1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return F.relu(z)+self.a*torch.sin(self.b*z)


# ---------------------------------------------------------------------------
# 4.3 Sigmoid-based adaptive functions
# ---------------------------------------------------------------------------

class ShapeAutotuningSigmoid(nn.Module):  # Eq.340
    def __init__(self,a=1): super().__init__(); self.a=_p(a)
    def forward(self,z):
        return 2*(1-torch.exp(-self.a*z))/(self.a*(1+torch.exp(-self.a*z)))

class GeneralizedTanh(nn.Module):  # 4.3.1 Eq.341
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return self.a*(1-torch.exp(-self.b*z))/(1+torch.exp(-self.b*z))

class TrainableAmplitude(nn.Module):  # 4.3.2 Eq.342
    def __init__(self,g:Callable[[Tensor],Tensor]|nn.Module=torch.tanh,a=1,b=0):
        super().__init__(); self.g=g; self.a,self.b=map(_p,(a,b))
    def forward(self,z): return self.a*self.g(z)+self.b

class ASSF(nn.Module):  # 4.3.3 Eq.343
    def __init__(self,a=1): super().__init__(); self.a=_p(a)
    def forward(self,z): return torch.sigmoid(self.a*z)

class SVAF(nn.Module):  # 4.3.4 Eq.344
    def __init__(self,a=1): super().__init__(); self.a=_p(a)
    def forward(self,z): return torch.tanh(self.a*z)

class TanhSoft(nn.Module):  # 4.3.5 Eq.345
    def __init__(self,a=1,b=0,c=1,d=1): super().__init__(); self.a,self.b,self.c,self.d=map(_p,(a,b,c,d))
    def forward(self,z): return torch.tanh(self.a*z+self.b*torch.exp(self.c*z))*torch.log(self.d+torch.exp(z))

class TanhSoft1(nn.Module):  # Eq.346
    def __init__(self,a=1): super().__init__(); self.a=_p(a)
    def forward(self,z): return torch.tanh(self.a*z)*F.softplus(z)

class TanhSoft2(nn.Module):  # Eq.347
    def __init__(self,b=1,c=1): super().__init__(); self.b,self.c=map(_p,(b,c))
    def forward(self,z): return z*torch.tanh(self.b*torch.exp(self.c*z))

class TanhSoft3(nn.Module):  # Eq.348
    def __init__(self,a=1): super().__init__(); self.a=_p(a)
    def forward(self,z): return torch.log1p(torch.exp(z)*torch.tanh(self.a*z))

class PSigmoid(nn.Module):  # 4.3.6 Eq.349
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return self.a*torch.sigmoid(self.b*z)

class ParametricSigmoidFunction(nn.Module):  # 4.3.7 Eq.350
    def __init__(self,m=1): super().__init__(); self.m=_p(m)
    def forward(self,z): return (1+torch.exp(-z)).pow(-self.m)

class STACTanh(nn.Module):  # 4.3.8 Eq.351
    def __init__(self,a=1,b=.1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z):
        lo=torch.tanh(-self.a)+self.b*(z+self.a)
        hi=torch.tanh(self.a)+self.b*(z-self.a)
        return torch.where(z<-self.a,lo,torch.where(z>self.a,hi,torch.tanh(z)))

class GRA(nn.Module):  # 4.3.9 Eq.352
    def __init__(self,a=1,b=1,c=1): super().__init__(); self.a,self.b,self.c=map(_p,(a,b,c))
    def forward(self,z): return 1-self.a/(self.a+(1+self.b*z).pow(self.c))


# ---------------------------------------------------------------------------
# 4.4 Adaptive sigmoid-weighted linear units
# ---------------------------------------------------------------------------

class Swish(nn.Module):  # 4.4.1 Eq.353
    def __init__(self,a=1): super().__init__(); self.a=_p(a)
    def forward(self,z): return z*torch.sigmoid(self.a*z)

class AHAF(nn.Module):  # 4.4.2 Eq.354
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return self.a*z*torch.sigmoid(self.b*z)

class PSSiLU(nn.Module):  # 4.4.3 Eq.355
    def __init__(self,a=1,b=0): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return z*(torch.sigmoid(self.a*z)-self.b)/(1-self.b)

class ESwish(nn.Module):  # 4.4.4 Eq.356
    def __init__(self,a=1.5,trainable=True): super().__init__(); self.a=_p(a) if trainable else float(a)
    def forward(self,z): return torch.as_tensor(self.a,dtype=z.dtype,device=z.device)*z*torch.sigmoid(z)

class ACONB(nn.Module):  # 4.4.5 Eq.357
    def __init__(self,a=1,b=.25): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z):
        d=1-self.b; return d*z*torch.sigmoid(self.a*d*z)+self.b*z

class ACONC(nn.Module):  # 4.4.6 Eq.358
    def __init__(self,a=1,b=0,c=1): super().__init__(); self.a,self.b,self.c=map(_p,(a,b,c))
    def forward(self,z):
        d=self.c-self.b; return d*z*torch.sigmoid(self.a*d*z)+self.b*z

class MetaACONC(nn.Module):
    """Formula-level MetaACON-C: caller supplies a hypernetwork producing alpha."""
    def __init__(self,alpha_net:nn.Module,b=0,c=1):
        super().__init__(); self.alpha_net=alpha_net; self.b,self.c=map(_p,(b,c))
    def forward(self,z):
        a=self.alpha_net(z); d=self.c-self.b
        return d*z*torch.sigmoid(a*d*z)+self.b*z

class PSGU(nn.Module):  # 4.4.7 Eq.359
    def __init__(self,a=.5): super().__init__(); self.a=_p(a)
    def forward(self,z): return z*torch.tanh(self.a*torch.sigmoid(z))

class TBSReLULearnable(nn.Module):  # 4.4.8 Eq.360
    def __init__(self,a=.5): super().__init__(); self.a=_p(a)
    def forward(self,z): return z*torch.tanh(self.a*torch.tanh(z/2))

class PATS(nn.Module):  # 4.4.9 Eqs.361-362
    def __init__(self,l=.5,u=.75): super().__init__(); self.l,self.u=float(l),float(u)
    def forward(self,z):
        if self.training: a=torch.empty((),device=z.device,dtype=z.dtype).uniform_(self.l,self.u)
        else: a=_const_like(z,(self.l+self.u)/2)  # paper omits inference rule; expected value used
        return z*torch.atan(a*math.pi*torch.sigmoid(z))

class AQuLU(nn.Module):  # 4.4.10 Eq.363
    def __init__(self,a=.25,b=.5): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z):
        lo=-self.b/self.a; hi=(1-self.b)/self.a
        return torch.where(z>=hi,z,torch.where(z>=lo,self.a*z*z+self.b*z,torch.zeros_like(z)))

class SinLU(nn.Module):  # 4.4.11 Eq.364
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return (z+self.a*torch.sin(self.b*z))*torch.sigmoid(z)

class ErfAct(nn.Module):  # 4.4.12 Eq.365
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return z*torch.erf(self.a*torch.exp(self.b*z))

class PSerf(nn.Module):  # 4.4.13 Eq.366
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return z*torch.erf(self.a*F.softplus(self.b*z))

class Swim(nn.Module):  # 4.4.14 Eq.367
    def __init__(self,a=.5): super().__init__(); self.a=_p(a)
    def forward(self,z): return .5*z*(1+self.a*z/torch.sqrt(1+z*z))


# ---------------------------------------------------------------------------
# 4.5 - 4.18
# ---------------------------------------------------------------------------

class TunedSoftmax(nn.Module):  # 4.5 Eq.368
    def __init__(self,c=0,dim=-1): super().__init__(); self.c=_p(c); self.dim=dim
    def forward(self,z):
        e=torch.exp(z); return e/(e.sum(dim=self.dim,keepdim=True)+self.c)

class GeneralizedLehmerSoftmax(nn.Module):  # 4.6 Eqs.369-372
    def __init__(self,a=2,b=1,c=2,d=1,dim=-1):
        super().__init__(); self.a,self.b,self.c,self.d=map(_p,(a,b,c,d)); self.dim=dim
    def glm(self,x,alpha,beta):
        num=(alpha.pow((beta+1)*x)).sum(dim=self.dim,keepdim=True)
        den=(alpha.pow(beta*x)).sum(dim=self.dim,keepdim=True)
        return torch.log(num/den)/torch.log(alpha)
    def norm(self,z):
        m=self.glm(z,self.a,self.b)
        return (z-m)/self.glm(z-m,self.c,self.d)
    def forward(self,z): return torch.softmax(self.norm(z),dim=self.dim)

class GeneralizedPowerSoftmax(nn.Module):  # 4.7 Eqs.373-376
    def __init__(self,a=2,b=1,c=2,d=1,dim=-1):
        super().__init__(); self.a,self.b,self.c,self.d=map(_p,(a,b,c,d)); self.dim=dim
    def gpm(self,x,alpha,beta):
        n=x.shape[self.dim]
        return (torch.log((alpha.pow(beta*x)).sum(dim=self.dim,keepdim=True))-math.log(n))/(beta*torch.log(alpha))
    def norm(self,z):
        m=self.gpm(z,self.a,self.b)
        return (z-m)/self.gpm(z-m,self.c,self.d)
    def forward(self,z): return torch.softmax(self.norm(z),dim=self.dim)

class ARBF(nn.Module):  # 4.8 Eq.377
    def __init__(self,a=0,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return torch.exp(-(z-self.a).pow(2)/(2*self.b.pow(2)))

class PGELU(nn.Module):  # 4.9 Eq.378
    def __init__(self,a=1): super().__init__(); self.a=_p(a)
    def forward(self,z): return z*_Phi(z/self.a)

class PFTS(nn.Module):  # 4.10 Eq.379
    def __init__(self,T=-.2): super().__init__(); self.T=_p(T)
    def forward(self,z): return F.relu(z)*torch.sigmoid(z)+self.T

class PFPM(nn.Module):  # 4.11 Eq.380
    def __init__(self,p=-.2): super().__init__(); self.p=_p(p)
    def forward(self,z): return torch.where(z>=0,z*torch.tanh(F.softplus(z))+self.p,self.p.expand_as(z))

class GEU(nn.Module):  # 4.12 Eq.381
    def __init__(self,a=1): super().__init__(); self.a=_p(a)
    def forward(self,z): return _Phi(z/self.a)

class SGT(nn.Module):  # 4.13 Eq.382
    def __init__(self,a=1,b=1,c=1,d=1): super().__init__(); self.a,self.c=float(a),float(c); self.b,self.d=map(_p,(b,d))
    def forward(self,z): return torch.where(z>=0,self.a*z.pow(self.b),self.c*z.pow(self.d))

class RSign(nn.Module):  # 4.14 Eq.383
    def __init__(self,a=0): super().__init__(); self.a=_p(a)
    def forward(self,z): return torch.where(z>=self.a,torch.ones_like(z),-torch.ones_like(z))

class PSIGRAMP(nn.Module):  # 4.15 Eq.384
    def __init__(self,a=.5,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z):
        t=1/(2*self.b)
        ramp=torch.where(z>=t,torch.ones_like(z),torch.where(z<=-t,torch.zeros_like(z),self.b*z+.5))
        return self.a*torch.sigmoid(z)+(1-self.a)*ramp

class LAAF(nn.Module):  # 4.16 Eqs.385,390
    def __init__(self,g:Callable[[Tensor],Tensor]|nn.Module=torch.tanh,a=1,n=1):
        super().__init__(); self.g=g; self.a=_p(a); self.n=float(n)
    def forward(self,z): return self.g(self.n*self.a*z)

class LAAFSigmoid(LAAF):  # Eq.386
    def __init__(self,a=1,n=1): super().__init__(torch.sigmoid,a,n)

class LAAFTanh(LAAF):  # Eq.387 / 391
    def __init__(self,a=1,n=1): super().__init__(torch.tanh,a,n)

class LAAFReLU(LAAF):  # Eq.388
    def __init__(self,a=1,n=1): super().__init__(F.relu,a,n)

class LAAFLeakyReLU(nn.Module):  # Eq.389
    def __init__(self,a=1,b=.01,n=1): super().__init__(); self.a=_p(a); self.b=float(b); self.n=float(n)
    def forward(self,z):
        x=self.n*self.a*z; return F.relu(x)-self.b*F.relu(-x)

class PSTanh(nn.Module):  # 4.16.2 Eq.392
    def __init__(self,a=.5,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return z*self.a*(1+torch.tanh(self.b*z))

class SSinH(nn.Module):  # 4.16.3 Eq.393
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return self.a*torch.sinh(self.b*z)

class SExp(nn.Module):  # 4.16.4 Eq.394
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return self.a*torch.expm1(self.b*z)

class LAU(nn.Module):  # 4.16.5 Eq.395
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return z*torch.log1p(self.a*torch.sigmoid(self.b*z))

class CosLU(nn.Module):  # 4.16.6 Eq.396
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return (z+self.a*torch.cos(self.b*z))*torch.sigmoid(z)

class AGumb(nn.Module):  # 4.16.7 Eq.397
    def __init__(self,a=1): super().__init__(); self.a=_p(a)
    def forward(self,z): return 1-(1+self.a*torch.exp(z)).pow(-1/self.a)

class SAAAF(nn.Module):  # 4.17 Eq.398
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z):
        # parsed formula: z / (z/a + exp(-z/b))
        return z/(z/self.a+torch.exp(-z/self.b))

class NoisyActivation(nn.Module):  # 4.18 Eq.399
    def __init__(self,h:Callable[[Tensor],Tensor]|nn.Module=torch.tanh,
                 linearization:Callable[[Tensor],Tensor]|nn.Module=_identity,
                 a=1,c=1,p=1,half_normal=True):
        super().__init__(); self.h,self.u=h,linearization; self.a=float(a); self.c=float(c); self.p=_p(p); self.half=half_normal
    def forward(self,z):
        hz,uz=self.h(z),self.u(z)
        eps=torch.randn_like(z)
        if self.half: eps=eps.abs()
        noise=torch.sigmoid(self.p*(hz-uz)).pow(2)*eps
        sign_1ma = 0.0 if self.a == 1.0 else (1.0 if (1.0-self.a) > 0 else -1.0)
        return self.a*hz+(1-self.a)*uz-torch.sign(z)*sign_1ma*self.c*noise

class NoisyInputActivation(nn.Module):  # Eqs.400-401
    def __init__(self,h:Callable[[Tensor],Tensor]|nn.Module=torch.tanh,
                 linearization:Callable[[Tensor],Tensor]|nn.Module=_identity,
                 c=1,p=1,b:Optional[float]=None):
        super().__init__(); self.h,self.u=h,linearization; self.c=float(c); self.p=_p(p); self.b=b
    def forward(self,z):
        eps=torch.randn_like(z)
        if self.b is None:
            s=self.c*torch.sigmoid(self.p*(self.h(z)-self.u(z))).pow(2)
        else: s=self.b
        return self.h(z+s*eps)


# ---------------------------------------------------------------------------
# 4.19 Fractional adaptive activation functions
# ---------------------------------------------------------------------------


class FractionalAdaptiveActivation(nn.Module):  # 4.19 Eq.402, executable GL realization
    """General g(z)=D^a f(z), using a finite Grünwald-Letnikov approximation."""
    def __init__(self,base:Callable[[Tensor],Tensor]|nn.Module=torch.tanh,a=.5,h=.05,terms=32):
        super().__init__(); self.base=base; self.a=_p(a); self.h=float(h); self.terms=int(terms)
    def forward(self,z):
        return _grunwald_letnikov(self.base,z,self.a,self.h,self.terms)

class FractionalReLU(nn.Module):  # 4.19.1 Eq.403
    def __init__(self,a=.5): super().__init__(); self.a=_p(a)
    def forward(self,z): return z.pow(1-self.a)/_gamma(2-self.a)

def _grunwald_letnikov(base:Callable[[Tensor],Tensor], z:Tensor, a:Tensor, h:float, terms:int):
    s=torch.zeros_like(z)
    for n in range(terms):
        n_t=_const_like(z,float(n))
        coef=(-1.0 if n%2 else 1.0)*_gamma(a+1)/(_gamma(n_t+1)*_gamma(1-n_t+a))
        s=s+coef*base(z-n*h)
    return s/(h**a)

class FractionalSoftplus(nn.Module):  # 4.19.2 Eqs.404-405
    def __init__(self,a=.5,h=.05,terms=32): super().__init__(); self.a=_p(a); self.h=float(h); self.terms=int(terms)
    def forward(self,z): return _grunwald_letnikov(F.softplus,z,self.a,self.h,self.terms)

class FractionalTanh(nn.Module):  # 4.19.3 Eqs.406-407
    def __init__(self,a=.5,h=.05,terms=32): super().__init__(); self.a=_p(a); self.h=float(h); self.terms=int(terms)
    def forward(self,z): return _grunwald_letnikov(torch.tanh,z,self.a,self.h,self.terms)

class FALU(nn.Module):  # 4.19.4 Eqs.408,410-412; practical approximation
    def __init__(self,a=.5,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z):
        g=z*torch.sigmoid(self.b*z)
        h=g+torch.sigmoid(z)*(1-g)
        lo=g+self.a*torch.sigmoid(self.b*z)*(1-g)
        hi=g+self.a*torch.sigmoid(self.b*z)*(1-2*h)
        return torch.where(self.a<=1,lo,hi)


class FALUExact(nn.Module):  # 4.19.4 Eqs.408-409 via finite GL truncation
    def __init__(self,a=.5,b=1,h=.05,terms=64):
        super().__init__(); self.a,self.b=map(_p,(a,b)); self.h=float(h); self.terms=int(terms)
    def forward(self,z):
        return _grunwald_letnikov(lambda x: x*torch.sigmoid(self.b*x),z,self.a,self.h,self.terms)

class FractionalLeakyReLU(nn.Module):  # 4.19.5 Eq.414
    def __init__(self,a=.5,b=.1): super().__init__(); self.a=float(a); self.b=float(b)
    def forward(self,z):
        base=z.pow(1-self.a)/math.gamma(2-self.a)
        return torch.where(z>=0,base,self.b*base)

class FractionalPReLU(nn.Module):  # 4.19.6 Eq.416
    def __init__(self,a=.5,b=.1): super().__init__(); self.a=float(a); self.b=_p(b)
    def forward(self,z):
        base=z.pow(1-self.a)/math.gamma(2-self.a)
        return torch.where(z>=0,base,self.b*base)

class FractionalELU(nn.Module):  # 4.19.7 Eq.418 truncated
    def __init__(self,a=.5,b=1,terms=32): super().__init__(); self.a=float(a); self.b=float(b); self.terms=int(terms)
    def forward(self,z):
        pos=z.pow(1-self.a)/math.gamma(2-self.a)
        s=torch.zeros_like(z)
        for k in range(self.terms):
            s=s + z.pow(k-self.a)/math.gamma(k+1-self.a)
        neg=self.b*s-self.b*z.pow(-self.a)/math.gamma(1-self.a)
        return torch.where(z>=0,pos,neg)

# Bernoulli numbers needed by FracSiLU series. Values B0..B20.
_BERNOULLI=[1.0,-0.5,1/6,0,-1/30,0,1/42,0,-1/30,0,5/66,0,-691/2730,0,7/6,0,-3617/510,0,43867/798,0,-174611/330]

class FractionalSiLU1(nn.Module):  # 4.19.8 Eq.421 truncated
    def __init__(self,a=.5,terms=12): super().__init__(); self.a=float(a); self.terms=min(int(terms),len(_BERNOULLI)-1)
    def _series(self,z):
        s=torch.zeros_like(z)
        for k in range(self.terms):
            B=_BERNOULLI[k+1]
            coef=(((-1)**k)+(2**(k+1)-1)*B*math.gamma(k+2)/(math.gamma(k+2-self.a)*math.factorial(k+1)))
            s=s+coef*z.pow(k+1-self.a)
        return s
    def forward(self,z):
        pos=z.pow(1-self.a)/math.gamma(2-self.a)
        return torch.where(z>=0,pos,self._series(z))

class FractionalSiLU2(FractionalSiLU1):  # Eq.423
    def forward(self,z): return self._series(z)

class FractionalGELU1(nn.Module):  # 4.19.9 Eq.426 truncated
    def __init__(self,a=.5,terms=12): super().__init__(); self.a=float(a); self.terms=int(terms)
    def _series(self,z):
        s=.5*z.pow(1-self.a)/math.gamma(2-self.a)
        t=torch.zeros_like(z)
        for k in range(self.terms):
            coef=(1/math.factorial(k))*((-0.5)**k)/(2*k+1)*math.gamma(2*k+3)/math.gamma(2*k+3-self.a)
            t=t+coef*z.pow(2*(k+1)-self.a)
        return s-t/math.sqrt(2*math.pi)
    def forward(self,z):
        pos=z.pow(1-self.a)/math.gamma(2-self.a)
        return torch.where(z>=0,pos,self._series(z))

class FractionalGELU2(FractionalGELU1):  # Eq.428
    def forward(self,z): return self._series(z)


# ---------------------------------------------------------------------------
# 4.20 - 4.48
# ---------------------------------------------------------------------------

class ScaledSoftsign(nn.Module):  # 4.20 Eq.429
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return self.a*z/(self.b+z.abs())

class ParameterizedSoftplus(nn.Module):  # 4.21 Eq.430
    def __init__(self,a=.5): super().__init__(); self.a=_p(a)
    def forward(self,z): return F.softplus(z)-self.a

class UAF(nn.Module):  # 4.22 Eq.431
    def __init__(self,a=1,b=0,c=0,d=1,e=0): super().__init__(); self.a,self.b,self.c,self.d,self.e=map(_p,(a,b,c,d,e))
    def forward(self,z):
        return F.softplus(self.a*(z+self.b)+self.c*z*z)-F.softplus(self.d*(z-self.b))+self.e

class LEAF(nn.Module):  # 4.23 Eq.432
    def __init__(self,a=1,b=0,c=1,d=0): super().__init__(); self.a,self.b,self.c,self.d=map(_p,(a,b,c,d))
    def forward(self,z): return (self.a*z+self.b)*torch.sigmoid(self.c*z)+self.d

class GReLU(nn.Module):  # 4.24 Eq.433
    def __init__(self,a=2,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return torch.log1p(self.a.pow(self.b*z))/(self.b*torch.log(self.a))

class MAF(nn.Module):  # 4.25 Eq.434
    def __init__(self,a=0,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return torch.sqrt((z-self.a).pow(2)+self.b.pow(2))

class EIS(nn.Module):  # 4.26 Eq.435
    def __init__(self,a=1,b=1,c=1,d=1,e=1): super().__init__(); self.a,self.b,self.c,self.d,self.e=map(_p,(a,b,c,d,e))
    def forward(self,z): return z*F.softplus(z).pow(self.a)/torch.sqrt(self.b+self.c*z*z+self.d*torch.exp(-self.e*z))

class EIS1(nn.Module):  # Eq.436
    def __init__(self,d=1,e=1): super().__init__(); self.d,self.e=map(_p,(d,e))
    def forward(self,z): return z*F.softplus(z)/(z+self.d*torch.exp(-self.e*z))

class EIS2(nn.Module):  # Eq.437
    def __init__(self,b=1,c=1): super().__init__(); self.b,self.c=map(_p,(b,c))
    def forward(self,z): return z*F.softplus(z)/torch.sqrt(self.b+self.c*z*z)

class EIS3(nn.Module):  # Eq.438
    def __init__(self,d=1,e=1): super().__init__(); self.d,self.e=map(_p,(d,e))
    def forward(self,z): return z/(1+self.d*torch.exp(-self.e*z))

class ELUS2L(nn.Module):  # 4.26.1 Eq.439
    def __init__(self,a=.5,b=.5): super().__init__(); self.soft=ParameterizedSoftplus(a); self.b=_p(b)
    def forward(self,z): return self.b*_elu_survey(z)+(1-self.b)*self.soft(z)

class GLN(nn.Module):  # 4.27 Eq.440
    def __init__(self,global_fn=torch.sin,local_fn=torch.tanh,a=0,b=0):
        super().__init__(); self.global_fn,self.local_fn=global_fn,local_fn; self.a,self.b=map(_p,(a,b))
    def forward(self,z):
        s=torch.sigmoid(self.a); return s*self.global_fn(z)+(1-s)*self.local_fn(z)-self.b

class NAF(nn.Module):  # 4.28 Eq.441
    def __init__(self,a=1,b=1,c=1,d=1): super().__init__(); self.a,self.b,self.c,self.d=map(_p,(a,b,c,d))
    def forward(self,z): return self.a*torch.exp(-self.b*z*z)+self.c/(1+torch.exp(-self.d*z))

class ScaledLogisticSigmoid(nn.Module):  # 4.28.1 Eq.442
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return self.a/(1+torch.exp(-self.b*z))

class SLS_SS(nn.Module):  # Eq.443
    def __init__(self,a=1,b=1,c=1,d=1): super().__init__(); self.a,self.b,self.c,self.d=map(_p,(a,b,c,d))
    def forward(self,z): return self.a*torch.sin(self.b*z)+self.c/(1+torch.exp(-self.d*z))

class NAFExtended(nn.Module):  # Eq.444
    def __init__(self,a=1,b=1,c=1,d=1,e=1,f=1):
        super().__init__(); self.a,self.b,self.c,self.d,self.e,self.f=map(_p,(a,b,c,d,e,f))
    def forward(self,z): return self.a*torch.sin(self.b*z)+self.c*torch.exp(-self.d*z*z)+self.e/(1+torch.exp(-self.f*z))

class APLU(nn.Module):  # 4.29 Eq.445
    def __init__(self,S=3): super().__init__(); self.a=nn.Parameter(torch.zeros(S)); self.b=nn.Parameter(torch.linspace(-1,1,S))
    def forward(self,z):
        y=F.relu(z)
        for a,b in zip(self.a,self.b): y=y+a*F.relu(-z+b)
        return y

class SPLASH(nn.Module):  # 4.30 Eq.446
    def __init__(self,hinges:Sequence[float]=(.0,.5,1.,2.)):
        super().__init__(); h=torch.tensor(tuple(hinges),dtype=torch.float32); self.register_buffer("b",h)
        self.ap=nn.Parameter(torch.ones(len(h))); self.am=nn.Parameter(torch.zeros(len(h)))
    def forward(self,z):
        y=torch.zeros_like(z)
        for ap,am,b in zip(self.ap,self.am,self.b): y=y+ap*F.relu(z-b)+am*F.relu(-z-b)
        return y

class MBA(nn.Module):  # 4.31 Eq.447
    def __init__(self,K=4,g:Callable[[Tensor],Tensor]|nn.Module=F.relu,dim=-1):
        super().__init__(); self.g=g; self.bias=nn.Parameter(torch.zeros(K)); self.dim=dim
    def forward(self,z): return torch.stack([self.g(z+b) for b in self.bias],dim=self.dim)

class MeLU(nn.Module):  # 4.32 Eqs.448-449
    def __init__(self,b:Sequence[float]=(-1,0,1),c:Sequence[float]=(1,1,1),prelu_a=100):
        super().__init__(); self.register_buffer("b",torch.tensor(b,dtype=torch.float32)); self.register_buffer("c",torch.tensor(c,dtype=torch.float32))
        self.a=nn.Parameter(torch.zeros(len(b))); self.prelu=PReLU(prelu_a)
    def forward(self,z):
        y=self.prelu(z)
        for a,b,c in zip(self.a,self.b,self.c): y=y+a*F.relu(c-(z-b).abs())
        return y

class MMeLU(nn.Module):  # 4.32.1 Eq.450
    def __init__(self,a=.5,b=1,c=0): super().__init__(); self.a,self.b,self.c=map(_p,(a,b,c))
    def forward(self,z): return self.a*F.relu(self.b-(z-self.c).abs())+(1-self.a)*F.relu(z)

class GaLUComponent(nn.Module):  # 4.32.2 Eq.451
    def __init__(self,b=0,c=1): super().__init__(); self.b=float(b); self.c=float(c)
    def forward(self,z): return F.relu(self.c-(z-self.b).abs())+torch.minimum((z-self.b-2*self.c).abs()-self.c,torch.zeros_like(z))


class GaLU(nn.Module):  # 4.32.2: Eq.448 form with Eq.451 Gaussian-ReLU components
    def __init__(self,b:Sequence[float]=(-1,0,1),c:Sequence[float]=(1,1,1),prelu_a=100):
        super().__init__()
        self.prelu=PReLU(prelu_a)
        self.a=nn.Parameter(torch.zeros(len(b)))
        self.parts=nn.ModuleList([GaLUComponent(bb,cc) for bb,cc in zip(b,c)])
    def forward(self,z):
        return self.prelu(z)+sum(a*p(z) for a,p in zip(self.a,self.parts))

class HardSwishAdaptive(nn.Module):  # 4.32.3 Eq.452
    def __init__(self,b=1): super().__init__(); self.b=_p(b)
    def forward(self,z): return 2*z*torch.clamp(.2*self.b*z+.5,0,1)

class SReLU(nn.Module):  # 4.33 Eq.453
    def __init__(self,tr=1,tl=0,ar=1,al=.1): super().__init__(); self.tr,self.tl,self.ar,self.al=map(_p,(tr,tl,ar,al))
    def forward(self,z):
        return torch.where(z>=self.tr,self.tr+self.ar*(z-self.tr),
               torch.where(z<=self.tl,self.tl+self.al*(z-self.tl),z))

class NActivation(nn.Module):  # 4.33.1 Eqs.454-456
    def __init__(self,a=-1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z):
        lo=torch.minimum(self.a,self.b); hi=torch.maximum(self.a,self.b)
        return torch.where(z<lo,z-2*lo,torch.where(z>hi,z-2*hi,-z))

class ALiSA(nn.Module):  # 4.33.2 Eq.457
    def __init__(self,ar=1,al=.1): super().__init__(); self.ar,self.al=map(_p,(ar,al))
    def forward(self,z): return torch.where(z>=1,self.ar*z-self.ar+1,torch.where(z<=0,self.al*z,z))

class AllReLU(nn.Module):  # 4.34 Eq.458
    def __init__(self,a=.1,layer_index=0): super().__init__(); self.a=float(a); self.layer_index=int(layer_index)
    def forward(self,z):
        slope=-self.a if self.layer_index%2==0 else self.a
        return torch.where(z<=0,slope*z,z)

class PLU(nn.Module):  # 4.35 Eq.459
    def __init__(self,a=.1,b=1): super().__init__(); self.a=_p(a); self.b=float(b)
    def forward(self,z): return torch.maximum(self.a*(z+self.b)-self.b,torch.minimum(self.a*(z-self.b)+self.b,z))

class AdaLU(nn.Module):  # 4.36 Eq.460
    def __init__(self,a=0,b=0,c=1,d=.1,e=-1): super().__init__(); self.a,self.b,self.c,self.d,self.e=map(_p,(a,b,c,d,e))
    def forward(self,z):
        x=z-self.a; p=self.c*x; n=self.d*x
        return torch.where((x>0)&(p>self.e),p+self.b,torch.where((x<=0)&(n>self.e),n+self.b,self.e+self.b))

class TSAF(nn.Module):  # 4.37 Eq.461
    def __init__(self,a=-1,b=1,c=1): super().__init__(); self.a,self.b,self.c=map(_p,(a,b,c))
    def forward(self,z):
        return (F.relu(z-self.a+self.c)+F.relu(z-self.a)+F.relu(z+self.b-self.c)-F.relu(z-self.b))/self.c

class RichardsCurve(nn.Module):  # Eq.462
    def __init__(self,A=0,K=1,C=1,Q=1,B=1,v=1,trainable=True):
        super().__init__()
        vals=(A,K,C,Q,B,v)
        if trainable: self.A,self.K,self.C,self.Q,self.B,self.v=map(_p,vals)
        else: self.A,self.K,self.C,self.Q,self.B,self.v=vals
    def forward(self,z):
        A,K,C,Q,B,v=[torch.as_tensor(v,dtype=z.dtype,device=z.device) for v in (self.A,self.K,self.C,self.Q,self.B,self.v)]
        return A+(K-A)/(C+Q*torch.exp(-B*z)).pow(1/v)

class ARiA(nn.Module):  # 4.38 Eq.463
    def __init__(self,**kwargs): super().__init__(); self.rc=RichardsCurve(**kwargs)
    def forward(self,z): return z*self.rc(z)

class ARiA2(nn.Module):  # Eq.464
    def __init__(self,a=1.5,b=2,trainable=True):
        super().__init__(); self.a=_p(a) if trainable else float(a); self.b=_p(b) if trainable else float(b)
    def forward(self,z):
        a=torch.as_tensor(self.a,dtype=z.dtype,device=z.device); b=torch.as_tensor(self.b,dtype=z.dtype,device=z.device)
        return z*(1+torch.exp(-b*z)).pow(-a)

class ModifiedWeibull(nn.Module):  # 4.39 Eq.465
    def __init__(self,a=1,b=2,c=1,d=2): super().__init__(); self.a,self.b,self.c,self.d=map(_p,(a,b,c,d))
    def forward(self,z): return (z/self.a).pow(self.b-1)*torch.exp(-(z/self.c).pow(self.d))

class Sincos(nn.Module):  # 4.40 Eq.466
    def __init__(self,a=1,b=1,c=1,d=1): super().__init__(); self.a,self.b,self.c,self.d=map(_p,(a,b,c,d))
    def forward(self,z): return self.a*torch.sin(self.b*z)+self.c*torch.cos(self.d*z)

class CSS(nn.Module):  # 4.41 Eq.467
    def __init__(self,a=1,b=1,c=1,d=1): super().__init__(); self.a,self.b,self.c,self.d=map(_p,(a,b,c,d))
    def forward(self,z): return self.a*torch.sin(self.b*z)+self.c*torch.sigmoid(self.d*z)

class CatAF(nn.Module):  # 4.42 Eq.468
    def __init__(self,g:Callable[[Tensor],Tensor]|nn.Module=F.relu,a=.5): super().__init__(); self.g=g; self.a=_p(a)
    def forward(self,z): return z*torch.sin(self.a)+self.g(z)*torch.cos(self.a)

class Expcos(nn.Module):  # 4.43 Eq.469
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return torch.exp(-self.a*z*z)*torch.cos(self.b*z)

class MTLU(nn.Module):  # 4.44 Eq.470
    def __init__(self,anchors:Sequence[float]=(-1,0,1)):
        super().__init__(); self.register_buffer("anchors",torch.tensor(sorted(anchors,reverse=True),dtype=torch.float32))
        K=len(anchors)+1; self.a=nn.Parameter(torch.ones(K)); self.b=nn.Parameter(torch.zeros(K))
    def forward(self,z):
        # intervals from +inf to -inf, anchors descending
        idx=torch.zeros_like(z,dtype=torch.long)
        for anchor in self.anchors: idx=idx+(z<anchor).long()
        return self.a[idx]*z+self.b[idx]

class CPN(nn.Module):  # 4.45 Eq.471
    def __init__(self,anchors:Sequence[float]=(-1,0,1),g=torch.sigmoid):
        super().__init__(); self.g=g; self.register_buffer("anchors",torch.tensor(sorted(anchors,reverse=True),dtype=torch.float32))
        K=len(anchors)+1; self.a=nn.Parameter(torch.ones(K)); self.b=nn.Parameter(torch.zeros(K)); self.c=nn.Parameter(torch.zeros(K))
    def forward(self,z):
        idx=torch.zeros_like(z,dtype=torch.long)
        for anchor in self.anchors: idx=idx+(z<anchor).long()
        return self.a[idx]*z+self.b[idx]+self.c[idx]*self.g(z)

class CPNnl(nn.Module):  # Eqs.472-473
    def __init__(self,K=4):
        super().__init__(); self.a=nn.Parameter(torch.ones(K)); self.b=nn.Parameter(torch.zeros(K)); self.c=nn.Parameter(torch.zeros(K))
    def forward(self,z):
        vals=torch.stack([a*z+b*(z*torch.sigmoid(z))+c for a,b,c in zip(self.a,self.b,self.c)],dim=0)
        return vals.max(dim=0).values

class LuTULinear(nn.Module):  # 4.46 Eq.474
    def __init__(self,a0=-12,s=.1,n=240):
        super().__init__(); self.a0=float(a0); self.s=float(s); self.n=int(n)
        self.b=nn.Parameter(torch.linspace(a0,a0+s*n,n+1))
    def forward(self,z):
        t=((z-self.a0)/self.s).clamp(0,self.n-1)
        j=torch.floor(t).long()
        aj=self.a0+self.s*j.to(z.dtype); aj1=aj+self.s
        return (self.b[j]*(aj1-z)+self.b[j+1]*(z-aj))/self.s

class LuTUCosine(nn.Module):  # Eqs.475-476
    def __init__(self,a0=-12,s=.1,n=240,t=1):
        super().__init__(); self.a0=float(a0); self.s=float(s); self.n=int(n); self.t=int(t)
        self.b=nn.Parameter(torch.linspace(a0,a0+s*n,n+1))
        self.register_buffer("anchors",torch.linspace(a0,a0+s*n,n+1))
    def r(self,x):
        tau=self.t*self.s
        return torch.where(x.abs()<=tau,(1+torch.cos(math.pi*x/tau))/(2*tau),torch.zeros_like(x))
    def forward(self,z):
        y=torch.zeros_like(z)
        for bj,aj in zip(self.b,self.anchors): y=y+bj*self.r(z-aj)
        return y

class Maxout(nn.Module):  # 4.47 Eq.477
    def __init__(self,K=4):
        super().__init__(); self.w=nn.Parameter(torch.ones(K)); self.b=nn.Parameter(torch.zeros(K))
    def forward(self,z): return torch.stack([w*z+b for w,b in zip(self.w,self.b)],0).max(0).values

class ABU(nn.Module):  # 4.48 Eq.478
    def __init__(self,functions:Sequence[Callable[[Tensor],Tensor]|nn.Module]=(torch.tanh,F.relu,_identity)):
        super().__init__(); self.functions=list(functions); self.a=nn.Parameter(torch.full((len(functions),),1/len(functions)))
    def forward(self,z):
        return sum(a*g(z) for a,g in zip(self.a,self.functions))

class ShiftedABU(nn.Module):  # Eq.479
    def __init__(self,functions=(torch.tanh,F.relu,_identity)):
        super().__init__(); self.functions=list(functions); n=len(functions); self.a=nn.Parameter(torch.ones(n)/n); self.b=nn.Parameter(torch.zeros(n))
    def forward(self,z): return sum(a*g(z-b) for a,b,g in zip(self.a,self.b,self.functions))

class MinMaxScaledABU(nn.Module):  # Eqs.480-482
    def __init__(self,functions=(torch.tanh,F.relu,_identity),eps=1e-6,dim=0):
        super().__init__(); self.functions=list(functions); self.eps=eps; self.dim=dim; self.logits=nn.Parameter(torch.zeros(len(functions)))
    def forward(self,z):
        w=torch.softmax(self.logits,0); out=0
        for a,g in zip(w,self.functions):
            y=g(z); mn=y.amin(dim=self.dim,keepdim=True); mx=y.amax(dim=self.dim,keepdim=True)
            out=out+a*(y-mn)/(mx-mn+self.eps)
        return out

class TCA(nn.Module):  # 4.48.1 Eq.483
    def __init__(self,functions=(torch.tanh,F.relu)):
        super().__init__(); self.functions=list(functions); n=len(functions); self.a=nn.Parameter(torch.zeros(n)); self.b=nn.Parameter(torch.zeros(n))
    def forward(self,z): return sum(f(torch.exp(a)*z+b) for f,a,b in zip(self.functions,self.a,self.b))/len(self.functions)

class TCAv2(nn.Module):  # Eq.484
    def __init__(self,functions=(torch.tanh,F.relu)):
        super().__init__(); self.functions=list(functions); n=len(functions); self.a=nn.Parameter(torch.zeros(n)); self.b=nn.Parameter(torch.zeros(n)); self.c=nn.Parameter(torch.zeros(n))
    def forward(self,z):
        w=torch.exp(self.a)
        return sum(wj*f(torch.exp(b)*z+c) for wj,f,b,c in zip(w,self.functions,self.b,self.c))/w.sum()

class APAF(nn.Module):  # 4.48.2 Eq.485
    def __init__(self,functions=(torch.tanh,F.relu,_identity)):
        super().__init__(); self.functions=list(functions); self.a=nn.Parameter(torch.ones(len(functions)))
    def forward(self,z): return sum(a*f(z) for a,f in zip(self.a,self.functions))/self.a.sum()

class GABU(nn.Module):  # 4.48.3 Eq.486
    def __init__(self,functions=(torch.tanh,F.relu,_identity)):
        super().__init__(); self.functions=list(functions); self.a=nn.Parameter(torch.zeros(len(functions)))
    def forward(self,z): return sum(torch.sigmoid(a)*f(z) for a,f in zip(self.a,self.functions))

class DKNNActivation(nn.Module):  # 4.48.4 Eq.487
    def __init__(self,functions=(torch.tanh,F.relu)):
        super().__init__(); self.functions=list(functions); n=len(functions); self.a=nn.Parameter(torch.ones(n)/n); self.b=nn.Parameter(torch.ones(n))
    def forward(self,z): return sum(a*f(b*z) for a,b,f in zip(self.a,self.b,self.functions))

class RowdyActivation(DKNNActivation):  # 4.48.5 Eqs.488-489
    def __init__(self,base=F.relu,n=3,c=1.0,use_cos=False):
        funcs=[base]
        for j in range(1,n+1):
            funcs.append((lambda j: (lambda z: c*(torch.cos(j*c*z) if use_cos else torch.sin(j*c*z))))(j))
        super().__init__(funcs)

class SLAF(nn.Module):  # 4.48.6 Eq.490
    def __init__(self,k=6): super().__init__(); self.a=nn.Parameter(torch.zeros(k))
    def forward(self,z): return _poly_eval(self.a,z)

class ChPAF(nn.Module):  # 4.48.7 Eqs.491-492
    def __init__(self,k=3): super().__init__(); self.a=nn.Parameter(torch.zeros(k+1))
    def forward(self,z):
        C=[torch.ones_like(z),z]
        for j in range(1,len(self.a)-1): C.append(2*z*C[-1]-C[-2])
        return sum(a*c for a,c in zip(self.a,C[:len(self.a)]))

class LPAF(nn.Module):  # 4.48.8 Eqs.493-494
    def __init__(self,k=3): super().__init__(); self.a=nn.Parameter(torch.zeros(k+1))
    def forward(self,z):
        G=[torch.ones_like(z),z]
        for n in range(1,len(self.a)-1): G.append(((2*n+1)/(n+1))*z*G[-1]-(n/(n+1))*G[-2])
        return sum(a*g for a,g in zip(self.a,G[:len(self.a)]))

class HPAF(nn.Module):  # 4.48.9 Eqs.495-497, physicists' Hermite
    def __init__(self,k=3): super().__init__(); self.a=nn.Parameter(torch.zeros(k+1))
    def forward(self,z):
        H=[torch.ones_like(z)]
        if len(self.a)>1: H.append(2*z)
        for n in range(1,len(self.a)-1): H.append(2*z*H[-1]-2*n*H[-2])
        return sum(a*h for a,h in zip(self.a,H))

class MoGU(nn.Module):  # 4.48.10 Eq.498
    def __init__(self,n=3):
        super().__init__(); self.a=nn.Parameter(torch.ones(n)); self.sigma=nn.Parameter(torch.ones(n)); self.mu=nn.Parameter(torch.linspace(-1,1,n))
    def forward(self,z):
        y=0
        for a,s,m in zip(self.a,self.sigma,self.mu):
            y=y+torch.sqrt(a/(2*math.pi*s*s))*torch.exp(-(z-m).pow(2)/(2*s*s))
        return y

class FourierSeriesActivation(nn.Module):  # 4.48.11 Eq.499
    def __init__(self,r=5):
        super().__init__(); self.a=_p(0); self.b=nn.Parameter(torch.zeros(r)); self.c=nn.Parameter(torch.zeros(r)); self.d=_p(1)
    def forward(self,z):
        y=self.a
        for j,(b,c) in enumerate(zip(self.b,self.c),start=1): y=y+b*torch.cos(j*self.d*z)+c*torch.sin(j*self.d*z)
        return y


# ---------------------------------------------------------------------------
# 4.49 - 4.57
# ---------------------------------------------------------------------------

class PAU(nn.Module):  # 4.49 Eq.500
    def __init__(self,m=5,n=4):
        super().__init__(); self.a=nn.Parameter(torch.zeros(m+1)); self.b=nn.Parameter(torch.zeros(n))
    def forward(self,z):
        num=_poly_eval(self.a,z)
        den=1+sum(b*z.pow(k) for k,b in enumerate(self.b,start=1))
        return num/den

class SafePAU(PAU):  # Eq.501
    def forward(self,z):
        num=_poly_eval(self.a,z)
        q=sum(b*z.pow(k) for k,b in enumerate(self.b,start=1))
        return num/(1+q.abs())

class RPAU(SafePAU):  # 4.50 Eq.502
    def __init__(self,m=5,n=4,noise=.1): super().__init__(m,n); self.noise=float(noise)
    def forward(self,z):
        if not self.training: return super().forward(z)
        coeff=torch.cat([self.a,self.b])
        # Eq.502: independent multiplicative-width uniform perturbation around each coefficient
        shape=(coeff.numel(),)+z.shape
        r=torch.empty(shape,device=z.device,dtype=z.dtype).uniform_(1-self.noise,1+self.noise)
        c=coeff.view((-1,)+(1,)*z.ndim)*r
        m=self.a.numel()
        num=sum(c[j]*z.pow(j) for j in range(m))
        q=sum(c[m+k-1]*z.pow(k) for k in range(1,self.b.numel()+1))
        return num/(1+q.abs())

class ERA(nn.Module):  # 4.51 Eq.503
    def __init__(self,m=5,n=4,eps=1e-6):
        super().__init__(); self.a=nn.Parameter(torch.zeros(m+1)); self.c=nn.Parameter(torch.linspace(-1,1,n)); self.d=nn.Parameter(torch.ones(n)); self.eps=eps
    def forward(self,z):
        den=torch.full_like(z,self.eps)
        prod=torch.ones_like(z)
        for c,d in zip(self.c,self.d): prod=prod*((z-c).pow(2)+d.pow(2))
        return _poly_eval(self.a,z)/(den+prod)

def _orthogonal_basis(z:Tensor,order:int,kind:str):
    kind=kind.lower()
    if kind=="chebyshev1":
        r=[torch.ones_like(z)]
        if order>=1:r.append(z)
        for n in range(1,order): r.append(2*z*r[-1]-r[-2])
    elif kind=="chebyshev2":
        r=[torch.ones_like(z)]
        if order>=1:r.append(2*z)
        for n in range(1,order): r.append(2*z*r[-1]-r[-2])
    elif kind=="laguerre":
        r=[torch.ones_like(z)]
        if order>=1:r.append(1-z)
        for n in range(1,order): r.append(((2*n+1-z)*r[-1]-n*r[-2])/(n+1))
    elif kind=="legendre":
        r=[torch.ones_like(z)]
        if order>=1:r.append(z)
        for n in range(1,order): r.append(((2*n+1)*z*r[-1]-n*r[-2])/(n+1))
    elif kind=="hermite_prob":
        r=[torch.ones_like(z)]
        if order>=1:r.append(z)
        for n in range(1,order): r.append(z*r[-1]-n*r[-2])
    elif kind=="hermite_phys":
        r=[torch.ones_like(z)]
        if order>=1:r.append(2*z)
        for n in range(1,order): r.append(2*z*r[-1]-2*n*r[-2])
    else: raise ValueError(kind)
    return r

class OPAU(nn.Module):  # 4.52 Eq.504
    def __init__(self,m=5,n=4,kind="chebyshev1",safe=False):
        super().__init__(); self.a=nn.Parameter(torch.zeros(m+1)); self.b=nn.Parameter(torch.zeros(n)); self.kind=kind; self.safe=safe
    def forward(self,z):
        R=_orthogonal_basis(z,max(len(self.a)-1,len(self.b)),self.kind)
        num=sum(a*r for a,r in zip(self.a,R))
        if self.safe: den=1+sum(b.abs()*r.abs() for b,r in zip(self.b,R[1:]))
        else: den=1+sum(b*r for b,r in zip(self.b,R[1:]))
        return num/den

class PPAF0(nn.Module):  # 4.53 Eq.506
    def __init__(self,knots:Sequence[float]=(-1,0,1)):
        super().__init__(); self.q=nn.Parameter(torch.tensor(knots,dtype=torch.float32))
    def forward(self,z):
        q=torch.sort(self.q).values; m=q.numel()+1
        y=torch.zeros_like(z)
        for k in range(1,m-1): y=torch.where((z>=q[k-1])&(z<q[k]),_const_like(z,k/m),y)
        y=torch.where(z>=q[-1],torch.ones_like(z),y)
        return y

class ReLUSpline(nn.Module):  # Eq.507
    def __init__(self,K=4):
        super().__init__(); self.a=nn.Parameter(torch.ones(K)); self.b=nn.Parameter(torch.ones(K)); self.c=nn.Parameter(torch.zeros(K))
    def forward(self,z): return sum(a*F.relu(b*z+c) for a,b,c in zip(self.a,self.b,self.c))

class TruG(nn.Module):  # 4.54 Eq.508
    def __init__(self,xi1=-1,xi2=1,sigma=1,trainable_bounds=True):
        super().__init__()
        if trainable_bounds: self.xi1,self.xi2=map(_p,(xi1,xi2))
        else: self.xi1,self.xi2=float(xi1),float(xi2)
        self.sigma=float(sigma)
    def forward(self,z):
        x1=(torch.as_tensor(self.xi1,dtype=z.dtype,device=z.device)-z)/self.sigma
        x2=(torch.as_tensor(self.xi2,dtype=z.dtype,device=z.device)-z)/self.sigma
        return z+self.sigma*(_phi(x1)-_phi(x2))/(_Phi(x1)-_Phi(x2))

class SquarePlus(nn.Module):  # 4.55.1 Eq.510
    def __init__(self,eps=.01,trainable=True): super().__init__(); self.eps=_p(eps) if trainable else float(eps)
    def forward(self,z): return .5*(z+_msrf(z,self.eps))

class StepPlus(nn.Module):  # 4.55.2 Eq.511
    def __init__(self,eps=.01,trainable=True): super().__init__(); self.eps=_p(eps) if trainable else float(eps)
    def forward(self,z): return .5*(1+z/_msrf(z,self.eps))

class BipolarPlus(nn.Module):  # Eq.512
    def __init__(self,eps=.01,trainable=True): super().__init__(); self.eps=_p(eps) if trainable else float(eps)
    def forward(self,z): return z/_msrf(z,self.eps)

class LReLUPlus(nn.Module):  # 4.55.3 Eq.513
    def __init__(self,a=.01,eps=.01,trainable_a=True): super().__init__(); self.a=_p(a) if trainable_a else float(a); self.eps=float(eps)
    def forward(self,z):
        a=torch.as_tensor(self.a,dtype=z.dtype,device=z.device); return .5*(z+a*z+_msrf((1-a)*z,self.eps))

class VReLUPlus(nn.Module):  # 4.55.4 Eq.514
    def __init__(self,eps=.01): super().__init__(); self.eps=float(eps)
    def forward(self,z): return _msrf(z,self.eps)

class SoftshrinkPlus(nn.Module):  # 4.55.5 Eq.515
    def __init__(self,a=.5,eps=.01): super().__init__(); self.a=float(a); self.eps=float(eps)
    def forward(self,z): return z+.5*(_msrf(z-self.a,self.eps)-_msrf(z+self.a,self.eps))

class PanPlus(nn.Module):  # 4.55.6 Eq.516
    def __init__(self,a=1,eps=.01): super().__init__(); self.a=float(a); self.eps=float(eps)
    def forward(self,z): return -self.a+.5*(_msrf(z-self.a,self.eps)+_msrf(z+self.a,self.eps))

class BReLUPlus(nn.Module):  # 4.55.7 Eq.517
    def __init__(self,eps=.01): super().__init__(); self.eps=float(eps)
    def forward(self,z): return .5*(1+_msrf(z,self.eps)-_msrf(z-1,self.eps))

class SReLUPlus(nn.Module):  # 4.55.8 Eq.518
    def __init__(self,a=.1,t=1,eps=.01): super().__init__(); self.a=_p(a); self.t=_p(t); self.eps=float(eps)
    def forward(self,z): return self.a*z+.5*(self.a-1)*(_msrf(z-self.t,self.eps)-_msrf(z+self.t,self.eps))

class HardTanhPlus(nn.Module):  # 4.55.9 Eq.519
    def __init__(self,eps=.01): super().__init__(); self.eps=float(eps)
    def forward(self,z): return .5*(_msrf(z+1,self.eps)-_msrf(z-1,self.eps))

class HardshrinkPlus(nn.Module):  # 4.55.10 Eq.520
    def __init__(self,a=.5,eps=.01): super().__init__(); self.a=float(a); self.eps=float(eps)
    def forward(self,z):
        return z*(1+.5*((z-self.a)/_msrf(z-self.a,self.eps)-(z+self.a)/_msrf(z+self.a,self.eps)))

class PhiPlus(nn.Module):  # 4.55.11 Eq.521
    def __init__(self,b=0,c=1,eps=.01): super().__init__(); self.b=float(b); self.c=float(c); self.eps=float(eps)
    def forward(self,z):
        u=self.c-_msrf(z-self.b,self.eps)
        return .5*(u+_msrf(u,self.eps))

class MeLUPlus(nn.Module):
    def __init__(self,b=(-1,0,1),c=(1,1,1),eps=.01):
        super().__init__(); self.base=LReLUPlus(eps=eps); self.a=nn.Parameter(torch.zeros(len(b))); self.parts=nn.ModuleList([PhiPlus(bb,cc,eps) for bb,cc in zip(b,c)])
    def forward(self,z): return self.base(z)+sum(a*p(z) for a,p in zip(self.a,self.parts))

class TSAFPlus(nn.Module):  # 4.55.12 Eq.522, literal survey expression
    def __init__(self,a=-1,b=1,c=1,eps=.01): super().__init__(); self.a,self.b,self.c=float(a),float(b),float(c); self.eps=float(eps)
    def forward(self,z):
        return (_msrf(z-self.a+self.c,self.eps)+(z-self.a).abs()+(z+self.b-self.c).abs()-(z-self.b).abs())/self.c

class ELUPlus(nn.Module):  # 4.55.13 Eq.523
    def __init__(self,a=1,eps=.01): super().__init__(); self.a=float(a); self.eps=float(eps)
    def forward(self,z):
        e=torch.expm1(z)/self.a
        return .5*(z+_msrf(z,self.eps))+.5*(e+_msrf(e,self.eps))

class SwishPlus(nn.Module):  # 4.55.14 Eq.524
    def __init__(self,eps=.01): super().__init__(); self.eps=float(eps)
    def forward(self,z): return .5*(z+z*z/_msrf(z,self.eps))

class MishPlus(nn.Module):  # 4.55.15 Eq.525
    def __init__(self,eps=.01): super().__init__(); self.bp=BipolarPlus(eps,trainable=False)
    def forward(self,z): return z*self.bp(self.bp(z))

class LogishPlus(nn.Module):  # 4.55.16 Eq.526
    def __init__(self,eps=.01): super().__init__(); self.sp=StepPlus(eps,trainable=False)
    def forward(self,z): return z*torch.log1p(self.sp(z))

class SoftsignPlus(nn.Module):  # 4.55.17 Eq.527
    def __init__(self,eps=.01): super().__init__(); self.eps=float(eps)
    def forward(self,z): return z/(1+_msrf(z,self.eps))


class SignReLUApprox(nn.Module):  # 4.55.18 Eq.528, intermediate approximation printed in survey
    def __init__(self,eps=.01): super().__init__(); self.eps=float(eps)
    def forward(self,z):
        az=z.abs()
        return .5*(z+az)+(z-az)/(2*_msrf(1-z,self.eps))

class SignReLUPlus(nn.Module):  # 4.55.18 Eq.529
    def __init__(self,eps=.01): super().__init__(); self.eps=float(eps)
    def forward(self,z):
        az=_msrf(z,self.eps); return .5*(z+az)+(z-az)/(2*_msrf(1-z,self.eps))

class VAF(nn.Module):  # 4.56.1 Eq.530
    def __init__(self,k=4,g=torch.tanh): super().__init__(); self.g=g; self.a=nn.Parameter(torch.ones(k)); self.b=nn.Parameter(torch.ones(k)); self.c=nn.Parameter(torch.zeros(k)); self.a0=_p(0)
    def forward(self,z): return sum(a*self.g(b*z+c) for a,b,c in zip(self.a,self.b,self.c))+self.a0

class FAB(nn.Module):  # 4.56.2 Eq.531 formula-level
    def __init__(self,functions:Sequence[Callable[[Tensor],Tensor]|nn.Module],scores:Optional[Sequence[float]]=None):
        super().__init__(); self.functions=list(functions); n=len(functions); self.s=nn.Parameter(torch.ones(n) if scores is None else torch.tensor(scores,dtype=torch.float32))
    def forward(self,z): return sum(s*f(z) for s,f in zip(self.s,self.functions))

class DYReLU(nn.Module):  # 4.56.3 Eq.532; hyperfunction returns (...,K,2)
    def __init__(self,hyper:nn.Module,K=2): super().__init__(); self.hyper=hyper; self.K=K
    def forward(self,z):
        p=self.hyper(z)
        a,b=p[...,0],p[...,1]
        vals=a*z.unsqueeze(-1)+b
        return vals.max(dim=-1).values

class RandomNNAdaptiveSigmoid(nn.Module):  # 4.56.4 Eq.533
    def __init__(self,a=1,b=0): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return torch.sigmoid(self.a*z+self.b)

class RandomNNAdaptiveSine(nn.Module):  # Eq.534
    def __init__(self,a=1,b=0): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return torch.sin(self.a*z+self.b)

class RandomNNAdaptiveExpDistance(nn.Module):  # Eq.535
    def __init__(self,a=1,b=0): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return torch.exp(-self.a*torch.abs(z-self.b))

class RandomNNAdaptiveStep(nn.Module):  # Eq.536
    def __init__(self,a=1,b=0): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return (self.a*z+self.b<=0).to(z.dtype)

class RandomNNAdaptiveMultiquadric(nn.Module):  # Eq.537
    def __init__(self,a=0,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return torch.sqrt((z-self.a).pow(2)+self.b.pow(2))

class KAF(nn.Module):  # 4.56.5 Eqs.538-540
    def __init__(self,D=20,delta=.2):
        super().__init__(); self.delta=float(delta); d=torch.linspace(-(D-1)/2*delta,(D-1)/2*delta,D); self.register_buffer("d",d)
        self.a=nn.Parameter(torch.zeros(D)); self.gamma=1/(6*delta*delta)
    def forward(self,z):
        return sum(a*torch.exp(-self.gamma*(z-d).pow(2)) for a,d in zip(self.a,self.d))

# 4.57 SAVE-inspired formulas from Table 3
class SAVEResonanceSine(nn.Module):
    def __init__(self,a=1,b=1,c=0): super().__init__(); self.a,self.b,self.c=map(_p,(a,b,c))
    def forward(self,z): return self.a*torch.sin(self.b*z+self.c)
class SAVEResonanceTanh(nn.Module):
    def __init__(self,a=1,b=1,c=0): super().__init__(); self.a,self.b,self.c=map(_p,(a,b,c))
    def forward(self,z): return self.a*torch.tanh(self.b*z+self.c)
class SAVENeutralShift(nn.Module):
    def __init__(self,a=0): super().__init__(); self.a=_p(a)
    def forward(self,z): return z+self.a
class SAVEReluPower(nn.Module):
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return self.a*F.relu(z).pow(self.b)
class SAVEExp(nn.Module):
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return self.a*torch.exp(self.b*z)
class SAVEMultiLevel(nn.Module):
    def __init__(self,functions=(torch.tanh,F.relu)):
        super().__init__(); self.functions=list(functions); self.a=nn.Parameter(torch.ones(len(functions)))
    def forward(self,z): return sum(a*F.relu(f(z)) for a,f in zip(self.a,self.functions))
class SAVEAsymmetric(nn.Module):
    def __init__(self,a=0): super().__init__(); self.a=_p(a)
    def forward(self,z): return F.relu(z-self.a)
class SAVEHarmonization(nn.Module):
    def __init__(self,n=2): super().__init__(); self.a=nn.Parameter(torch.ones(n))
    def forward(self,z): return self.a.sum()*F.relu(z)
class SAVEValueAddedPower(nn.Module):
    def __init__(self,a=1,b=.5): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return self.a*z.pow(1-self.b)
class SAVEValueAddedExp(nn.Module):
    def __init__(self,a=1,b=1): super().__init__(); self.a,self.b=map(_p,(a,b))
    def forward(self,z): return self.a*(1-torch.exp(-self.b*z))
class SAVEPrisonerProduct(nn.Module):
    def __init__(self,f=torch.tanh,g=torch.sigmoid): super().__init__(); self.f,self.g=f,g
    def forward(self,z): return F.relu(self.f(z)*self.g(z))
class SAVEPrisonerExpSigmoid(nn.Module):
    def __init__(self,a=1): super().__init__(); self.a=_p(a)
    def forward(self,z): return F.relu(torch.exp(z)*torch.sigmoid(self.a*z))
class SAVEShipwrecked(nn.Module):
    def __init__(self,a=1,b=0,c=1,d=1): super().__init__(); self.a,self.b,self.c,self.d=map(_p,(a,b,c,d))
    def forward(self,z): return F.relu((self.a*z+self.b)/(self.c*z+self.d))


# ---------------------------------------------------------------------------
# Coverage map and survey caveats
# ---------------------------------------------------------------------------

SECTION4_COVERAGE = {
"4.1":["TAAF"],
"4.2.1":["PReLU"],"4.2.2":["PositivePReLU"],"4.2.3":["MarginReLU"],"4.2.4":["FunnelReLU","FunnelPReLU"],
"4.2.5":["RPReLU"],"4.2.6":["SAU"],"4.2.7":["SMU"],"4.2.8":["LeLeLU"],"4.2.9":["PREU"],"4.2.10":["RTPReLU"],
"4.2.11":["ProbAct"],"4.2.12":["AOAF"],"4.2.13":["DLReLU","ExpDLReLU"],"4.2.14":["DynamicReLU"],"4.2.15":["FlexibleReLUFull","FReLU"],
"4.2.16":["ShiLU"],"4.2.17":["StarReLU"],"4.2.18":["AdaptiveHardTanh"],"4.2.19":["AReLU"],"4.2.20":["DPReLU"],
"4.2.21":["DualLine"],"4.2.22":["PiLU"],"4.2.23":["DPAF"],"4.2.24":["FPAF"],"4.2.25":["EPReLU"],"4.2.26":["PairedReLU"],
"4.2.27":["Tent"],"4.2.28":["Hat"],"4.2.29":["RMAF"],"4.2.30":["PTELU"],"4.2.31":["TaLU"],"4.2.32":["PTaLU"],
"4.2.33":["TanhLU"],"4.2.34":["TeLU"],"4.2.35":["TReLU","TReLU2"],"4.2.36":["ReLTanh"],"4.2.37":["BLU"],"4.2.38":["ReBLU"],
"4.2.39":["DELU"],"4.2.40":["SCMish"],"4.2.41":["SCSwish"],"4.2.42":["ParametricSwish"],"4.2.43":["PELUFull","PELU"],
"4.2.44":["EDELU"],"4.2.45":["MixLReLUElu","GatedPReLUPELU"],"4.2.46":["FELU"],"4.2.47":["PPlusFELU"],"4.2.48":["MPELU"],
"4.2.49":["PE2ReLU","PE2Id","PE2ReLU1"],"4.2.50":["SoftExponential"],"4.2.51":["CELU"],"4.2.52":["ErfReLU"],
"4.2.53":["PSELU"],"4.2.54":["LPSELU"],"4.2.55":["LPSELURP"],"4.2.56":["ShELU","SvELU","PShELU","PSvELU"],
"4.2.57":["TSwish"],"4.2.58":["RePSU"],"4.2.59":["PDELU"],"4.2.60":["EELU"],"4.2.61":["PFPLUS"],"4.2.62":["PVLU"],
"4.3":["ShapeAutotuningSigmoid"],"4.3.1":["GeneralizedTanh"],"4.3.2":["TrainableAmplitude"],"4.3.3":["ASSF"],"4.3.4":["SVAF"],
"4.3.5":["TanhSoft","TanhSoft1","TanhSoft2","TanhSoft3"],"4.3.6":["PSigmoid"],"4.3.7":["ParametricSigmoidFunction"],
"4.3.8":["STACTanh"],"4.3.9":["GRA"],
"4.4.1":["Swish"],"4.4.2":["AHAF"],"4.4.3":["PSSiLU"],"4.4.4":["ESwish"],"4.4.5":["ACONB"],"4.4.6":["ACONC","MetaACONC"],
"4.4.7":["PSGU"],"4.4.8":["TBSReLULearnable"],"4.4.9":["PATS"],"4.4.10":["AQuLU"],"4.4.11":["SinLU"],"4.4.12":["ErfAct"],
"4.4.13":["PSerf"],"4.4.14":["Swim"],
"4.5":["TunedSoftmax"],"4.6":["GeneralizedLehmerSoftmax"],"4.7":["GeneralizedPowerSoftmax"],"4.8":["ARBF"],"4.9":["PGELU"],
"4.10":["PFTS"],"4.11":["PFPM"],"4.12":["GEU"],"4.13":["SGT"],"4.14":["RSign"],"4.15":["PSIGRAMP"],
"4.16":["LAAF","LAAFSigmoid","LAAFTanh","LAAFReLU","LAAFLeakyReLU"],"4.16.1":["LAAFTanh"],"4.16.2":["PSTanh"],
"4.16.3":["SSinH"],"4.16.4":["SExp"],"4.16.5":["LAU"],"4.16.6":["CosLU"],"4.16.7":["AGumb"],
"4.17":["SAAAF"],"4.18":["NoisyActivation","NoisyInputActivation"],
"4.19":["FractionalAdaptiveActivation"],"4.19.1":["FractionalReLU"],"4.19.2":["FractionalSoftplus"],"4.19.3":["FractionalTanh"],"4.19.4":["FALUExact","FALU"],
"4.19.5":["FractionalLeakyReLU"],"4.19.6":["FractionalPReLU"],"4.19.7":["FractionalELU"],
"4.19.8":["FractionalSiLU1","FractionalSiLU2"],"4.19.9":["FractionalGELU1","FractionalGELU2"],
"4.20":["ScaledSoftsign"],"4.21":["ParameterizedSoftplus"],"4.22":["UAF"],"4.23":["LEAF"],"4.24":["GReLU"],"4.25":["MAF"],
"4.26":["EIS","EIS1","EIS2","EIS3"],"4.26.1":["ELUS2L"],"4.27":["GLN"],"4.28":["NAF"],"4.28.1":["ScaledLogisticSigmoid","SLS_SS","NAFExtended"],
"4.29":["APLU"],"4.30":["SPLASH"],"4.31":["MBA"],"4.32":["MeLU"],"4.32.1":["MMeLU"],"4.32.2":["GaLUComponent","GaLU"],"4.32.3":["HardSwishAdaptive"],
"4.33":["SReLU"],"4.33.1":["NActivation"],"4.33.2":["ALiSA"],"4.34":["AllReLU"],"4.35":["PLU"],"4.36":["AdaLU"],
"4.37":["TSAF"],"4.38":["RichardsCurve","ARiA","ARiA2"],"4.39":["ModifiedWeibull"],"4.40":["Sincos"],"4.41":["CSS"],
"4.42":["CatAF"],"4.43":["Expcos"],"4.44":["MTLU"],"4.45":["CPN","CPNnl"],"4.46":["LuTULinear","LuTUCosine"],"4.47":["Maxout"],
"4.48":["ABU","ShiftedABU","MinMaxScaledABU"],"4.48.1":["TCA","TCAv2"],"4.48.2":["APAF"],"4.48.3":["GABU"],
"4.48.4":["DKNNActivation"],"4.48.5":["RowdyActivation"],"4.48.6":["SLAF"],"4.48.7":["ChPAF"],"4.48.8":["LPAF"],
"4.48.9":["HPAF"],"4.48.10":["MoGU"],"4.48.11":["FourierSeriesActivation"],
"4.49":["PAU","SafePAU"],"4.50":["RPAU"],"4.51":["ERA"],"4.52":["OPAU"],"4.53":["PPAF0","ReLUSpline"],
"4.54":["TruG"],"4.55":["_msrf"],"4.55.1":["SquarePlus"],"4.55.2":["StepPlus","BipolarPlus"],"4.55.3":["LReLUPlus"],"4.55.4":["VReLUPlus"],
"4.55.5":["SoftshrinkPlus"],"4.55.6":["PanPlus"],"4.55.7":["BReLUPlus"],"4.55.8":["SReLUPlus"],"4.55.9":["HardTanhPlus"],
"4.55.10":["HardshrinkPlus"],"4.55.11":["PhiPlus","MeLUPlus"],"4.55.12":["TSAFPlus"],"4.55.13":["ELUPlus"],
"4.55.14":["SwishPlus"],"4.55.15":["MishPlus"],"4.55.16":["LogishPlus"],"4.55.17":["SoftsignPlus"],"4.55.18":["SignReLUApprox","SignReLUPlus"],
"4.56.1":["VAF"],"4.56.2":["FAB"],"4.56.3":["DYReLU"],
"4.56.4":["RandomNNAdaptiveSigmoid","RandomNNAdaptiveSine","RandomNNAdaptiveExpDistance","RandomNNAdaptiveStep","RandomNNAdaptiveMultiquadric"],
"4.56.5":["KAF"],
"4.57":["SAVEResonanceSine","SAVEResonanceTanh","SAVENeutralShift","SAVEReluPower","SAVEExp","SAVEMultiLevel","SAVEAsymmetric",
        "SAVEHarmonization","SAVEValueAddedPower","SAVEValueAddedExp","SAVEPrisonerProduct","SAVEPrisonerExpSigmoid","SAVEShipwrecked"],
}


# Independent equation-to-subsection audit extracted from the paper text.
EQUATION_SECTION = {
    255: '4.1',
    256: '4.1',
    257: '4.2.1',
    258: '4.2.2',
    259: '4.2.3',
    260: '4.2.4',
    261: '4.2.4',
    262: '4.2.5',
    263: '4.2.6',
    264: '4.2.6',
    265: '4.2.7',
    266: '4.2.8',
    267: '4.2.9',
    268: '4.2.10',
    269: '4.2.11',
    270: '4.2.11',
    271: '4.2.12',
    272: '4.2.13',
    273: '4.2.13',
    274: '4.2.14',
    275: '4.2.15',
    276: '4.2.15',
    277: '4.2.16',
    278: '4.2.17',
    279: '4.2.18',
    280: '4.2.19',
    281: '4.2.20',
    282: '4.2.21',
    283: '4.2.22',
    284: '4.2.23',
    285: '4.2.24',
    286: '4.2.25',
    287: '4.2.26',
    288: '4.2.27',
    289: '4.2.28',
    290: '4.2.29',
    291: '4.2.30',
    292: '4.2.31',
    293: '4.2.32',
    294: '4.2.33',
    295: '4.2.34',
    296: '4.2.35',
    297: '4.2.35',
    298: '4.2.36',
    299: '4.2.36',
    300: '4.2.37',
    301: '4.2.38',
    302: '4.2.39',
    303: '4.2.40',
    304: '4.2.41',
    305: '4.2.42',
    306: '4.2.43',
    307: '4.2.43',
    308: '4.2.44',
    309: '4.2.44',
    310: '4.2.45',
    311: '4.2.45',
    312: '4.2.46',
    313: '4.2.47',
    314: '4.2.48',
    315: '4.2.49',
    316: '4.2.49',
    317: '4.2.49',
    318: '4.2.50',
    319: '4.2.51',
    320: '4.2.52',
    321: '4.2.53',
    322: '4.2.54',
    323: '4.2.55',
    324: '4.2.56',
    325: '4.2.56',
    326: '4.2.56',
    327: '4.2.56',
    328: '4.2.57',
    329: '4.2.58',
    330: '4.2.58',
    331: '4.2.58',
    332: '4.2.59',
    333: '4.2.60',
    334: '4.2.60',
    335: '4.2.60',
    336: '4.2.60',
    337: '4.2.61',
    338: '4.2.61',
    339: '4.2.62',
    340: '4.3',
    341: '4.3.1',
    342: '4.3.2',
    343: '4.3.3',
    344: '4.3.4',
    345: '4.3.5',
    346: '4.3.5',
    347: '4.3.5',
    348: '4.3.5',
    349: '4.3.6',
    350: '4.3.7',
    351: '4.3.8',
    352: '4.3.9',
    353: '4.4.1',
    354: '4.4.2',
    355: '4.4.3',
    356: '4.4.4',
    357: '4.4.5',
    358: '4.4.6',
    359: '4.4.7',
    360: '4.4.8',
    361: '4.4.9',
    362: '4.4.9',
    363: '4.4.10',
    364: '4.4.11',
    365: '4.4.12',
    366: '4.4.13',
    367: '4.4.14',
    368: '4.5',
    369: '4.6',
    370: '4.6',
    371: '4.6',
    372: '4.6',
    373: '4.7',
    374: '4.7',
    375: '4.7',
    376: '4.7',
    377: '4.8',
    378: '4.9',
    379: '4.10',
    380: '4.11',
    381: '4.12',
    382: '4.13',
    383: '4.14',
    384: '4.15',
    385: '4.16',
    386: '4.16',
    387: '4.16',
    388: '4.16',
    389: '4.16',
    390: '4.16',
    391: '4.16.1',
    392: '4.16.2',
    393: '4.16.3',
    394: '4.16.4',
    395: '4.16.5',
    396: '4.16.6',
    397: '4.16.7',
    398: '4.17',
    399: '4.18',
    400: '4.18',
    401: '4.18',
    402: '4.19',
    403: '4.19.1',
    404: '4.19.2',
    405: '4.19.2',
    406: '4.19.3',
    407: '4.19.3',
    408: '4.19.4',
    409: '4.19.4',
    410: '4.19.4',
    411: '4.19.4',
    412: '4.19.4',
    413: '4.19.5',
    414: '4.19.5',
    415: '4.19.6',
    416: '4.19.6',
    417: '4.19.7',
    418: '4.19.7',
    419: '4.19.8',
    420: '4.19.8',
    421: '4.19.8',
    422: '4.19.8',
    423: '4.19.8',
    424: '4.19.9',
    425: '4.19.9',
    426: '4.19.9',
    427: '4.19.9',
    428: '4.19.9',
    429: '4.20',
    430: '4.21',
    431: '4.22',
    432: '4.23',
    433: '4.24',
    434: '4.25',
    435: '4.26',
    436: '4.26',
    437: '4.26',
    438: '4.26',
    439: '4.26.1',
    440: '4.27',
    441: '4.28',
    442: '4.28.1',
    443: '4.28.1',
    444: '4.28.1',
    445: '4.29',
    446: '4.30',
    447: '4.31',
    448: '4.32.2',
    449: '4.32',
    450: '4.32.1',
    451: '4.32.2',
    452: '4.32.3',
    453: '4.33',
    454: '4.33.1',
    455: '4.33.1',
    456: '4.33.1',
    457: '4.33.2',
    458: '4.34',
    459: '4.35',
    460: '4.36',
    461: '4.37',
    462: '4.38',
    463: '4.38',
    464: '4.38',
    465: '4.39',
    466: '4.40',
    467: '4.41',
    468: '4.42',
    469: '4.43',
    470: '4.44',
    471: '4.45',
    472: '4.45',
    473: '4.45',
    474: '4.46',
    475: '4.46',
    476: '4.46',
    477: '4.47',
    478: '4.48',
    479: '4.48.1',
    480: '4.48',
    481: '4.48',
    482: '4.48',
    483: '4.48.1',
    484: '4.48.1',
    485: '4.48.2',
    486: '4.48.3',
    487: '4.48.5',
    488: '4.48.5',
    489: '4.48.5',
    490: '4.48.6',
    491: '4.48.7',
    492: '4.48.7',
    493: '4.48.8',
    494: '4.48.8',
    495: '4.48.9',
    496: '4.48.9',
    497: '4.48.9',
    498: '4.48.10',
    499: '4.48.11',
    500: '4.49',
    501: '4.49',
    502: '4.50',
    503: '4.51',
    504: '4.52',
    505: '4.52',
    506: '4.53',
    507: '4.53',
    508: '4.54',
    509: '4.55',
    510: '4.55.1',
    511: '4.55.2',
    512: '4.55.2',
    513: '4.55.3',
    514: '4.55.4',
    515: '4.55.5',
    516: '4.55.6',
    517: '4.55.7',
    518: '4.55.8',
    519: '4.55.9',
    520: '4.55.10',
    521: '4.55.11',
    522: '4.55.12',
    523: '4.55.13',
    524: '4.55.14',
    525: '4.55.15',
    526: '4.55.16',
    527: '4.55.17',
    528: '4.55.18',
    529: '4.55.18',
    530: '4.56.1',
    531: '4.56.2',
    532: '4.56.3',
    533: '4.56.4',
    534: '4.56.4',
    535: '4.56.4',
    536: '4.56.4',
    537: '4.56.4',
    538: '4.56.5',
    539: '4.56.5',
    540: '4.56.5',
}
EQUATION_COVERAGE = {eq: SECTION4_COVERAGE[section] for eq, section in EQUATION_SECTION.items()}

SURVEY_AMBIGUITIES = {
    "4.2.4 FunPReLU": "The survey prints FunReLU exactly and only says FunPReLU is defined similarly. FunnelPReLU implements the natural threshold-centered PReLU extension.",
    "4.2.28 Hat": "The PDF itself prints z in both middle branches of Eq.289. Implementation preserves that literal formula rather than silently changing it to a-z.",
    "4.2.13 exp-DLReLU": "Survey prose says c_t=exp(-b_t) but the extracted line also prints exp(MSE); implementation uses exp(-MSE), consistent with the definition and prose.",
    "4.4.9 PATS": "Survey explicitly says inference behavior was not specified. Implementation uses E[a]=(l+u)/2 in eval mode.",
    "4.17 SAAAF": "PDF text extraction is typography-damaged. Implementation follows the visible fraction z/(z/a + exp(-z/b)).",
    "4.18 Noisy activation": "Requires the chosen saturating h and its linearization u. Both are caller-supplied.",
    "4.19 fractional family": "Printed definitions contain infinite limits/sums. Finite `terms`/`h` approximations are required for executable PyTorch.",
    "4.19.5-4.19.9 negative powers": "For negative real inputs and non-integer fractional exponents the printed real-valued formulas are not generally defined; PyTorch may yield NaN, faithfully exposing the domain issue.",
    "4.37 TSAF": "The printed sum of ReLUs is implemented literally.",
    "4.45 CPN": "The survey's interval inequalities are typography-confused; implementation uses descending fixed anchors defining consecutive intervals.",
    "4.46 LuTU": "The paper's printed interval direction is reversed under ascending anchors; implementation performs ordinary interpolation between adjacent ascending anchors.",
    "4.52 OPAU": "The survey writes r_i in numerator but describes r_j. Implementation uses the index matching each coefficient.",
    "4.53 splines": "The survey describes general cubic spline interpolation without a single explicit cubic formula. It explicitly prints PPAF0 and a ReLU-spline form; those are implemented.",
    "4.54 TruG": "Implemented exactly from Eq.508 using standard-normal phi/Phi evaluated at standardized truncation points.",
    "4.55.12 TSAFPlus": "Eq.522 is implemented literally, including which absolute values are mollified and which are not.",
    "4.56 complex approaches": "NIN, MIN, WHE, NPF, and hyperactivations are architectures rather than explicit scalar formulas; explicit formulas begin at 4.56.1.",
    "4.57 SAVE": "All formulas in Table 3 are implemented; the survey provides no deeper validation of them.",
}

REGISTRY={name:obj for name,obj in globals().copy().items()
          if isinstance(obj,type) and issubclass(obj,nn.Module) and obj is not nn.Module}
