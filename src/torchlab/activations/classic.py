"""PyTorch implementations of fixed/activation functions from Section 3 of
Kunc & Kléma, *Three Decades of Activations* (2024).

Design notes
------------
* All scalar/tensor-local activations are nn.Module classes.
* Fixed parameters are constructor arguments, not nn.Parameter objects.
* Numerically stable PyTorch primitives are used where possible.
* A few activations in the survey are architecture/context dependent (e.g. SCAA,
  chaotic multi-output activations). They are included as explicit modules when
  their input contract can be expressed cleanly; see SPECIAL_CASES at bottom.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


def _sech(x: Tensor) -> Tensor:
    return 1.0 / torch.cosh(x)


# 3.1 ------------------------------------------------------------------------
class BinaryStep(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return (x >= 0).to(x.dtype)


# 3.2 Sigmoid family ----------------------------------------------------------
class Sigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x)


class Tanh(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(x)


class ScaledSigmoid(nn.Module):
    """Eq. (4): 4 sigmoid(x) - 2."""
    def forward(self, x: Tensor) -> Tensor:
        return 4.0 * torch.sigmoid(x) - 2.0


class ShiftedScaledSigmoid(nn.Module):
    def __init__(self, a: float = 0.02, b: float = 600.0):
        super().__init__(); self.a, self.b = a, b
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.a * (x - self.b))


class VariantSigmoid(nn.Module):
    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 0.0):
        super().__init__(); self.a, self.b, self.c = a, b, c
    def forward(self, x: Tensor) -> Tensor:
        return self.a * torch.sigmoid(self.b * x) - self.c


class ScaledTanh(nn.Module):
    def __init__(self, a: float = 1.7159, b: float = 2/3):
        super().__init__(); self.a, self.b = a, b
    def forward(self, x: Tensor) -> Tensor:
        return self.a * torch.tanh(self.b * x)


class BimodalSigmoid(nn.Module):
    def __init__(self, b: float = 1.0):
        super().__init__(); self.b = b
    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * (torch.sigmoid(x) + torch.sigmoid(x + self.b))


class Arctan(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return torch.atan(x)


class ArctanGR(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.atan(x) / ((1.0 + math.sqrt(2.0)) / 2.0)


class SigmoidAlgebraic(nn.Module):
    def __init__(self, a: float = 0.0): super().__init__(); self.a = a
    def forward(self, x: Tensor) -> Tensor:
        ax = x.abs()
        inner = x * (1 + self.a * ax) / (1 + ax * (1 + self.a * ax))
        return torch.sigmoid(inner)


class TripleStateSigmoid(nn.Module):
    def __init__(self, a: float = 1.0, b: float = 2.0): super().__init__(); self.a, self.b = a, b
    def forward(self, x: Tensor) -> Tensor:
        s = torch.sigmoid(x)
        return s * (s + torch.sigmoid(x - self.a) + torch.sigmoid(x - self.b))


class ImprovedLogisticSigmoid(nn.Module):
    def __init__(self, a: float = 0.1, b: float = 2.0): super().__init__(); self.a, self.b = a, b
    def forward(self, x: Tensor) -> Tensor:
        sb = torch.sigmoid(torch.as_tensor(self.b, dtype=x.dtype, device=x.device))
        return torch.where(x >= self.b, self.a*(x-self.b)+sb,
               torch.where(x <= -self.b, self.a*(x+self.b)+sb, torch.sigmoid(x)))


class SigLin(nn.Module):
    def __init__(self, a: float = 0.1): super().__init__(); self.a = a
    def forward(self, x: Tensor) -> Tensor: return torch.sigmoid(x) + self.a*x


class PenalizedTanh(nn.Module):
    def __init__(self, a: float = 4.0): super().__init__(); self.a = a
    def forward(self, x: Tensor) -> Tensor:
        y = torch.tanh(x); return torch.where(x >= 0, y, y/self.a)


class SoftRootSign(nn.Module):
    def __init__(self, a: float = 2.0, b: float = 3.0): super().__init__(); self.a, self.b = a, b
    def forward(self, x: Tensor) -> Tensor:
        return x / (x/self.a + torch.exp(-x/self.b))


class SoftClipping(nn.Module):
    def __init__(self, a: float = 1.0): super().__init__(); self.a = a
    def forward(self, x: Tensor) -> Tensor:
        # log(1+e^{ax}) - log(1+e^{a(x-1)}) over a, stably.
        return (F.softplus(self.a*x) - F.softplus(self.a*(x-1.0))) / self.a




class Softsign(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return x/(1+x.abs())


class SmoothStep(nn.Module):
    def __init__(self, a: float = 2.0): super().__init__(); self.a=a
    def forward(self, x: Tensor) -> Tensor:
        mid = -2*x.pow(3)/(self.a**3) + 3*x/(2*self.a) + 0.5
        return torch.where(x >= self.a/2, torch.ones_like(x),
               torch.where(x <= -self.a/2, torch.zeros_like(x), mid))


class Elliott(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return 0.5*x/(1+x.abs()) + 0.5


class SincSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        s = torch.sigmoid(x)
        return torch.sinc(s / math.pi)  # torch.sinc(u)=sin(pi*u)/(pi*u)


class SigmoidGumbel(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * torch.exp(-torch.exp(-x))


class NewSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        num = torch.exp(x) - torch.exp(-x)
        den = torch.sqrt(2.0*(torch.exp(2*x)+torch.exp(-2*x)))
        return num/den


class LogLog(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return torch.exp(-torch.exp(-x))


class ComplementaryLogLog(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return 1.0 - torch.exp(-torch.exp(-x))


class ModifiedComplementaryLogLog(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return 1.0 - 2.0*torch.exp(-0.7*torch.exp(-x))


class SechSig(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return (x + _sech(x))*torch.sigmoid(x)


class ParametricSechSig(nn.Module):
    def __init__(self, a: float = 1.0): super().__init__(); self.a=a
    def forward(self, x: Tensor) -> Tensor: return (x+self.a*_sech(x+self.a))*torch.sigmoid(x)


class TanhSig(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return (x+torch.tanh(x))*torch.sigmoid(x)


class ParametricTanhSig(nn.Module):
    def __init__(self, a: float=1.0): super().__init__(); self.a=a
    def forward(self, x: Tensor) -> Tensor: return (x+self.a*torch.tanh(x+self.a))*torch.sigmoid(x)


class MultiStateActivation(nn.Module):
    def __init__(self, offsets: Sequence[float], a: float = 0.0):
        super().__init__(); self.offsets=tuple(float(v) for v in offsets); self.a=float(a)
    def forward(self, x: Tensor) -> Tensor:
        y = torch.full_like(x, self.a)
        for b in self.offsets: y = y + torch.sigmoid(x-b)
        return y


class SymmetricalMSAF(nn.Module):
    def __init__(self, a: float=-2.0): super().__init__(); self.a=a
    def forward(self, x: Tensor) -> Tensor: return -1 + torch.sigmoid(x) + torch.sigmoid(x+self.a)


class Rootsig(nn.Module):
    def __init__(self, a: float=1.0): super().__init__(); self.a=a
    def forward(self, x: Tensor) -> Tensor: return self.a*x/(1+torch.sqrt(1+(self.a*x).pow(2)))


# 3.3 sigmoid-weighted linear units ------------------------------------------
class SiLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return x*torch.sigmoid(x)


class GELU(nn.Module):
    def __init__(self, approximate: str='none'): super().__init__(); self.approximate=approximate
    def forward(self, x: Tensor) -> Tensor: return F.gelu(x, approximate=self.approximate)


class SGELU(nn.Module):
    def __init__(self, a: float=1.0): super().__init__(); self.a=a
    def forward(self, x: Tensor) -> Tensor: return self.a*x*torch.erf(x/math.sqrt(2.0))


class CauchyLinearUnit(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return x*(torch.atan(x)/math.pi + 0.5)


class LaplaceLinearUnit(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        cdf = torch.where(x>=0, 1-0.5*torch.exp(-x), 0.5*torch.exp(x))
        return x*cdf


class CollapsingLinearUnit(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x / (1.0 - x*torch.exp(-(x+torch.exp(x))))


class TripleStateSwish(nn.Module):
    def __init__(self,a:float=1.0,b:float=2.0): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor:
        s=torch.sigmoid(x); return x*s*(s+torch.sigmoid(x-self.a)+torch.sigmoid(x-self.b))


class GeneralizedSwish(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.sigmoid(torch.exp(-x))


class ExponentialSwish(nn.Module):
    def forward(self,x:Tensor)->Tensor: return torch.exp(-x)*torch.sigmoid(x)


class SigmoidDerivative(nn.Module):
    def forward(self,x:Tensor)->Tensor:
        s=torch.sigmoid(x); return s*(1-s)


class Gish(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.log(2-torch.exp(-torch.exp(x)))


class Logish(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.log1p(torch.sigmoid(x))


class LogLogish(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*(1-torch.exp(-torch.exp(x)))


class ExpExpish(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.exp(-torch.exp(-x))


class SelfArctan(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.atan(x)


class ParametricLogish(nn.Module):
    def __init__(self,a:float=1.0,b:float=10.0): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor: return self.a*x*torch.log1p(torch.sigmoid(self.b*x))


class Phish(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.tanh(F.gelu(x))


class Suish(nn.Module):
    def forward(self,x:Tensor)->Tensor: return torch.maximum(x, x*torch.exp(-x.abs()))


class TSReLU(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.tanh(torch.sigmoid(x))


class TBSReLU(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.tanh(torch.tanh(x/2.0))


class LogSigmoid(nn.Module):
    def forward(self,x:Tensor)->Tensor: return F.logsigmoid(x)


class DSiLU(nn.Module):
    def forward(self,x:Tensor)->Tensor:
        s=torch.sigmoid(x); return s*(1+x*(1-s))


class DoubleSiLU(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.sigmoid(x*torch.sigmoid(x))


class ModifiedSiLU(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.sigmoid(x)+torch.exp(-x.pow(2)-1)/4


class TSiLU(nn.Module):
    def forward(self,x:Tensor)->Tensor: return torch.tanh(x*torch.sigmoid(x))


class ASiLU(nn.Module):
    def forward(self,x:Tensor)->Tensor: return torch.atan(x*torch.sigmoid(x))


class SwAT(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.sigmoid(torch.atan(x))


class RectifiedHyperbolicSecant(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*_sech(x)


class LiSHT(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.tanh(x)


class Mish(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.tanh(F.softplus(x))


class Smish(nn.Module):
    def __init__(self,a:float=1.0,b:float=1.0): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor: return self.a*x*torch.tanh(torch.log1p(torch.sigmoid(self.b*x)))


class TanhExp(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.tanh(torch.exp(x))


class Serf(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.erf(F.softplus(x))


class EANAF(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.exp(x)/(torch.exp(x)+2.0)


class SinSig(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.sin((math.pi/2)*torch.sigmoid(x))


class SiELU(nn.Module):
    def forward(self,x:Tensor)->Tensor:
        return x*torch.sigmoid(2*math.sqrt(2/math.pi)*(x+0.044715*x.pow(3)))


# 3.4 Gated units -------------------------------------------------------------
class GLU(nn.Module):
    def forward(self, z: Tensor, zp: Tensor) -> Tensor: return z*torch.sigmoid(zp)
class GTU(nn.Module):
    def forward(self, z: Tensor, zp: Tensor) -> Tensor: return torch.tanh(z)*torch.sigmoid(zp)
class ReGLU(nn.Module):
    def forward(self, z: Tensor, zp: Tensor) -> Tensor: return z*F.relu(zp)
class GEGLU(nn.Module):
    def forward(self, z: Tensor, zp: Tensor) -> Tensor: return z*F.gelu(zp)


# 3.5 ------------------------------------------------------------------------
class Softmax(nn.Module):
    def __init__(self, dim:int=-1): super().__init__(); self.dim=dim
    def forward(self,x:Tensor)->Tensor: return torch.softmax(x,dim=self.dim)

class BetaSoftmax(nn.Module):
    """Eq. (89) for a supplied positive b. The survey only says b is random in N+
    and gives no sampling distribution, so sampling is intentionally left to caller."""
    def __init__(self,b:float=1.0,dim:int=-1): super().__init__(); self.b,self.dim=b,dim
    def forward(self,x:Tensor)->Tensor: return torch.softmax(self.b*x,dim=self.dim)


# 3.6 ReLU family -------------------------------------------------------------
class ReLU(nn.Module):
    def forward(self,x:Tensor)->Tensor: return F.relu(x)
class ShiftedReLU(nn.Module):
    def forward(self,x:Tensor)->Tensor: return torch.maximum(x, torch.full_like(x,-1.0))
class LeakyReLU(nn.Module):
    def __init__(self,a:float=100.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return torch.where(x>=0,x,x/self.a)
class VeryLeakyReLU(LeakyReLU):
    def __init__(self): super().__init__(a=3.0)
class OptimizedLeakyReLU(nn.Module):
    def __init__(self,l:float=3.0,u:float=8.0): super().__init__(); self.a=(u+l)/(u-l)
    def forward(self,x:Tensor)->Tensor: return torch.where(x>=0,x,x*math.exp(-self.a))
class SlopedReLU(nn.Module):
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return self.a*F.relu(x)
class NoisyReLU(nn.Module):
    """NReLU, Eq. (99). Noise scale is sigma(z), the input standard deviation.
    `dim=None` uses the standard deviation of the whole tensor; set `dim` to
    compute it along a particular input dimension."""
    def __init__(self,dim:Optional[int]=None,unbiased:bool=False):
        super().__init__(); self.dim,self.unbiased=dim,unbiased
    def forward(self,x:Tensor)->Tensor:
        sigma=x.std(dim=self.dim,keepdim=self.dim is not None,unbiased=self.unbiased)
        a=torch.randn_like(x)*sigma
        return F.relu(x+a)
class SineReLU(nn.Module):
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return torch.where(x>=0,x,self.a*(torch.sin(x)-torch.cos(x)))
class Minsin(nn.Module):
    def forward(self,x:Tensor)->Tensor: return torch.minimum(x, torch.sin(x))
class VariationalLinearUnit(nn.Module):
    def __init__(self,a:float=0.1,b:float=1.0): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor: return F.relu(x)+self.a*torch.sin(self.b*x)
class NaturalLogReLU(nn.Module):
    def __init__(self,beta:float=1.0): super().__init__(); self.beta=beta
    def forward(self,x:Tensor)->Tensor: return torch.log1p(self.beta*F.relu(x))
class SoftplusLinearUnit(nn.Module):
    """Eq. (107); defaults are the paper's continuity/differentiability choice."""
    def __init__(self,a:float=1.0,b:float=2.0,c:float=2*math.log(2.0)):
        super().__init__(); self.a,self.b,self.c=a,b,c
    def forward(self,x:Tensor)->Tensor:
        return torch.where(x>=0,self.a*x,self.b*F.softplus(x)-self.c)
class RectifiedSoftplus(nn.Module):
    """Eq. (109)."""
    def __init__(self,a:float=1.5): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor:
        return torch.where(x>=0,self.a*x+math.log(2.0),F.softplus(x))
class ParametricRectifiedNonlinearUnit(nn.Module):
    """PReNU, Eq. (110). The parameter is fixed in Section 3."""
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor:
        return torch.where(x>=0,x-self.a*torch.log1p(x),torch.zeros_like(x))
class BoundedReLU(nn.Module):
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return torch.clamp(x,0,self.a)
class HardSigmoid(nn.Module):
    def forward(self,x:Tensor)->Tensor: return torch.clamp((x+1.0)/2.0,0,1)
class HardTanh(nn.Module):
    """Eq. (114), with conventional defaults a=-1, b=1."""
    def __init__(self,a:float=-1.0,b:float=1.0): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor: return torch.clamp(x,self.a,self.b)
class VerticalShiftedHardTanh(nn.Module):
    """SvHardTanh, Eqs. (115)-(116)."""
    def __init__(self,a:float=0.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return torch.clamp(x,-1.0,1.0)+self.a
class HorizontalShiftedHardTanh(nn.Module):
    """ShHardTanh, Eq. (117)."""
    def __init__(self,a:float=0.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return torch.clamp(x+self.a,-1.0,1.0)
ShiftedHardTanh = HorizontalShiftedHardTanh
class HardSwish(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.clamp(x+3,0,6)/6
class TruncatedRectified(nn.Module):
    """TRec, Eq. (119): thresholded one-sided linear unit."""
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return torch.where(x>self.a,x,torch.zeros_like(x))
class HardShrink(nn.Module):
    def __init__(self,lambd:float=0.5): super().__init__(); self.lambd=lambd
    def forward(self,x:Tensor)->Tensor: return F.hardshrink(x,lambd=self.lambd)
class SoftShrink(nn.Module):
    def __init__(self,lambd:float=0.5): super().__init__(); self.lambd=lambd
    def forward(self,x:Tensor)->Tensor: return F.softshrink(x,lambd=self.lambd)
class VReLU(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x.abs()
class Pan(nn.Module):
    """Eq. (124): max(|z|-a, 0)."""
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return torch.clamp(x.abs()-self.a,min=0.0)
class AbsLU(nn.Module):
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return torch.where(x>=0,x,self.a*x.abs())
class MirroredReLU(nn.Module):
    """mReLU, Eq. (126): triangular tent on [-1,1]."""
    def forward(self,x:Tensor)->Tensor: return torch.clamp(1.0-x.abs(),min=0.0)
class LSPTLU(nn.Module):
    """Eq. (127)."""
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor:
        return torch.where(x<0,0.2*x,
               torch.where(x<=self.a,x,
               torch.where(x<=2*self.a,2*self.a-x,torch.zeros_like(x))))
class SoftModulusQ(nn.Module):
    """Eq. (128). The PDF's inequality is internally inconsistent; this uses the
    continuity-consistent interpretation: cubic polynomial for |z|<=1, |z| outside."""
    def forward(self,x:Tensor)->Tensor:
        ax=x.abs(); return torch.where(ax<=1.0,x.pow(2)*(2.0-ax),ax)
class SoftModulusT(nn.Module):
    def __init__(self,a:float=0.01): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return x*torch.tanh(x/self.a)
class LiReLU(nn.Module):
    """Eq. (131)."""
    def __init__(self,a:float=0.1): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return torch.where(x>=0,(self.a+1.0)*x,self.a*x)
class CReLU(nn.Module):
    def __init__(self,dim:int=-1): super().__init__(); self.dim=dim
    def forward(self,x:Tensor)->Tensor: return torch.cat((F.relu(x),F.relu(-x)),dim=self.dim)
class NCReLU(nn.Module):
    def __init__(self,dim:int=-1): super().__init__(); self.dim=dim
    def forward(self,x:Tensor)->Tensor: return torch.cat((F.relu(x),-F.relu(-x)),dim=self.dim)
class DualReLU(nn.Module):
    """Eq. (135): two inputs, one output."""
    def forward(self,z:Tensor,zp:Tensor)->Tensor: return F.relu(z)-F.relu(zp)
class RePU(nn.Module):
    def __init__(self,p:float=2.0): super().__init__(); self.p=p
    def forward(self,x:Tensor)->Tensor: return F.relu(x).pow(self.p)
class ApproximateReLU(nn.Module):
    """AppReLU, Eq. (142): a*z^b for z>=0, else 0."""
    def __init__(self,a:float=1.0,b:float=1.0): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor: return self.a*F.relu(x).pow(self.b)
class EvenPowerLinearActivation(nn.Module):
    """EPLAF, Eq. (143)."""
    def __init__(self,d:float=2.0): super().__init__(); self.d=d
    def forward(self,x:Tensor)->Tensor:
        off=1.0-1.0/self.d
        return torch.where(x>=1.0,x-off,torch.where(x>=-1.0,x.abs().pow(self.d)/self.d,-x-off))
PowerLinearActivation = EvenPowerLinearActivation
class ELU(nn.Module):
    """Eq. (151), following the survey's 1/a parameterization."""
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor:
        return torch.where(x>=0,x,torch.expm1(x)/self.a)
class REU(nn.Module):
    def forward(self,x:Tensor)->Tensor: return torch.where(x>=0,x,x*torch.exp(x))
class SigLU(nn.Module):
    """Eq. (155); negative branch equals tanh(z)."""
    def forward(self,x:Tensor)->Tensor: return torch.where(x>=0,x,torch.tanh(x))
class SaRa(nn.Module):
    """Eq. (156), using the survey's stated intended formula."""
    def __init__(self,a:float=0.5,b:float=0.7): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor:
        return torch.where(x>=0,x,x/(1.0+self.a*torch.exp(-self.b*x)))


# 3.7 ------------------------------------------------------------------------
class Maxsig(nn.Module):
    def forward(self,x:Tensor)->Tensor: return torch.maximum(x,torch.sigmoid(x))
class TanhLinearUnit(nn.Module):
    """ThLU, Eq. (158)."""
    def forward(self,x:Tensor)->Tensor: return torch.where(x>=0,x,torch.tanh(x/2.0))
class DualELU(nn.Module):
    """Eq. (159): two-input difference of ELUs."""
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def _elu(self,x:Tensor)->Tensor: return torch.where(x>=0,x,torch.expm1(x)/self.a)
    def forward(self,z:Tensor,zp:Tensor)->Tensor: return self._elu(z)-self._elu(zp)
class DifferenceELU(nn.Module):
    """DiffELU, Eq. (160)."""
    def __init__(self,a:float=0.3,b:float=0.1): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor:
        return torch.where(x>=0,x,self.a*(x*torch.exp(x)-self.b*torch.exp(self.b*x)))
class PolyLU(nn.Module):
    """Eq. (161)."""
    def forward(self,x:Tensor)->Tensor: return torch.where(x>=0,x,1.0/(1.0-x)-1.0)
class InversePolyLU(nn.Module):
    """IpLU, Eq. (162)."""
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return torch.where(x>=0,x,1.0/(1.0+x.abs().pow(self.a)))
class PoLU(nn.Module):
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return torch.where(x>=0,x,(1-x).pow(-self.a)-1)
class PFLU(nn.Module):
    """Eq. (164)."""
    def forward(self,x:Tensor)->Tensor: return 0.5*x*(1.0+x/torch.sqrt(1.0+x.pow(2)))
class LSELU(nn.Module):
    """Eq. (173)."""
    def __init__(self,a:float=1.05078,b:float=1.6733,c:float=0.1): super().__init__(); self.a,self.b,self.c=a,b,c
    def forward(self,x:Tensor)->Tensor:
        return torch.where(x>=0,self.a*x,self.a*self.b*torch.expm1(x)+self.a*self.c*x)
class SERLU(nn.Module):
    def __init__(self,alpha:float=2.90427,lam:float=1.07862): super().__init__(); self.alpha,self.lam=alpha,lam
    def forward(self,x:Tensor)->Tensor: return self.lam*torch.where(x>=0,x,self.alpha*x*torch.exp(x))
class ELiSH(nn.Module):
    def forward(self,x:Tensor)->Tensor:
        s=torch.sigmoid(x); return torch.where(x>=0,x*s,torch.expm1(x)*s)
class HardELiSH(nn.Module):
    """Eq. (179), copied literally from the survey."""
    def forward(self,x:Tensor)->Tensor:
        hs=torch.clamp((x+1)/2,0,1)
        return torch.where(x>=0,x*hs,(1.0+torch.exp(-x))*hs)


# 3.8 square-based ------------------------------------------------------------
class SQNL(nn.Module):
    def forward(self,x:Tensor)->Tensor:
        return torch.where(x>2,torch.ones_like(x),torch.where(x>=0,x-x.pow(2)/4,
               torch.where(x>=-2,x+x.pow(2)/4,-torch.ones_like(x))))
class SQLU(nn.Module):
    def forward(self,x:Tensor)->Tensor:
        return torch.where(x>0,x,torch.where(x>=-2,x+x.pow(2)/4,-torch.ones_like(x)))
class Squish(nn.Module):
    """Eq. (185)."""
    def forward(self,x:Tensor)->Tensor:
        return torch.where(x>0,x+x.pow(2)/32.0,
               torch.where(x>=-2,x+x.pow(2)/2.0,torch.zeros_like(x)))
class SqREU(nn.Module):
    """Eq. (186)."""
    def forward(self,x:Tensor)->Tensor:
        return torch.where(x>0,x,torch.where(x>=-2,x+x.pow(2)/2.0,torch.zeros_like(x)))
class SqSoftplus(nn.Module):
    """Eq. (187): thresholds +/-1/2."""
    def forward(self,x:Tensor)->Tensor:
        return torch.where(x>0.5,x,
               torch.where(x>=-0.5,x+0.5*(x+0.5).pow(2),torch.zeros_like(x)))
class LogSQNL(nn.Module):
    def forward(self,x:Tensor)->Tensor: return 0.5*(SQNL()(x)+1.0)
class ISRLU(nn.Module):
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return torch.where(x>=0,x,x/torch.sqrt(1+self.a*x.pow(2)))
class ISRU(nn.Module):
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return x/torch.sqrt(1+self.a*x.pow(2))
class ModifiedElliott(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x/torch.sqrt(1+x.pow(2))+0.5


# 3.9 onward -----------------------------------------------------------------
class SQRT(nn.Module):
    def forward(self,x:Tensor)->Tensor: return torch.sign(x)*torch.sqrt(x.abs())
class SSAF(nn.Module):
    def __init__(self,a:float=0.5): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return torch.sign(x)*torch.sqrt(2*self.a*x.abs())
class BentIdentity(nn.Module):
    def forward(self,x:Tensor)->Tensor: return (torch.sqrt(x.pow(2)+1)-1)/2+x
class Mishra(nn.Module):
    def forward(self,x:Tensor)->Tensor:
        s=x/(1+x.abs()); return 0.5*s.pow(2)+0.5*s
class SBAF(nn.Module):
    def __init__(self,k:float=0.98,alpha:float=0.5): super().__init__(); self.k,self.alpha=k,alpha
    def forward(self,x:Tensor)->Tensor:
        return 1/(1+self.k*x.pow(self.alpha)*(1-x).pow(1-self.alpha))
class Symexp(nn.Module):
    def forward(self,x:Tensor)->Tensor: return torch.sign(x)*torch.expm1(x.abs())
class SPOCU(nn.Module):
    def __init__(self,a:float=1.0,b:float=0.5,c:float=1.0,d:float=1.0): super().__init__(); self.a,self.b,self.c,self.d=a,b,c,d
    def _r(self,x:Tensor)->Tensor: return x.pow(3)*(x.pow(5)-2*x.pow(4)+2)
    def _h(self,x:Tensor)->Tensor:
        rd=self._r(torch.as_tensor(self.d,dtype=x.dtype,device=x.device))
        return torch.where(x>=self.d,rd,torch.where(x>=0,self._r(x),x))
    def forward(self,x:Tensor)->Tensor:
        xb=torch.as_tensor(self.b,dtype=x.dtype,device=x.device)
        return self.a*self._h(x/self.c+self.b)-self.a*self._h(xb)
class PUAF(nn.Module):
    def __init__(self,a:float=1.0,b:float=5.0,c:float=10.0): super().__init__(); self.a,self.b,self.c=a,b,c
    def forward(self,x:Tensor)->Tensor:
        # Eq. (204); c=0 is handled as the ReLU limiting case.
        if self.c == 0: return F.relu(x).pow(self.a)
        middle=x.pow(self.a)*(self.c+x).pow(self.b)/((self.c+x).pow(self.b)+(self.c-x).pow(self.b))
        return torch.where(x>self.c,x.pow(self.a),torch.where(x<-self.c,torch.zeros_like(x),middle))
class BiFiring(nn.Module):
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor:
        mid=x.pow(2)/(2*self.a)
        return torch.where(x>self.a,x-self.a/2,torch.where(x<-self.a,-x-self.a/2,mid))
class BoundedBiFiring(nn.Module):
    def __init__(self,a:float=1.0,b:float=3.0): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor:
        y=BiFiring(self.a)(x); return torch.clamp(y,max=self.b)
class PiecewiseRBF(nn.Module):
    def __init__(self,a:float=3.0,b:float=1.0): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor:
        center=torch.exp(-x.pow(2)/(self.b**2))
        pos=torch.exp(-(x-2*self.a).pow(2)/(self.b**2))
        neg=torch.exp(-(x+2*self.a).pow(2)/(self.b**2))
        return torch.where(x>=self.a,pos,torch.where(x<=-self.a,neg,center))
class Softplus(nn.Module):
    def __init__(self,beta:float=1.0,threshold:float=20.0): super().__init__(); self.beta,self.threshold=beta,threshold
    def forward(self,x:Tensor)->Tensor: return F.softplus(x,beta=self.beta,threshold=self.threshold)
class ParametricSoftplus(nn.Module):
    """Eq. (206): a * (softplus(x) - b)."""
    def __init__(self,a:float=1.5,b:float=math.log(2.0)): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor: return self.a*(F.softplus(x)-self.b)
class SoftPlusPlus(nn.Module):
    """Eq. (207): log(1+exp(a*x)) + x/b - log(2)."""
    def __init__(self,a:float=1.0,b:float=2.0): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor: return F.softplus(self.a*x)+x/self.b-math.log(2.0)
class RandSoftplus(nn.Module):
    """RSP, Eq. (208), for a supplied layer noise coefficient alpha.
    The survey delegates the procedure that determines alpha from noise level to [338]."""
    def __init__(self,alpha:float=0.5): super().__init__(); self.alpha=alpha
    def forward(self,x:Tensor)->Tensor:
        return (1.0-self.alpha)*F.relu(x)+self.alpha*F.softplus(x)
class ArandaOrdaz(nn.Module):
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return 1-(1+self.a*torch.exp(x)).pow(-1/self.a)
class CombHSine(nn.Module):
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return torch.sinh(self.a*x)+torch.asinh(self.a*x)
class ModifiedArcsinh(nn.Module):
    def forward(self,x:Tensor)->Tensor: return torch.asinh(x)*torch.sqrt(x.abs())/12.0
class HyperSinh(nn.Module):
    def forward(self,x:Tensor)->Tensor: return torch.where(x>0,torch.sinh(x)/3.0,x.pow(3)/4.0)
class Sine(nn.Module):
    def forward(self,x:Tensor)->Tensor: return torch.sin(math.pi*x)
class ScaledShiftedSine(nn.Module):
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return 0.5*torch.sin(self.a*x)+0.3
class Cosine(nn.Module):
    def forward(self,x:Tensor)->Tensor: return 1.0-torch.cos(x)
class Cosid(nn.Module):
    def forward(self,x:Tensor)->Tensor: return torch.cos(x)-x
class Sinp(nn.Module):
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return torch.sin(x)-self.a*x
class GrowingCosineUnit(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.cos(x)
class AmplifyingSineUnit(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x*torch.sin(x)
class Sinc(nn.Module):
    def forward(self,x:Tensor)->Tensor: return torch.sinc(x)
class ShiftedSineUnit(nn.Module):
    def forward(self,x:Tensor)->Tensor: return math.pi*torch.sinc(x-math.pi)
class DecayingSineUnit(nn.Module):
    def forward(self,x:Tensor)->Tensor: return (math.pi/2.0)*(torch.sinc(x-math.pi)-torch.sinc(x+math.pi))
class Polyexp(nn.Module):
    def __init__(self,a:float=1.0,b:float=1.0,c:float=1.0,d:float=1.0): super().__init__(); self.a,self.b,self.c,self.d=a,b,c,d
    def forward(self,x:Tensor)->Tensor:
        return (self.a*x.pow(2)+self.b*x+self.c)*torch.exp(-self.d*x.pow(2))
class Exponential(nn.Module):
    def forward(self,x:Tensor)->Tensor: return torch.exp(-x)
class ETanh(nn.Module):
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return self.a*torch.exp(x)*torch.tanh(x)
class EvolvedTanhReLU(nn.Module):
    """Eq. (232): recurrent variant from the survey."""
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return self.a*(torch.tanh(x.pow(2))+F.relu(x))
class Wave(nn.Module):
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return (1.0-x.pow(2))*torch.exp(-self.a*x.pow(2))
class NonMonotonicCubicUnit(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x-x.pow(3)
class Triple(nn.Module):
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return self.a*x.pow(3)
class ShiftedQuadraticUnit(nn.Module):
    def forward(self,x:Tensor)->Tensor: return x.pow(2)+x



# 3.46 k-WTA -----------------------------------------------------------------
class KWinnersTakeAll(nn.Module):
    def __init__(self,k:int=1,dim:int=-1): super().__init__(); self.k,self.dim=k,dim
    def forward(self,x:Tensor)->Tensor:
        if self.k <= 0: return torch.zeros_like(x)
        if self.k >= x.shape[self.dim]: return x
        _, idx = torch.topk(x,self.k,dim=self.dim)
        mask = torch.zeros_like(x,dtype=torch.bool).scatter_(self.dim,idx,True)
        return torch.where(mask,x,torch.zeros_like(x))


# 3.48 chaotic ---------------------------------------------------------------
class HybridChaoticActivation(nn.Module):
    """Returns the j-th recursively generated chaotic output for each input."""
    def __init__(self, outputs:int, r:float=4.0): super().__init__(); self.outputs,self.r=outputs,r
    def forward(self,x:Tensor)->Tensor:
        c=torch.sigmoid(x); ys=[]
        for _ in range(self.outputs):
            c=self.r*c*(1-c); ys.append(c)
        return torch.stack(ys,dim=-1)

class FusionChaoticActivation(nn.Module):
    def __init__(self,r:float=4.0,a:float=0.0,b:float=0.0,c:float=1.0,d:float=0.0,output_unit:bool=False):
        super().__init__(); self.r,self.a,self.b,self.c,self.d,self.output_unit=r,a,b,c,d,output_unit
    def forward(self,x:Tensor)->Tensor:
        y=self.r*x*(1-x)+x+self.a-(self.b/(2*math.pi))*torch.sin(2*math.pi*x)
        if self.output_unit: y=y+torch.exp(-self.c*x.pow(2))+self.d
        return y

class CascadeChaoticActivation(nn.Module):
    def __init__(self,a:float=1.0,b:float=1.0): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor: return self.a*torch.sin(math.pi*self.b*torch.sin(math.pi*x))



# -----------------------------------------------------------------------------
# Paper-audit corrections and complete Section 3 coverage.
# These definitions follow the survey's printed formulas. Where the survey itself
# is internally inconsistent/typographically malformed, the class docstring says
# exactly what interpretation is used.
# -----------------------------------------------------------------------------

class Probit(nn.Module):
    """§3.2 (unnumbered): standard normal CDF Φ(z)."""
    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class ScaledTanhHalf(nn.Module):
    """§3.2 (mentioned variant): tanh(z/2)."""
    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(x / 2.0)

class Hexpo(nn.Module):
    """§3.2.12, Eq. (19), literal survey formula."""
    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0, d: float = 1.0):
        super().__init__(); self.a,self.b,self.c,self.d=a,b,c,d
    def forward(self, x: Tensor) -> Tensor:
        pos = -self.a * (torch.exp(-x / self.b) - 1.0)
        neg = self.c * (torch.exp(-x / self.d) - 1.0)
        return torch.where(x >= 0, pos, neg)

class Root2Sigmoid(nn.Module):
    """§3.2.19, Eq. (26), as reconstructed by the survey authors."""
    def forward(self, x: Tensor) -> Tensor:
        ln2 = math.log(2.0)
        root_2z = torch.exp(0.5 * ln2 * x)          # sqrt(2^z)
        root_2mz = torch.exp(-0.5 * ln2 * x)       # sqrt(2^-z)
        inner = 2.0 * (torch.exp(ln2 * x) + torch.exp(-ln2 * x))
        return (root_2z - root_2mz) / (2.0 * math.sqrt(2.0) * torch.sqrt(inner))

class UnnamedSigmoid37(nn.Module):
    """§3.2.25, Eq. (37), literal formula (undefined at z=±a)."""
    def __init__(self, a: float = 1.0): super().__init__(); self.a=a
    def forward(self, x: Tensor) -> Tensor:
        return x * (torch.sign(x) * x - self.a) / (x.pow(2) - self.a**2)

class UnnamedSigmoid38(nn.Module):
    """§3.2.25, Eq. (38)."""
    def __init__(self, a: float = 1.0): super().__init__(); self.a=a
    def forward(self, x: Tensor) -> Tensor:
        ax = self.a * x
        return ax / (1.0 + ax.abs())

class UnnamedSigmoid39(nn.Module):
    """§3.2.25, Eq. (39)."""
    def __init__(self, a: float = 1.0): super().__init__(); self.a=a
    def forward(self, x: Tensor) -> Tensor:
        ax = self.a * x
        return ax / torch.sqrt(1.0 + ax.pow(2))

class BipolarSigmoid(nn.Module):
    """§3.2.26, Eq. (41): σ₂(z)=2σ(z)-1."""
    def forward(self, x: Tensor) -> Tensor:
        return 2.0 * torch.sigmoid(x) - 1.0

class _PiecewiseCombo(nn.Module):
    def __init__(self, a: float = 1.0): super().__init__(); self.a=a
    def _g(self, x: Tensor) -> Tensor: raise NotImplementedError
    def _h(self, x: Tensor) -> Tensor: raise NotImplementedError
    def forward(self, x: Tensor) -> Tensor:
        return torch.where(x >= 0, self._g(x), self._h(x))

class BipolarSigmoidTanh(_PiecewiseCombo):
    """§3.2.26, Eq. (40), pair {σ₂(z), tanh(z)}."""
    def _g(self,x): return 2.0*torch.sigmoid(x)-1.0
    def _h(self,x): return torch.tanh(x)

class BipolarSigmoidZero(_PiecewiseCombo):
    """§3.2.26, pair {σ₂(z), 0}."""
    def _g(self,x): return 2.0*torch.sigmoid(x)-1.0
    def _h(self,x): return torch.zeros_like(x)

class TanhZero(_PiecewiseCombo):
    """§3.2.26, pair {tanh(z), 0}."""
    def _g(self,x): return torch.tanh(x)
    def _h(self,x): return torch.zeros_like(x)

class BipolarSigmoidLinear(_PiecewiseCombo):
    """§3.2.26, pair {σ₂(z), a z}."""
    def _g(self,x): return 2.0*torch.sigmoid(x)-1.0
    def _h(self,x): return self.a*x

class TanhLinearCombination(_PiecewiseCombo):
    """§3.2.26, pair {tanh(z), a z}."""
    def _g(self,x): return torch.tanh(x)
    def _h(self,x): return self.a*x

class WeightedSigmoidGate(nn.Module):
    """§3.3, Eq. (44): WiG; raw input x_i gated by σ(z)."""
    def forward(self, x_i: Tensor, z: Tensor) -> Tensor:
        return x_i * torch.sigmoid(z)

class SwiGLU(nn.Module):
    """§3.4.4, Eq. (87). The survey states swish has its own trainable β."""
    def __init__(self, beta: float = 1.0, trainable: bool = True):
        super().__init__()
        t = torch.tensor(float(beta))
        if trainable: self.beta = nn.Parameter(t)
        else: self.register_buffer('beta', t)
    def forward(self, z: Tensor, zp: Tensor) -> Tensor:
        return z * (zp * torch.sigmoid(self.beta * zp))

class RandomizedLeakyReLU(nn.Module):
    """§3.6.3, Eqs. (95)-(96). One a_i per neuron/feature in training."""
    def __init__(self,l:float=3.0,u:float=8.0,feature_dim:int=-1):
        super().__init__(); self.l,self.u,self.feature_dim=l,u,feature_dim
    def forward(self,x:Tensor)->Tensor:
        if self.training:
            dim=self.feature_dim % x.ndim
            shape=[1]*x.ndim; shape[dim]=x.shape[dim]
            a=torch.empty(shape,device=x.device,dtype=x.dtype).uniform_(self.l,self.u)
        else:
            a=torch.as_tensor((self.l+self.u)/2,device=x.device,dtype=x.dtype)
        return torch.where(x>=0,x,x/a)

class SoftsignRandomizedLeakyReLU(nn.Module):
    """§3.6.4, Eq. (97)."""
    def __init__(self,l:float=1/8,u:float=1/3,feature_dim:int=-1):
        super().__init__(); self.l,self.u,self.feature_dim=l,u,feature_dim
    def forward(self,x:Tensor)->Tensor:
        base=1.0/(1.0+x).pow(2)
        if self.training:
            dim=self.feature_dim % x.ndim; shape=[1]*x.ndim; shape[dim]=x.shape[dim]
            a=torch.empty(shape,device=x.device,dtype=x.dtype).uniform_(self.l,self.u)
        else:
            a=torch.as_tensor((self.l+self.u)/2,device=x.device,dtype=x.dtype)
        return torch.where(x>=0,base+x,base+a*x)

class SpatialContextAwareActivation(nn.Module):
    """§3.6.10, Eq. (104): max(X, f_DW(X)). Pass the depth-wise convolution/module explicitly."""
    def __init__(self, depthwise: nn.Module): super().__init__(); self.depthwise=depthwise
    def forward(self,x:Tensor)->Tensor: return torch.maximum(x,self.depthwise(x))

class RandomlyTranslationalReLU(nn.Module):
    """§3.6.11, Eq. (105), a_i~N(0, variance); a_i=0 at inference."""
    def __init__(self,variance:float=0.75**2,feature_dim:int=-1):
        super().__init__(); self.variance,self.feature_dim=variance,feature_dim
    def forward(self,x:Tensor)->Tensor:
        if not self.training: return F.relu(x)
        dim=self.feature_dim % x.ndim; shape=[1]*x.ndim; shape[dim]=x.shape[dim]
        a=torch.randn(shape,device=x.device,dtype=x.dtype)*math.sqrt(self.variance)
        return F.relu(x+a)

class BoundedLeakyReLU(nn.Module):
    """§3.6.24, Eq. (122). c=(1-a)b ensures continuity."""
    def __init__(self,a:float=0.01,b:float=1.0): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor:
        c=(1.0-self.a)*self.b
        return torch.where(x<=0,self.a*x,torch.where(x<self.b,x,self.a*x+c))

class SignReLU(nn.Module):
    """§3.6.32, Eq. (130)."""
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor:
        return torch.where(x>=0,x,self.a*x/(x.abs()+1.0))

class BipolarActivationFunction(nn.Module):
    """§3.6.35, Eq. (134), alternating neuron parity; g defaults to ReLU."""
    def __init__(self,g:Optional[nn.Module]=None,dim:int=-1):
        super().__init__(); self.g=g if g is not None else ReLU(); self.dim=dim
    def forward(self,x:Tensor)->Tensor:
        dim=self.dim%x.ndim
        idx=torch.arange(x.shape[dim],device=x.device)
        shape=[1]*x.ndim; shape[dim]=x.shape[dim]
        even=(idx%2==0).reshape(shape)
        return torch.where(even,self.g(x),-self.g(-x))

class OrthogonalPermutationLinearUnit(nn.Module):
    """§3.6.37, Eqs. (136)-(137). Pairwise (max,min), preserving tensor shape."""
    def __init__(self,dim:int=-1): super().__init__(); self.dim=dim
    def forward(self,x:Tensor)->Tensor:
        dim=self.dim%x.ndim
        if x.shape[dim]%2: raise ValueError('OPLU requires an even size along dim')
        y=x.movedim(dim,-1); shp=y.shape
        pairs=y.reshape(*shp[:-1],shp[-1]//2,2)
        hi=pairs.max(dim=-1).values; lo=pairs.min(dim=-1).values
        out=torch.stack((hi,lo),dim=-1).reshape(shp)
        return out.movedim(-1,dim)

class ElasticReLU(nn.Module):
    """§3.6.38, Eq. (138). k_i~U(1-α,1+α) per neuron during training; k_i=1 at test."""
    def __init__(self,alpha:float=0.1,feature_dim:int=-1): super().__init__(); self.alpha,self.feature_dim=alpha,feature_dim
    def forward(self,x:Tensor)->Tensor:
        if self.training:
            dim=self.feature_dim%x.ndim; shape=[1]*x.ndim; shape[dim]=x.shape[dim]
            k=torch.empty(shape,device=x.device,dtype=x.dtype).uniform_(1-self.alpha,1+self.alpha)
        else: k=torch.as_tensor(1.0,device=x.device,dtype=x.dtype)
        return torch.where(x>=0,k*x,torch.zeros_like(x))

class AlternatingRePU(nn.Module):
    """§3.6.39, Eqs. (140)-(141). Pass epoch parity; inference returns their mean."""
    def __init__(self,b:float=1.1): super().__init__(); self.b=b
    def forward(self,x:Tensor,epoch:Optional[int]=None)->Tensor:
        y1=F.relu(x).pow(self.b); y2=F.relu(x).pow(1.0/self.b)
        if not self.training or epoch is None: return 0.5*(y1+y2)
        return y1 if epoch%2==1 else y2

class OddPowerLinearActivation(nn.Module):
    """§3.6.41, Eq. (144), literal survey formula."""
    def __init__(self,d:float=3.0): super().__init__(); self.d=d
    def forward(self,x:Tensor)->Tensor:
        off=1.0-1.0/self.d
        posmid=x.abs().pow(self.d)/self.d
        negmid=-x.abs().pow(self.d)/self.d
        return torch.where(x>=1.0,x-off,
               torch.where(x>=0.0,posmid,
               torch.where(x>=-1.0,negmid,-x-off)))

class AverageBiasedReLU(nn.Module):
    """§3.6.42, Eq. (145). a_i is the mean input activation map for each neuron/filter."""
    def __init__(self,reduce_dims:Optional[Sequence[int]]=None,keepdim:bool=True):
        super().__init__(); self.reduce_dims=None if reduce_dims is None else tuple(reduce_dims); self.keepdim=keepdim
    def forward(self,x:Tensor)->Tensor:
        dims=self.reduce_dims
        if dims is None:
            # Common N,C,... tensor convention: average batch and spatial dims, preserve channel.
            dims=tuple(i for i in range(x.ndim) if i!=1) if x.ndim>=2 else (0,)
        a=x.mean(dim=dims,keepdim=True)
        return F.relu(x-a)

class DelayReLU(nn.Module):
    """§3.6.43, Eq. (146)."""
    def __init__(self,a:float=0.08): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return F.relu(x-self.a)

class DisplacedReLU(nn.Module):
    """§3.6.44, Eq. (147): max(z,-a)."""
    def __init__(self,a:float=1.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return torch.maximum(x,torch.full_like(x,-self.a))

class ModifiedLeakyReLU(nn.Module):
    """§3.6.45, Eq. (148)."""
    def __init__(self,a:float=0.1): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor: return torch.where(x+self.a>0,x,-self.a*x)

class FlattedTSwish(nn.Module):
    """§3.6.46, Eq. (149)."""
    def __init__(self,T:float=-0.20): super().__init__(); self.T=T
    def forward(self,x:Tensor)->Tensor: return F.relu(x)*torch.sigmoid(x)+self.T

class OptimalActivationFunction(nn.Module):
    """§3.6.47, Eq. (150)."""
    def forward(self,x:Tensor)->Tensor: return F.relu(x)+x*torch.sigmoid(x)

class ApicalDendriteActivation(nn.Module):
    """§3.6.50, Eq. (153)."""
    def __init__(self,a:float=1.0,b:float=0.0): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor: return torch.where(x>=0,torch.exp(-self.a*x+self.b),torch.zeros_like(x))

class LeakyApicalDendriteActivation(nn.Module):
    """§3.6.51, Eq. (154)."""
    def __init__(self,a:float=1.0,b:float=0.0,c:float=0.01): super().__init__(); self.a,self.b,self.c=a,b,c
    def forward(self,x:Tensor)->Tensor: return torch.where(x>=0,torch.exp(-self.a*x+self.b),self.c*x)

class FasterPFLU(nn.Module):
    """§3.7.8, Eq. (165)."""
    def forward(self,x:Tensor)->Tensor:
        return torch.where(x>=0,x,x+x.pow(2)/torch.sqrt(1.0+x.pow(2)))

class EACU(nn.Module):
    """§3.7.9, Eqs. (166)-(168). The survey explicitly makes a_i adaptive."""
    def __init__(self,a_init:float=1.0,feature_dim:int=-1,trainable:bool=True):
        super().__init__(); self.feature_dim=feature_dim
        t=torch.tensor(float(a_init))
        if trainable: self.a=nn.Parameter(t)
        else: self.register_buffer('a',t)
    def forward(self,x:Tensor)->Tensor:
        # b_i: s_i if 0.5<s_i<1.5, otherwise 1; s_i~N(0,0.01).
        if self.training:
            dim=self.feature_dim%x.ndim; shape=[1]*x.ndim; shape[dim]=x.shape[dim]
            s=torch.randn(shape,device=x.device,dtype=x.dtype)*math.sqrt(0.01)
            b=torch.where((s>0.5)&(s<1.5),s,torch.ones_like(s))
        else: b=torch.as_tensor(1.0,device=x.device,dtype=x.dtype)
        neg=self.a*x*torch.tanh(F.softplus(self.a*x))
        return torch.where(x>=0,b*x,neg)

class LipschitzReLU(nn.Module):
    """§3.7.10, Eqs. (169)-(171). φ and μ are arbitrary R→R functions."""
    def __init__(self,phi:Optional[nn.Module]=None,mu:Optional[nn.Module]=None):
        super().__init__(); self.phi=phi if phi is not None else nn.Identity(); self.mu=mu if mu is not None else nn.Identity()
    def forward(self,x:Tensor)->Tensor:
        p=torch.maximum(self.phi(x),torch.zeros_like(x)); n=torch.minimum(self.mu(x),torch.zeros_like(x))
        return torch.where(x>0,p,n)

class SELU(nn.Module):
    """§3.7.11, Eq. (172)."""
    def __init__(self,a:float=1.05078,b:float=1.6733): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor: return torch.where(x>=0,self.a*x,self.a*self.b*torch.expm1(x))

class ScaledScaledELU(nn.Module):
    """§3.7.14, Eq. (175), sSELU."""
    def __init__(self,a:float=1.05078,b:float=1.6733,c:float=1.0): super().__init__(); self.a,self.b,self.c=a,b,c
    def forward(self,x:Tensor)->Tensor: return torch.where(x>=0,self.a*x,self.a*self.b*torch.expm1(self.c*x))

class RSigELU(nn.Module):
    """§3.7.15, Eq. (176). The printed middle inequality is impossible; interpreted as 0≤z≤1."""
    def __init__(self,a:float=0.5): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor:
        pos=x*torch.sigmoid(x).pow(self.a)+x
        return torch.where(x>1,pos,torch.where(x>=0,x,self.a*torch.expm1(x)))

class HardSReLUE(nn.Module):
    """§3.7.16, Eq. (177)."""
    def __init__(self,a:float=0.5): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor:
        hs=torch.clamp((x+1.0)/2.0,0.0,1.0)
        return torch.where(x>=0,self.a*x*hs+x,self.a*torch.expm1(x))

class RSigELUD(nn.Module):
    """§3.7.19, Eq. (180)."""
    def __init__(self,a:float=0.5,b:float=0.5): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor:
        pos=x*torch.sigmoid(x).pow(self.a)+x
        return torch.where(x>1,pos,torch.where(x>=0,x,self.b*torch.expm1(x)))

class LSReLU(nn.Module):
    """§3.7.20, Eq. (181)."""
    def __init__(self,a:float=1.0,b:float=1.0): super().__init__(); self.a,self.b=a,b
    def forward(self,x:Tensor)->Tensor:
        tail=torch.log1p(self.a*x)+abs(math.log1p(self.a*self.b)-self.b)
        return torch.where(x<=0,x/(1.0+x.abs()),torch.where(x<=self.b,x,tail))

class SquareSoftmax(nn.Module):
    """§3.8.7, Eq. (189), c=4 in the paper."""
    def __init__(self,c:float=4.0,dim:int=-1): super().__init__(); self.c,self.dim=c,dim
    def forward(self,x:Tensor)->Tensor:
        q=(x+self.c).pow(2); return q/q.sum(dim=self.dim,keepdim=True)

class LinearQuadratic(nn.Module):
    """§3.8.8, Eq. (190), literal polynomial branches."""
    def __init__(self,a:float=0.1): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor:
        hi=2.0-2.0*self.a; lo=-2.0+2.0*self.a
        poly=1.0-2.0*x+x.pow(2)
        mid=0.25*x*(4.0-x.abs())
        return torch.where(x>=hi,self.a*x+poly,torch.where(x<=lo,self.a*x-poly,mid))

class LogarithmicActivation(nn.Module):
    """§3.13, Eq. (199), literal survey formula. Note f(0)=ln(0)+1=-∞."""
    def forward(self,x:Tensor)->Tensor:
        pos=torch.log(x)+1.0
        neg=-torch.log(-x)+1.0
        return torch.where(x>=0,pos,neg)

class PiecewiseMexicanHat(nn.Module):
    """§3.23, Eq. (212)."""
    def __init__(self,a:float=4.0): super().__init__(); self.a=a
    def forward(self,x:Tensor)->Tensor:
        coef=(2.0/math.sqrt(3.0))*(math.pi**(-0.25))
        un=x+self.a; up=x-self.a
        neg=coef*(1.0-un.pow(2))*torch.exp(-un.pow(2)/2.0)
        pos=coef*(1.0-up.pow(2))*torch.exp(-up.pow(2)/2.0)
        return torch.where(x<0,neg,pos)

class Arctid(nn.Module):
    """§3.28, Eq. (217): atan(z)^2 - z."""
    def forward(self,x:Tensor)->Tensor: return torch.atan(x).pow(2)-x

class PlainSine(nn.Module):
    """§3.29, unscaled sine variant explicitly mentioned in the survey."""
    def forward(self,x:Tensor)->Tensor: return torch.sin(x)

class EvolvedMaxTanhLogReLU(nn.Module):
    """§3.40.1, Eq. (233): max(tanh(log(z)), ReLU(z)); domain z>0 for log."""
    def forward(self,x:Tensor)->Tensor: return torch.maximum(torch.tanh(torch.log(x)),F.relu(x))

class KDAC(nn.Module):
    """§3.45, Eqs. (238)-(245).

    The survey prints `h` in Eqs. (238),(243) but defines only h_max and h_min;
    this implementation uses h_max at those two occurrences, matching the
    survey's referenced implementation convention. a and b are adaptive as stated.
    """
    def __init__(self,a:float=1.0,b:float=1.0,c:float=0.01,k:float=1.0,trainable:bool=True):
        super().__init__(); self.c,self.k=c,k
        ta,tb=torch.tensor(float(a)),torch.tensor(float(b))
        if trainable:
            self.a=nn.Parameter(ta); self.b=nn.Parameter(tb)
        else:
            self.register_buffer('a',ta); self.register_buffer('b',tb)
    def _clip(self,x:Tensor)->Tensor: return x.clamp(0,1)
    def _hmax(self,x:Tensor,y:Tensor)->Tensor: return self._clip(0.5-0.5*(x-y)/self.c)
    def _hmin(self,x:Tensor,y:Tensor)->Tensor: return self._clip(0.5+0.5*(x-y)/self.c)
    def forward(self,z:Tensor)->Tensor:
        p=self.a*z; s=torch.tanh(z); q=self._hmin(self.b*z,s)
        rneg=self.b*z*(1-q)+s*self._hmax(q,s)+self.k*q*(1-q)
        r=torch.where(z>0,p,rneg)
        hpr=self._hmax(p,r)
        return p*(1-hpr)+r*hpr+self.k*hpr*(1-hpr)

class VolatilityBasedActivation(nn.Module):
    """§3.47, Eqs. (247)-(248), *literal survey formula*.

    The printed formula has no square on (mean-z_j), so the radicand can be
    negative and, algebraically, sums to zero in exact arithmetic. This class
    intentionally preserves the printed formula rather than silently replacing
    it with a standard deviation.
    """
    def __init__(self,dim:int=-1,keepdim:bool=False): super().__init__(); self.dim,self.keepdim=dim,keepdim
    def forward(self,x:Tensor)->Tensor:
        mean=x.mean(dim=self.dim,keepdim=True)
        return torch.sqrt(torch.sum(mean-x,dim=self.dim,keepdim=self.keepdim)/x.shape[self.dim])

# Additional explicit variants/formulas mentioned inside Section 3.
class HardSigmoid02(nn.Module):
    """§3.6.17, Eq. (113): max(0,min(0.2z+0.5,1))."""
    def forward(self,x:Tensor)->Tensor: return torch.clamp(0.2*x+0.5,0.0,1.0)

class MaxTanh(nn.Module):
    """§3.7.1 mentioned maxtanh: max(z,tanh(z))."""
    def forward(self,x:Tensor)->Tensor: return torch.maximum(x,torch.tanh(x))

# Survey aliases / independently named identical functions.
S_RReLU = SoftsignRandomizedLeakyReLU
RTReLU = RandomlyTranslationalReLU
SCAA = SpatialContextAwareActivation
BAF = BipolarActivationFunction
OPLU = OrthogonalPermutationLinearUnit
ABReLU = AverageBiasedReLU
DRLU = DelayReLU
DisReLU = DisplacedReLU
MLReLU = ModifiedLeakyReLU
FTS = FlattedTSwish
OAF = OptimalActivationFunction
ADA = ApicalDendriteActivation
LADA = LeakyApicalDendriteActivation
FPFLU = FasterPFLU
sSELU = ScaledScaledELU
SQMAX = SquareSoftmax
PMAF = PiecewiseMexicanHat
Root2sigmoid = Root2Sigmoid
FPLUS = PolyLU
LRTLU = LSPTLU
DLU = SignReLU
ReLUSwish = FlattedTSwish

# Coverage metadata: every numbered Section 3 subsection that presents an AF.
SECTION3_COVERAGE = {
    '3.1': ['BinaryStep'],
    '3.2': ['Sigmoid','Probit','Tanh','ScaledSigmoid','ScaledTanhHalf'],
    '3.2.1': ['ShiftedScaledSigmoid'], '3.2.2': ['VariantSigmoid'],
    '3.2.3': ['ScaledTanh','BimodalSigmoid'], '3.2.4': ['Arctan','ArctanGR'],
    '3.2.5': ['SigmoidAlgebraic'], '3.2.6': ['TripleStateSigmoid'],
    '3.2.7': ['ImprovedLogisticSigmoid'], '3.2.8': ['SigLin'],
    '3.2.9': ['PenalizedTanh'], '3.2.10': ['SoftRootSign'], '3.2.11': ['SoftClipping'],
    '3.2.12': ['Hexpo'], '3.2.13': ['Softsign'], '3.2.14': ['SmoothStep'],
    '3.2.15': ['Elliott'], '3.2.16': ['SincSigmoid'], '3.2.17': ['SigmoidGumbel'],
    '3.2.18': ['NewSigmoid'], '3.2.19': ['Root2Sigmoid'], '3.2.20': ['LogLog'],
    '3.2.21': ['ComplementaryLogLog','ModifiedComplementaryLogLog'],
    '3.2.22': ['SechSig','ParametricSechSig'], '3.2.23': ['TanhSig','ParametricTanhSig'],
    '3.2.24': ['MultiStateActivation','SymmetricalMSAF'],
    '3.2.25': ['Rootsig','UnnamedSigmoid37','UnnamedSigmoid38','UnnamedSigmoid39'],
    '3.2.26': ['BipolarSigmoidTanh','BipolarSigmoidZero','TanhZero','BipolarSigmoidLinear','TanhLinearCombination'],
    '3.3': ['SiLU','WeightedSigmoidGate'], '3.3.1': ['GELU'], '3.3.2': ['SGELU'],
    '3.3.3': ['CauchyLinearUnit'], '3.3.4': ['LaplaceLinearUnit'], '3.3.5': ['CollapsingLinearUnit'],
    '3.3.6': ['TripleStateSwish'], '3.3.7': ['GeneralizedSwish'], '3.3.8': ['ExponentialSwish'],
    '3.3.9': ['SigmoidDerivative'], '3.3.10': ['Gish'], '3.3.11': ['Logish'],
    '3.3.12': ['LogLogish'], '3.3.13': ['ExpExpish'], '3.3.14': ['SelfArctan'],
    '3.3.15': ['ParametricLogish'], '3.3.16': ['Phish'], '3.3.17': ['Suish'],
    '3.3.18': ['TSReLU'], '3.3.19': ['TBSReLU'], '3.3.20': ['LogSigmoid'],
    '3.3.21': ['DSiLU'], '3.3.22': ['DoubleSiLU'], '3.3.23': ['ModifiedSiLU'],
    '3.3.24': ['TSiLU'], '3.3.25': ['ASiLU'], '3.3.26': ['SwAT'],
    '3.3.27': ['RectifiedHyperbolicSecant'], '3.3.28': ['LiSHT'], '3.3.29': ['Mish'],
    '3.3.30': ['Smish'], '3.3.31': ['TanhExp'], '3.3.32': ['Serf'],
    '3.3.33': ['EANAF'], '3.3.34': ['SinSig'], '3.3.35': ['SiELU'],
    '3.4': ['GLU'], '3.4.1': ['GTU'], '3.4.2': ['ReGLU'], '3.4.3': ['GEGLU'], '3.4.4': ['SwiGLU'],
    '3.5': ['Softmax'], '3.5.1': ['BetaSoftmax'],
    '3.6': ['ReLU'], '3.6.1': ['ShiftedReLU'], '3.6.2': ['LeakyReLU','VeryLeakyReLU','OptimizedLeakyReLU'],
    '3.6.3': ['RandomizedLeakyReLU'], '3.6.4': ['SoftsignRandomizedLeakyReLU'], '3.6.5': ['SlopedReLU'],
    '3.6.6': ['NoisyReLU'], '3.6.7': ['SineReLU'], '3.6.8': ['Minsin'], '3.6.9': ['VariationalLinearUnit'],
    '3.6.10': ['SpatialContextAwareActivation'], '3.6.11': ['RandomlyTranslationalReLU'],
    '3.6.12': ['NaturalLogReLU'], '3.6.13': ['SoftplusLinearUnit'], '3.6.14': ['RectifiedSoftplus'],
    '3.6.15': ['ParametricRectifiedNonlinearUnit'], '3.6.16': ['BoundedReLU'],
    '3.6.17': ['HardSigmoid','HardSigmoid02'], '3.6.18': ['HardTanh'],
    '3.6.19': ['VerticalShiftedHardTanh','HorizontalShiftedHardTanh'], '3.6.20': ['HardSwish'],
    '3.6.21': ['TruncatedRectified'], '3.6.22': ['HardShrink'], '3.6.23': ['SoftShrink'],
    '3.6.24': ['BoundedLeakyReLU'], '3.6.25': ['VReLU'], '3.6.26': ['Pan'], '3.6.27': ['AbsLU'],
    '3.6.28': ['MirroredReLU'], '3.6.29': ['LSPTLU'], '3.6.30': ['SoftModulusQ'],
    '3.6.31': ['SoftModulusT'], '3.6.32': ['SignReLU'], '3.6.33': ['LiReLU'],
    '3.6.34': ['CReLU'], '3.6.35': ['NCReLU','BipolarActivationFunction'], '3.6.36': ['DualReLU'],
    '3.6.37': ['OrthogonalPermutationLinearUnit'], '3.6.38': ['ElasticReLU'],
    '3.6.39': ['RePU','AlternatingRePU'], '3.6.40': ['ApproximateReLU'],
    '3.6.41': ['EvenPowerLinearActivation','OddPowerLinearActivation'], '3.6.42': ['AverageBiasedReLU'],
    '3.6.43': ['DelayReLU'], '3.6.44': ['DisplacedReLU'], '3.6.45': ['ModifiedLeakyReLU'],
    '3.6.46': ['FlattedTSwish'], '3.6.47': ['OptimalActivationFunction'], '3.6.48': ['ELU'],
    '3.6.49': ['REU'], '3.6.50': ['ApicalDendriteActivation'], '3.6.51': ['LeakyApicalDendriteActivation'],
    '3.6.52': ['SigLU'], '3.6.53': ['SaRa'],
    '3.7': ['Maxsig'], '3.7.1': ['TanhLinearUnit','MaxTanh'], '3.7.2': ['DualELU'],
    '3.7.3': ['DifferenceELU'], '3.7.4': ['PolyLU'], '3.7.5': ['InversePolyLU'], '3.7.6': ['PoLU'],
    '3.7.7': ['PFLU'], '3.7.8': ['FasterPFLU'], '3.7.9': ['EACU'], '3.7.10': ['LipschitzReLU'],
    '3.7.11': ['SELU'], '3.7.12': ['LSELU'], '3.7.13': ['SERLU'], '3.7.14': ['ScaledScaledELU'],
    '3.7.15': ['RSigELU'], '3.7.16': ['HardSReLUE'], '3.7.17': ['ELiSH'], '3.7.18': ['HardELiSH'],
    '3.7.19': ['RSigELUD'], '3.7.20': ['LSReLU'],
    '3.8.1': ['SQNL'], '3.8.2': ['SQLU'], '3.8.3': ['Squish'], '3.8.4': ['SqREU'],
    '3.8.5': ['SqSoftplus'], '3.8.6': ['LogSQNL'], '3.8.7': ['SquareSoftmax'], '3.8.8': ['LinearQuadratic'],
    '3.8.9': ['ISRLU'], '3.8.10': ['ISRU'], '3.8.11': ['ModifiedElliott'],
    '3.9': ['SQRT','SSAF'], '3.10': ['BentIdentity'], '3.11': ['Mishra'], '3.12': ['SBAF'],
    '3.13': ['LogarithmicActivation'], '3.14': ['Symexp'], '3.15': ['SPOCU'], '3.16': ['PUAF'],
    '3.17': ['Softplus'], '3.18': ['ParametricSoftplus'], '3.18.1': ['SoftPlusPlus'], '3.19': ['RandSoftplus'],
    '3.20': ['ArandaOrdaz'], '3.21': ['BiFiring'], '3.22': ['BoundedBiFiring'], '3.23': ['PiecewiseMexicanHat'],
    '3.24': ['PiecewiseRBF'], '3.25': ['CombHSine'], '3.26': ['ModifiedArcsinh'], '3.27': ['HyperSinh'],
    '3.28': ['Arctid'], '3.29': ['Sine','PlainSine','ScaledShiftedSine'], '3.30': ['Cosine'],
    '3.31': ['Cosid'], '3.32': ['Sinp'], '3.33': ['GrowingCosineUnit'], '3.34': ['AmplifyingSineUnit'],
    '3.35': ['Sinc','ShiftedSineUnit'], '3.36': ['DecayingSineUnit'], '3.37': ['HcLSH'], '3.38': ['Polyexp'],
    '3.39': ['Exponential'], '3.40': ['ETanh'], '3.40.1': ['EvolvedTanhReLU','EvolvedMaxTanhLogReLU'],
    '3.41': ['Wave'], '3.42': ['NonMonotonicCubicUnit'], '3.43': ['Triple'], '3.44': ['ShiftedQuadraticUnit'],
    '3.45': ['KDAC'], '3.46': ['KWinnersTakeAll'], '3.47': ['VolatilityBasedActivation'],
    '3.48.1': ['HybridChaoticActivation'], '3.48.2': ['FusionChaoticActivation'], '3.48.3': ['CascadeChaoticActivation'],
}

# Items mentioned by the survey but not mathematically specified well enough there
# to implement from Section 3 alone without importing an external source.
SURVEY_UNSPECIFIED = {
    'n-sigmoid': 'The survey explicitly omits it because the source formula appears unintended (§3.2).',
    'tanh36/tanh3': 'Named spline approximations are mentioned but their spline coefficients are not printed in Section 3.',
    'LRTanh': 'Mentioned with a modified-backprop derivative, but no activation formula is printed in Section 3.',
    'pRPPSG/piecewise sigmoid approximations': 'Mentioned only by reference; formulas are not printed.',
    'three additional bimodal-derivative sigmoids': 'The survey prints only Eq. (8) and says three more exist in the cited source.',
    'nonadaptive PTELU': 'Mentioned in §3.2.26 but its formula is not printed there.',
}

# Rebuild aliases and registry after the audited definitions.
LReLU = LeakyReLU
RReLU = RandomizedLeakyReLU
BReLU = BoundedReLU
CReLUActivation = CReLU
ISRLUActivation = ISRLU
ISRUActivation = ISRU

REGISTRY = {
    name: cls for name, cls in globals().copy().items()
    if isinstance(cls, type) and issubclass(cls, nn.Module) and cls is not nn.Module
}

# Final small audit additions discovered during page-image verification.
class GELUTanhApprox(nn.Module):
    """§3.3.1, Eq. (46)."""
    def forward(self,x:Tensor)->Tensor:
        return 0.5*x*(1.0+torch.tanh(math.sqrt(2.0/math.pi)*(x+0.044715*x.pow(3))))

class GELUSigmoidApprox(nn.Module):
    """§3.3.1, Eq. (47)."""
    def forward(self,x:Tensor)->Tensor: return x*torch.sigmoid(1.702*x)

class ReQU(RePU):
    """§3.6.39: rectified quadratic unit, RePU with exponent 2."""
    def __init__(self): super().__init__(p=2.0)

class ReCU(RePU):
    """§3.6.39: rectified cubic unit, RePU with exponent 3."""
    def __init__(self): super().__init__(p=3.0)

class HcLSH(nn.Module):
    """§3.37, Eq. (228), verified against the rendered PDF page."""
    def forward(self,x:Tensor)->Tensor:
        pos=torch.log(torch.cosh(x)+x*torch.cosh(x/2.0))
        neg=torch.log(torch.cosh(x))+x
        return torch.where(x>=0,pos,neg)

Symlog = LogarithmicActivation
SinPN = Sinp

SECTION3_COVERAGE['3.3.1'] = ['GELU','GELUTanhApprox','GELUSigmoidApprox']
SECTION3_COVERAGE['3.6.39'] = ['RePU','ReQU','ReCU','AlternatingRePU']
SECTION3_COVERAGE['3.37'] = ['HcLSH']

# Rebuild registry one last time after final audit additions.
REGISTRY = {
    name: cls for name, cls in globals().copy().items()
    if isinstance(cls, type) and issubclass(cls, nn.Module) and cls is not nn.Module
}
