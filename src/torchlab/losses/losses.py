"""
missing_losses.py

A broad collection of loss functions that are not exposed as dedicated
torch.nn loss modules in core PyTorch (PyTorch 2.13 docs checked Aug 2026).

Notes
-----
* Some functions (e.g. sigmoid focal loss and IoU box losses) exist in
  torchvision.ops; they are not duplicated here unless the implementation
  extends beyond the torchvision variant.
* Inputs are logits unless a class explicitly says probabilities.
* All implementations are autograd-compatible.
* `reduction` follows PyTorch conventions where meaningful.
* Research-loss conventions vary between papers. This file documents the
  exact convention implemented for each loss.

Primary references are named in docstrings/comments. No implementation code
has been copied from third-party projects.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reduce(x: Tensor, reduction: str = "mean") -> Tensor:
    if reduction == "none":
        return x
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")

def _eps(x: Tensor, eps: float) -> Tensor:
    return torch.as_tensor(eps, dtype=x.dtype, device=x.device)

def _one_hot(target: Tensor, c: int, dtype: torch.dtype) -> Tensor:
    # N,... -> N,C,...
    return F.one_hot(target.long(), c).movedim(-1, 1).to(dtype)

def _flatten_binary(probs: Tensor, target: Tensor):
    return probs.reshape(probs.shape[0], -1), target.reshape(target.shape[0], -1)

def _safe_log(x: Tensor, eps: float = 1e-7) -> Tensor:
    return torch.log(x.clamp_min(eps))

def _cosine_matrix(x: Tensor) -> Tensor:
    x = F.normalize(x, dim=-1)
    return x @ x.T


# ---------------------------------------------------------------------------
# Classification / noisy-label losses
# ---------------------------------------------------------------------------

class BinaryFocalLoss(nn.Module):
    """Lin et al. 2017 focal loss: -alpha_t (1-p_t)^gamma log(p_t)."""
    def __init__(self, alpha: float = .25, gamma: float = 2., reduction="mean"):
        super().__init__(); self.alpha=alpha; self.gamma=gamma; self.reduction=reduction
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        target=target.to(logits.dtype)
        bce=F.binary_cross_entropy_with_logits(logits,target,reduction="none")
        p=torch.sigmoid(logits)
        pt=p*target+(1-p)*(1-target)
        at=self.alpha*target+(1-self.alpha)*(1-target)
        return _reduce(at*(1-pt).pow(self.gamma)*bce,self.reduction)

class MulticlassFocalLoss(nn.Module):
    """Multiclass focal loss using p_t from softmax."""
    def __init__(self, gamma=2., alpha: Optional[Tensor]=None, reduction="mean"):
        super().__init__(); self.gamma=gamma; self.reduction=reduction
        self.register_buffer("alpha", alpha if alpha is not None else None)
    def forward(self, logits, target):
        logp=F.log_softmax(logits,dim=1)
        logpt=logp.gather(1,target.unsqueeze(1)).squeeze(1)
        pt=logpt.exp()
        loss=-(1-pt).pow(self.gamma)*logpt
        if self.alpha is not None: loss=loss*self.alpha[target]
        return _reduce(loss,self.reduction)

class AsymmetricLoss(nn.Module):
    """Ridnik et al. 2021 ASL for multi-label classification."""
    def __init__(self,gamma_neg=4.,gamma_pos=1.,clip=.05,eps=1e-8,reduction="mean"):
        super().__init__(); self.gn=gamma_neg; self.gp=gamma_pos; self.clip=clip; self.eps=eps; self.reduction=reduction
    def forward(self,logits,target):
        y=target.to(logits.dtype)
        p=torch.sigmoid(logits)
        pn=(1-p)
        if self.clip>0: pn=(pn+self.clip).clamp(max=1)
        logloss=y*_safe_log(p,self.eps)+(1-y)*_safe_log(pn,self.eps)
        pt=p*y+pn*(1-y)
        gamma=self.gp*y+self.gn*(1-y)
        return _reduce(-logloss*(1-pt).pow(gamma),self.reduction)

class Poly1CrossEntropyLoss(nn.Module):
    """Poly-1 CE (Leng et al. 2022): CE + epsilon*(1-p_t)."""
    def __init__(self,epsilon=1.,reduction="mean"):
        super().__init__(); self.epsilon=epsilon; self.reduction=reduction
    def forward(self,logits,target):
        ce=F.cross_entropy(logits,target,reduction="none")
        pt=F.softmax(logits,dim=1).gather(1,target[:,None]).squeeze(1)
        return _reduce(ce+self.epsilon*(1-pt),self.reduction)

class Poly1FocalLoss(nn.Module):
    """Poly-1 focal: FL + epsilon*(1-p_t)^(gamma+1)."""
    def __init__(self,epsilon=1.,gamma=2.,alpha=.25,reduction="mean"):
        super().__init__(); self.epsilon=epsilon; self.gamma=gamma; self.alpha=alpha; self.reduction=reduction
    def forward(self,logits,target):
        y=target.to(logits.dtype)
        p=torch.sigmoid(logits); pt=p*y+(1-p)*(1-y)
        bce=F.binary_cross_entropy_with_logits(logits,y,reduction="none")
        at=self.alpha*y+(1-self.alpha)*(1-y)
        fl=at*(1-pt).pow(self.gamma)*bce
        return _reduce(fl+self.epsilon*(1-pt).pow(self.gamma+1),self.reduction)

class GeneralizedCrossEntropyLoss(nn.Module):
    """Zhang & Sabuncu 2018 L_q=(1-p_y^q)/q; q->0 gives CE."""
    def __init__(self,q=.7,reduction="mean"):
        super().__init__(); self.q=q; self.reduction=reduction
    def forward(self,logits,target):
        py=F.softmax(logits,1).gather(1,target[:,None]).squeeze(1)
        if abs(self.q)<1e-8: loss=-_safe_log(py)
        else: loss=(1-py.pow(self.q))/self.q
        return _reduce(loss,self.reduction)

class ReverseCrossEntropyLoss(nn.Module):
    """RCE from Wang et al. 2019. One-hot targets are clipped before log."""
    def __init__(self,eps=1e-4,reduction="mean"):
        super().__init__(); self.eps=eps; self.reduction=reduction
    def forward(self,logits,target):
        p=F.softmax(logits,1)
        y=_one_hot(target,logits.shape[1],logits.dtype).clamp(self.eps,1)
        return _reduce(-(p*_safe_log(y,self.eps)).sum(1),self.reduction)

class SymmetricCrossEntropyLoss(nn.Module):
    """SCE = alpha*CE + beta*RCE (Wang et al. 2019)."""
    def __init__(self,alpha=1.,beta=1.,eps=1e-4,reduction="mean"):
        super().__init__(); self.alpha=alpha; self.beta=beta; self.eps=eps; self.reduction=reduction
    def forward(self,logits,target):
        ce=F.cross_entropy(logits,target,reduction="none")
        p=F.softmax(logits,1)
        y=_one_hot(target,logits.shape[1],logits.dtype).clamp(self.eps,1)
        rce=-(p*_safe_log(y,self.eps)).sum(1)
        return _reduce(self.alpha*ce+self.beta*rce,self.reduction)

class NormalizedCrossEntropyLoss(nn.Module):
    """NCE used in robust-label-loss literature: -log p_y / -sum_c log p_c."""
    def __init__(self,eps=1e-7,reduction="mean"):
        super().__init__(); self.eps=eps; self.reduction=reduction
    def forward(self,logits,target):
        lp=F.log_softmax(logits,1)
        num=-lp.gather(1,target[:,None]).squeeze(1)
        den=-lp.sum(1)
        return _reduce(num/den.clamp_min(self.eps),self.reduction)

class BrierLoss(nn.Module):
    """Multiclass Brier score: sum_c (p_c-y_c)^2."""
    def __init__(self,reduction="mean"): super().__init__(); self.reduction=reduction
    def forward(self,logits,target):
        p=F.softmax(logits,1); y=_one_hot(target,logits.shape[1],logits.dtype)
        return _reduce((p-y).pow(2).sum(1),self.reduction)

class LabelDistributionFocalLoss(nn.Module):
    """Quality Focal Loss style continuous target y in [0,1]:
       BCE(logit,y)*|y-sigmoid(logit)|^beta.
       Li et al., Generalized Focal Loss, 2020.
    """
    def __init__(self,beta=2.,reduction="mean"): super().__init__(); self.beta=beta; self.reduction=reduction
    def forward(self,logits,target):
        y=target.to(logits.dtype); p=torch.sigmoid(logits)
        loss=F.binary_cross_entropy_with_logits(logits,y,reduction="none")*(y-p).abs().pow(self.beta)
        return _reduce(loss,self.reduction)

class DistributionFocalLoss(nn.Module):
    """GFL distribution focal loss for a scalar target between adjacent bins."""
    def __init__(self,reduction="mean"): super().__init__(); self.reduction=reduction
    def forward(self,logits,target):
        # logits: N,K ; target in [0,K-1]
        left=torch.floor(target).long()
        right=(left+1).clamp_max(logits.shape[1]-1)
        wl=(right.to(target.dtype)-target)
        wr=(target-left.to(target.dtype))
        same=(left==right)
        wl=torch.where(same,torch.ones_like(wl),wl)
        wr=torch.where(same,torch.zeros_like(wr),wr)
        l=F.cross_entropy(logits,left,reduction="none")*wl
        r=F.cross_entropy(logits,right,reduction="none")*wr
        return _reduce(l+r,self.reduction)

class VarifocalLoss(nn.Module):
    """Zhang et al. 2020: BCE weighted by q for positives and
       alpha*p^gamma for negatives. target is IoU-aware score q in [0,1].
    """
    def __init__(self,alpha=.75,gamma=2.,reduction="mean"):
        super().__init__(); self.alpha=alpha; self.gamma=gamma; self.reduction=reduction
    def forward(self,logits,target):
        q=target.to(logits.dtype); p=torch.sigmoid(logits)
        weight=torch.where(q>0,q,self.alpha*p.pow(self.gamma))
        return _reduce(F.binary_cross_entropy_with_logits(logits,q,reduction="none")*weight,self.reduction)


# ---------------------------------------------------------------------------
# Segmentation / overlap / topology losses
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft Sørensen-Dice loss. V-Net-style squared denominator optional."""
    def __init__(self,smooth=1e-6,from_logits=True,squared_denominator=False,batch=False):
        super().__init__(); self.smooth=smooth; self.from_logits=from_logits; self.squared=squared_denominator; self.batch=batch
    def forward(self,pred,target):
        p=torch.sigmoid(pred) if self.from_logits else pred
        t=target.to(p.dtype)
        dims=tuple(range(0,p.ndim)) if self.batch else tuple(range(1,p.ndim))
        inter=(p*t).sum(dims)
        if self.squared: den=(p*p+t*t).sum(dims)
        else: den=(p+t).sum(dims)
        return 1-((2*inter+self.smooth)/(den+self.smooth)).mean()

class GeneralizedDiceLoss(nn.Module):
    """Sudre et al. 2017, inverse squared class-volume weights."""
    def __init__(self,smooth=1e-6,from_logits=True):
        super().__init__(); self.smooth=smooth; self.from_logits=from_logits
    def forward(self,pred,target):
        # pred N,C,... target N,... or one-hot N,C,...
        p=F.softmax(pred,1) if self.from_logits else pred
        t=_one_hot(target,p.shape[1],p.dtype) if target.ndim==p.ndim-1 else target.to(p.dtype)
        dims=(0,)+tuple(range(2,p.ndim))
        vol=t.sum(dims)
        w=1/(vol.pow(2)+self.smooth)
        inter=(p*t).sum(dims)
        den=(p+t).sum(dims)
        score=2*(w*inter).sum()/(w*den).sum().clamp_min(self.smooth)
        return 1-score

class TverskyLoss(nn.Module):
    """Salehi et al. 2017: 1 - TP/(TP + alpha*FP + beta*FN)."""
    def __init__(self,alpha=.5,beta=.5,smooth=1e-6,from_logits=True):
        super().__init__(); self.alpha=alpha; self.beta=beta; self.smooth=smooth; self.from_logits=from_logits
    def forward(self,pred,target):
        p=torch.sigmoid(pred) if self.from_logits else pred; t=target.to(p.dtype)
        dims=tuple(range(1,p.ndim))
        tp=(p*t).sum(dims); fp=(p*(1-t)).sum(dims); fn=((1-p)*t).sum(dims)
        tv=(tp+self.smooth)/(tp+self.alpha*fp+self.beta*fn+self.smooth)
        return 1-tv.mean()

class FocalTverskyLoss(nn.Module):
    """Abraham & Khan 2018 Eq.(4): (1-TI)^(1/gamma).
    Their Eq.(3) uses alpha on false negatives and beta on false positives
    (opposite the alpha/beta naming in Salehi et al.'s original Tversky paper).
    """
    def __init__(self,alpha=.7,beta=.3,gamma=4/3,smooth=1e-6,from_logits=True):
        super().__init__(); self.alpha=alpha; self.beta=beta; self.gamma=gamma; self.smooth=smooth; self.from_logits=from_logits
    def forward(self,pred,target):
        p=torch.sigmoid(pred) if self.from_logits else pred; t=target.to(p.dtype)
        dims=tuple(range(1,p.ndim))
        tp=(p*t).sum(dims); fp=(p*(1-t)).sum(dims); fn=((1-p)*t).sum(dims)
        tv=(tp+self.smooth)/(tp+self.alpha*fn+self.beta*fp+self.smooth)
        return (1-tv).pow(1/self.gamma).mean()

class JaccardLoss(nn.Module):
    """Soft IoU/Jaccard loss."""
    def __init__(self,smooth=1e-6,from_logits=True):
        super().__init__(); self.smooth=smooth; self.from_logits=from_logits
    def forward(self,pred,target):
        p=torch.sigmoid(pred) if self.from_logits else pred; t=target.to(p.dtype)
        dims=tuple(range(1,p.ndim))
        inter=(p*t).sum(dims); union=(p+t-p*t).sum(dims)
        return (1-(inter+self.smooth)/(union+self.smooth)).mean()

def _lovasz_grad(gt_sorted: Tensor) -> Tensor:
    p=len(gt_sorted); gts=gt_sorted.sum()
    inter=gts-gt_sorted.cumsum(0)
    union=gts+(1-gt_sorted).cumsum(0)
    j=1-inter/union
    if p>1: j=torch.cat([j[:1], j[1:p]-j[:-1]])
    return j

class LovaszHingeLoss(nn.Module):
    """Binary Lovász hinge from Berman et al. 2018."""
    def forward(self,logits,target):
        losses=[]
        for logit,y in zip(logits,target):
            logit=logit.reshape(-1); y=y.reshape(-1).to(logit.dtype)
            signs=2*y-1; errors=1-logit*signs
            errors,perm=torch.sort(errors,descending=True)
            y_sorted=y[perm]
            losses.append(torch.dot(F.relu(errors),_lovasz_grad(y_sorted)))
        return torch.stack(losses).mean()

class BoundaryLoss(nn.Module):
    """Kervadec et al. boundary loss: mean(probability * signed distance map).
       `signed_distance` must be precomputed from the ground-truth mask.
    """
    def __init__(self,from_logits=True): super().__init__(); self.from_logits=from_logits
    def forward(self,pred,signed_distance):
        p=torch.sigmoid(pred) if self.from_logits else pred
        return (p*signed_distance.to(p.dtype)).mean()

class HausdorffDTLoss(nn.Module):
    """Karimi & Salcudean distance-transform surrogate:
       mean((p-y)^2 * (d_pred^alpha + d_gt^alpha)).
       Distance maps are supplied by caller to avoid non-differentiable DT code.
    """
    def __init__(self,alpha=2.,from_logits=True): super().__init__(); self.alpha=alpha; self.from_logits=from_logits
    def forward(self,pred,target,d_pred,d_target):
        p=torch.sigmoid(pred) if self.from_logits else pred; y=target.to(p.dtype)
        return ((p-y).pow(2)*(d_pred.pow(self.alpha)+d_target.pow(self.alpha))).mean()

def _soft_erode(img: Tensor) -> Tensor:
    if img.ndim==4:
        p1=-F.max_pool2d(-img,(3,1),(1,1),(1,0))
        p2=-F.max_pool2d(-img,(1,3),(1,1),(0,1))
    elif img.ndim==5:
        p1=-F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
        p2=-F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
        p3=-F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
        return torch.minimum(torch.minimum(p1,p2),p3)
    else: raise ValueError("soft-clDice expects N,C,H,W or N,C,D,H,W")
    return torch.minimum(p1,p2)

def _soft_dilate(img: Tensor) -> Tensor:
    return F.max_pool2d(img,3,1,1) if img.ndim==4 else F.max_pool3d(img,3,1,1)

def _soft_open(img: Tensor) -> Tensor: return _soft_dilate(_soft_erode(img))

def _soft_skel(img: Tensor,iters:int) -> Tensor:
    skel=F.relu(img-_soft_open(img))
    for _ in range(iters):
        img=_soft_erode(img)
        delta=F.relu(img-_soft_open(img))
        skel=skel+F.relu(delta-skel*delta)
    return skel

class SoftCLDiceLoss(nn.Module):
    """Differentiable soft-clDice, Shit et al. 2020."""
    def __init__(self,iters=10,smooth=1.,from_logits=True):
        super().__init__(); self.iters=iters; self.smooth=smooth; self.from_logits=from_logits
    def forward(self,pred,target):
        p=torch.sigmoid(pred) if self.from_logits else pred; t=target.to(p.dtype)
        sp=_soft_skel(p,self.iters); st=_soft_skel(t,self.iters)
        tprec=(sp*t).sum()+self.smooth
        tprec=tprec/(sp.sum()+self.smooth)
        tsens=(st*p).sum()+self.smooth
        tsens=tsens/(st.sum()+self.smooth)
        return 1-(2*tprec*tsens)/(tprec+tsens)

class ComboLoss(nn.Module):
    """Dice + weighted BCE combination (Taghanaki et al. 2019-style)."""
    def __init__(self,alpha=.5,beta=.5,smooth=1e-6):
        super().__init__(); self.alpha=alpha; self.beta=beta; self.smooth=smooth
    def forward(self,logits,target):
        y=target.to(logits.dtype)
        p=torch.sigmoid(logits)
        bce=-(self.beta*y*_safe_log(p)+(1-self.beta)*(1-y)*_safe_log(1-p)).mean()
        dims=tuple(range(1,p.ndim)); inter=(p*y).sum(dims); den=(p+y).sum(dims)
        dice=((2*inter+self.smooth)/(den+self.smooth)).mean()
        return self.alpha*bce-(1-self.alpha)*dice

class LogCoshDiceLoss(nn.Module):
    """log(cosh(DiceLoss)), a smooth transformation of soft Dice loss."""
    def __init__(self,**dice_kwargs): super().__init__(); self.dice=DiceLoss(**dice_kwargs)
    def forward(self,pred,target):
        x=self.dice(pred,target)
        return x + F.softplus(-2*x) - math.log(2.0)


# ---------------------------------------------------------------------------
# Regression / robust losses
# ---------------------------------------------------------------------------

class RMSELoss(nn.Module):
    def __init__(self,eps=1e-12): super().__init__(); self.eps=eps
    def forward(self,pred,target): return torch.sqrt(F.mse_loss(pred,target)+self.eps)

class LogCoshLoss(nn.Module):
    """Stable log(cosh(error))."""
    def __init__(self,reduction="mean"): super().__init__(); self.reduction=reduction
    def forward(self,pred,target):
        x=pred-target
        return _reduce(x+F.softplus(-2*x)-math.log(2.0),self.reduction)

class QuantileLoss(nn.Module):
    """Pinball/check loss: max(q*e,(q-1)*e), e=y-yhat."""
    def __init__(self,q=.5,reduction="mean"):
        super().__init__(); self.q=q; self.reduction=reduction
    def forward(self,pred,target):
        e=target-pred
        return _reduce(torch.maximum(self.q*e,(self.q-1)*e),self.reduction)

class CharbonnierLoss(nn.Module):
    """sqrt(error^2 + eps^2), optionally minus eps so rho(0)=0."""
    def __init__(self,eps=1e-3,zero_at_zero=True,reduction="mean"):
        super().__init__(); self.eps=eps; self.zero=zero_at_zero; self.reduction=reduction
    def forward(self,pred,target):
        x=torch.sqrt((pred-target).pow(2)+self.eps**2)
        if self.zero: x=x-self.eps
        return _reduce(x,self.reduction)

class PseudoHuberLoss(nn.Module):
    """delta^2(sqrt(1+(e/delta)^2)-1)."""
    def __init__(self,delta=1.,reduction="mean"): super().__init__(); self.delta=delta; self.reduction=reduction
    def forward(self,pred,target):
        e=(pred-target)/self.delta
        return _reduce(self.delta**2*(torch.sqrt(1+e*e)-1),self.reduction)

class WingLoss(nn.Module):
    """Feng et al. 2018."""
    def __init__(self,w=10.,epsilon=2.,reduction="mean"):
        super().__init__(); self.w=w; self.epsilon=epsilon; self.reduction=reduction
        self.C=w-w*math.log(1+w/epsilon)
    def forward(self,pred,target):
        x=(pred-target).abs()
        loss=torch.where(x<self.w,self.w*torch.log1p(x/self.epsilon),x-self.C)
        return _reduce(loss,self.reduction)

class AdaptiveWingLoss(nn.Module):
    """Wang et al. 2019 Adaptive Wing loss."""
    def __init__(self,omega=14.,theta=.5,epsilon=1.,alpha=2.1,reduction="mean"):
        super().__init__(); self.omega=omega; self.theta=theta; self.epsilon=epsilon; self.alpha=alpha; self.reduction=reduction
    def forward(self,pred,target):
        y=target; x=(y-pred).abs()
        power=self.alpha-y
        # A and C chosen for continuity and matching derivative at theta.
        ratio=torch.as_tensor(self.theta/self.epsilon,dtype=pred.dtype,device=pred.device)
        t=ratio.pow(power)
        A=self.omega*(1/(1+t))*power*ratio.pow(power-1)/self.epsilon
        C=self.theta*A-self.omega*torch.log1p(t)
        loss=torch.where(x<self.theta,self.omega*torch.log1p((x/self.epsilon).pow(power)),A*x-C)
        return _reduce(loss,self.reduction)

class BerHuLoss(nn.Module):
    """Reverse Huber (berHu): |e| if <=c else (e^2+c^2)/(2c)."""
    def __init__(self,c:Optional[float]=None,fraction=.2,reduction="mean"):
        super().__init__(); self.c=c; self.fraction=fraction; self.reduction=reduction
    def forward(self,pred,target):
        e=(pred-target).abs()
        c=self.c if self.c is not None else self.fraction*e.detach().max().clamp_min(1e-12)
        loss=torch.where(e<=c,e,(e*e+c*c)/(2*c))
        return _reduce(loss,self.reduction)

class TukeyBiweightLoss(nn.Module):
    """Tukey's bisquare robust loss."""
    def __init__(self,c=4.685,reduction="mean"): super().__init__(); self.c=c; self.reduction=reduction
    def forward(self,pred,target):
        e=(pred-target); u=e/self.c
        inside=(self.c**2/6)*(1-(1-u*u).pow(3))
        loss=torch.where(u.abs()<=1,inside,torch.full_like(e,self.c**2/6))
        return _reduce(loss,self.reduction)

class CauchyLoss(nn.Module):
    """c^2/2 * log(1+(e/c)^2)."""
    def __init__(self,c=1.,reduction="mean"): super().__init__(); self.c=c; self.reduction=reduction
    def forward(self,pred,target):
        e=(pred-target)/self.c
        return _reduce(.5*self.c**2*torch.log1p(e*e),self.reduction)

class WelschLoss(nn.Module):
    """c^2/2 * (1-exp(-(e/c)^2))."""
    def __init__(self,c=1.,reduction="mean"): super().__init__(); self.c=c; self.reduction=reduction
    def forward(self,pred,target):
        u=(pred-target)/self.c
        return _reduce(.5*self.c**2*(1-torch.exp(-u*u)),self.reduction)

class GemanMcClureLoss(nn.Module):
    """0.5*e^2/(1+e^2/c^2), scaled to approach c^2/2."""
    def __init__(self,c=1.,reduction="mean"): super().__init__(); self.c=c; self.reduction=reduction
    def forward(self,pred,target):
        e=pred-target
        return _reduce(.5*e*e/(1+(e/self.c).pow(2)),self.reduction)

class BarronLoss(nn.Module):
    """Barron 2019 general robust loss rho(x; alpha, c), excluding density normalizer."""
    def __init__(self,alpha=1.,c=1.,reduction="mean"):
        super().__init__(); self.alpha=alpha; self.c=c; self.reduction=reduction
    def forward(self,pred,target):
        x=(pred-target)/self.c; a=float(self.alpha)
        if abs(a-2)<1e-8: loss=.5*x*x
        elif abs(a)<1e-8: loss=torch.log1p(.5*x*x)
        else:
            b=abs(a-2)
            loss=(b/a)*((x*x/b+1).pow(a/2)-1)
        return _reduce(loss,self.reduction)

class ScaleInvariantLogLoss(nn.Module):
    """SILog-style depth loss: sqrt(mean(d^2)-lambda*mean(d)^2)."""
    def __init__(self,lam=.85,eps=1e-8): super().__init__(); self.lam=lam; self.eps=eps
    def forward(self,pred,target):
        d=_safe_log(pred,self.eps)-_safe_log(target,self.eps)
        return torch.sqrt((d*d).mean()-self.lam*d.mean().pow(2)+self.eps)

class ScaleInvariantMSELoss(nn.Module):
    """Eigen et al. scale-invariant error without outer sqrt."""
    def __init__(self,lam=1.,eps=1e-8): super().__init__(); self.lam=lam; self.eps=eps
    def forward(self,pred,target):
        d=_safe_log(pred,self.eps)-_safe_log(target,self.eps)
        return (d*d).mean()-self.lam*d.mean().pow(2)


# ---------------------------------------------------------------------------
# Distribution / divergence losses
# ---------------------------------------------------------------------------

class JensenShannonLoss(nn.Module):
    """JSD(P||Q)=1/2 KL(P||M)+1/2 KL(Q||M). Inputs are logits by default."""
    def __init__(self,from_logits=True,eps=1e-7,reduction="mean"):
        super().__init__(); self.from_logits=from_logits; self.eps=eps; self.reduction=reduction
    def forward(self,p,q):
        if self.from_logits: p=F.softmax(p,-1); q=F.softmax(q,-1)
        m=.5*(p+q)
        js=.5*(p*(_safe_log(p,self.eps)-_safe_log(m,self.eps))).sum(-1)+.5*(q*(_safe_log(q,self.eps)-_safe_log(m,self.eps))).sum(-1)
        return _reduce(js,self.reduction)

class JeffreysDivergenceLoss(nn.Module):
    """KL(P||Q)+KL(Q||P)."""
    def __init__(self,from_logits=True,eps=1e-7,reduction="mean"):
        super().__init__(); self.from_logits=from_logits; self.eps=eps; self.reduction=reduction
    def forward(self,p,q):
        if self.from_logits: p=F.softmax(p,-1); q=F.softmax(q,-1)
        lp,lq=_safe_log(p,self.eps),_safe_log(q,self.eps)
        return _reduce((p*(lp-lq)+q*(lq-lp)).sum(-1),self.reduction)

class HellingerLoss(nn.Module):
    """H(P,Q)=||sqrt(P)-sqrt(Q)||_2/sqrt(2)."""
    def __init__(self,from_logits=True,eps=1e-12,reduction="mean"):
        super().__init__(); self.from_logits=from_logits; self.eps=eps; self.reduction=reduction
    def forward(self,p,q):
        if self.from_logits: p=F.softmax(p,-1); q=F.softmax(q,-1)
        h=torch.sqrt(.5*((torch.sqrt(p.clamp_min(self.eps))-torch.sqrt(q.clamp_min(self.eps)))**2).sum(-1))
        return _reduce(h,self.reduction)

class BhattacharyyaLoss(nn.Module):
    """Bhattacharyya distance = -log sum sqrt(pq)."""
    def __init__(self,from_logits=True,eps=1e-7,reduction="mean"):
        super().__init__(); self.from_logits=from_logits; self.eps=eps; self.reduction=reduction
    def forward(self,p,q):
        if self.from_logits: p=F.softmax(p,-1); q=F.softmax(q,-1)
        bc=torch.sqrt((p*q).clamp_min(0)).sum(-1)
        return _reduce(-_safe_log(bc,self.eps),self.reduction)

class TriangularDiscriminationLoss(nn.Module):
    """sum (p-q)^2/(p+q)."""
    def __init__(self,from_logits=True,eps=1e-7,reduction="mean"):
        super().__init__(); self.from_logits=from_logits; self.eps=eps; self.reduction=reduction
    def forward(self,p,q):
        if self.from_logits: p=F.softmax(p,-1); q=F.softmax(q,-1)
        return _reduce(((p-q).pow(2)/(p+q+self.eps)).sum(-1),self.reduction)

class TotalVariationDistanceLoss(nn.Module):
    """1/2 * L1 distance between probability vectors."""
    def __init__(self,from_logits=True,reduction="mean"):
        super().__init__(); self.from_logits=from_logits; self.reduction=reduction
    def forward(self,p,q):
        if self.from_logits: p=F.softmax(p,-1); q=F.softmax(q,-1)
        return _reduce(.5*(p-q).abs().sum(-1),self.reduction)

class Wasserstein1DLoss(nn.Module):
    """Discrete 1-Wasserstein distance for ordered equal-spaced bins:
       sum |CDF_p-CDF_q|.
    """
    def __init__(self,from_logits=True,reduction="mean"):
        super().__init__(); self.from_logits=from_logits; self.reduction=reduction
    def forward(self,p,q):
        if self.from_logits: p=F.softmax(p,-1); q=F.softmax(q,-1)
        return _reduce((p.cumsum(-1)-q.cumsum(-1)).abs().sum(-1),self.reduction)


# ---------------------------------------------------------------------------
# Metric / representation learning
# ---------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """Hadsell et al. 2006: y*d^2 + (1-y)*max(m-d,0)^2."""
    def __init__(self,margin=1.,reduction="mean"):
        super().__init__(); self.margin=margin; self.reduction=reduction
    def forward(self,x1,x2,similar):
        d=F.pairwise_distance(x1,x2)
        y=similar.to(d.dtype)
        loss=.5*(y*d.pow(2)+(1-y)*F.relu(self.margin-d).pow(2))
        return _reduce(loss,self.reduction)

class NTXentLoss(nn.Module):
    """SimCLR NT-Xent. Input z1,z2 shape N,D; positives are matching rows."""
    def __init__(self,temperature=.5): super().__init__(); self.temperature=temperature
    def forward(self,z1,z2):
        n=z1.shape[0]; z=F.normalize(torch.cat([z1,z2],0),dim=1)
        sim=(z@z.T)/self.temperature
        sim.fill_diagonal_(-torch.inf)
        pos=torch.arange(n,device=z.device)
        target=torch.cat([pos+n,pos])
        return F.cross_entropy(sim,target)

class SupervisedContrastiveLoss(nn.Module):
    """Khosla et al. 2020 SupCon. features N,V,D and labels N."""
    def __init__(self,temperature=.07,base_temperature=.07):
        super().__init__(); self.temperature=temperature; self.base_temperature=base_temperature
    def forward(self,features,labels):
        n,v,d=features.shape
        f=F.normalize(features.reshape(n*v,d),dim=1)
        lab=labels.repeat_interleave(v)
        logits=(f@f.T)/self.temperature
        selfmask=torch.eye(n*v,dtype=torch.bool,device=f.device)
        same=lab[:,None].eq(lab[None,:]) & ~selfmask
        logits=logits.masked_fill(selfmask,-torch.inf)
        logprob=logits-torch.logsumexp(logits,dim=1,keepdim=True)
        count=same.sum(1).clamp_min(1)
        mean_log=(logprob.masked_fill(~same,0).sum(1)/count)
        return -(self.temperature/self.base_temperature)*mean_log.mean()

class CenterLoss(nn.Module):
    """Wen et al. 2016 center loss. Centers are trainable Parameters."""
    def __init__(self,num_classes,feat_dim):
        super().__init__(); self.centers=nn.Parameter(torch.randn(num_classes,feat_dim))
    def forward(self,features,labels):
        return .5*(features-self.centers[labels]).pow(2).sum(1).mean()

class NPairLoss(nn.Module):
    """Sohn 2016 N-pair loss for one positive per anchor."""
    def forward(self,anchors,positives):
        # logits_ij = a_i dot p_j; target i
        return F.cross_entropy(anchors@positives.T,torch.arange(anchors.shape[0],device=anchors.device))

class LiftedStructuredLoss(nn.Module):
    """Song et al. 2016 lifted structured embedding, batch formulation."""
    def __init__(self,margin=1.): super().__init__(); self.margin=margin
    def forward(self,emb,labels):
        d=torch.cdist(emb,emb)
        losses=[]
        n=len(labels)
        for i in range(n):
            for j in range(i+1,n):
                if labels[i]!=labels[j]: continue
                ni=torch.where(labels!=labels[i])[0]
                nj=torch.where(labels!=labels[j])[0]
                if len(ni)==0: continue
                neg=torch.logsumexp(torch.cat([self.margin-d[i,ni],self.margin-d[j,nj]]),0)
                losses.append(.5*F.relu(neg+d[i,j]).pow(2))
        return torch.stack(losses).mean() if losses else emb.sum()*0

class MultiSimilarityLoss(nn.Module):
    """Wang et al. 2019 multi-similarity weighting, using all valid pairs.
       Pair mining threshold epsilon is included.
    """
    def __init__(self,alpha=2.,beta=50.,base=.5,epsilon=.1):
        super().__init__(); self.alpha=alpha; self.beta=beta; self.base=base; self.epsilon=epsilon
    def forward(self,emb,labels):
        s=_cosine_matrix(emb); out=[]
        for i in range(len(labels)):
            pos=s[i][labels==labels[i]]; neg=s[i][labels!=labels[i]]
            pos=pos[pos<1-1e-5]
            if len(pos)==0 or len(neg)==0: continue
            pos=pos[pos < neg.max()+self.epsilon]
            neg=neg[neg > pos.min()-self.epsilon] if len(pos) else neg
            if len(pos)==0 or len(neg)==0: continue
            lp=torch.log1p(torch.exp(-self.alpha*(pos-self.base)).sum())/self.alpha
            ln=torch.log1p(torch.exp(self.beta*(neg-self.base)).sum())/self.beta
            out.append(lp+ln)
        return torch.stack(out).mean() if out else emb.sum()*0

class CircleLoss(nn.Module):
    """Sun et al. 2020 pair-similarity Circle loss."""
    def __init__(self,m=.25,gamma=256.):
        super().__init__(); self.m=m; self.gamma=gamma
    def forward(self,emb,labels):
        s=_cosine_matrix(emb); vals=[]
        op=1+self.m; on=-self.m; dp=1-self.m; dn=self.m
        for i in range(len(labels)):
            pos=s[i][labels==labels[i]]
            pos=pos[pos<1-1e-5]; neg=s[i][labels!=labels[i]]
            if len(pos)==0 or len(neg)==0: continue
            ap=F.relu(op-pos.detach()); an=F.relu(neg.detach()-on)
            lp=-self.gamma*ap*(pos-dp)
            ln=self.gamma*an*(neg-dn)
            vals.append(F.softplus(torch.logsumexp(lp,0)+torch.logsumexp(ln,0)))
        return torch.stack(vals).mean() if vals else emb.sum()*0


# ---------------------------------------------------------------------------
# Angular-margin classification
# ---------------------------------------------------------------------------

class CosFaceLoss(nn.Module):
    """CosFace/LMCL. Inputs are cosine logits from normalized features/weights."""
    def __init__(self,s=64.,m=.35): super().__init__(); self.s=s; self.m=m
    def forward(self,cosine_logits,target):
        one=F.one_hot(target,cosine_logits.shape[1]).to(cosine_logits.dtype)
        logits=self.s*(cosine_logits-one*self.m)
        return F.cross_entropy(logits,target)

class ArcFaceLoss(nn.Module):
    """ArcFace. Inputs are cos(theta) logits from normalized features/weights."""
    def __init__(self,s=64.,m=.5,eps=1e-7): super().__init__(); self.s=s; self.m=m; self.eps=eps
    def forward(self,cosine_logits,target):
        c=cosine_logits.clamp(-1+self.eps,1-self.eps)
        theta=torch.acos(c)
        target_cos=torch.cos(theta+self.m)
        one=F.one_hot(target,c.shape[1]).to(c.dtype)
        logits=self.s*(c*(1-one)+target_cos*one)
        return F.cross_entropy(logits,target)


# ---------------------------------------------------------------------------
# Ranking / ordinal
# ---------------------------------------------------------------------------

class PairwiseLogisticRankingLoss(nn.Module):
    """log(1+exp(-y*(s1-s2))), y in {-1,+1}."""
    def __init__(self,reduction="mean"): super().__init__(); self.reduction=reduction
    def forward(self,s1,s2,y):
        return _reduce(F.softplus(-y.to(s1.dtype)*(s1-s2)),self.reduction)

class ListMLELoss(nn.Module):
    """Plackett-Luce/ListMLE negative log likelihood.
       `target` is relevance; descending relevance defines ground-truth order.
    """
    def forward(self,scores,target):
        order=torch.argsort(target,dim=1,descending=True)
        s=scores.gather(1,order)
        # -sum_i (s_i - logsumexp(s_i...s_n))
        rev=torch.flip(s,dims=[1])
        lse=torch.flip(torch.logcumsumexp(rev,dim=1),dims=[1])
        return (lse-s).sum(1).mean()

class CORALLoss(nn.Module):
    """CORAL ordinal regression BCE over K-1 cumulative thresholds.
       logits: N,K-1; target: integer N in [0,K-1].
    """
    def forward(self,logits,target):
        levels=torch.arange(logits.shape[1],device=logits.device)[None,:]
        y=(target[:,None]>levels).to(logits.dtype)
        return F.binary_cross_entropy_with_logits(logits,y)


# ---------------------------------------------------------------------------
# GAN objectives
# ---------------------------------------------------------------------------

class LSGANDiscriminatorLoss(nn.Module):
    """Least Squares GAN D loss: 1/2 E[(D(real)-b)^2 + (D(fake)-a)^2]."""
    def __init__(self,real_label=1.,fake_label=0.): super().__init__(); self.r=real_label; self.f=fake_label
    def forward(self,real_scores,fake_scores):
        return .5*((real_scores-self.r).pow(2).mean()+(fake_scores-self.f).pow(2).mean())

class LSGANGeneratorLoss(nn.Module):
    def __init__(self,target_label=1.): super().__init__(); self.t=target_label
    def forward(self,fake_scores): return .5*(fake_scores-self.t).pow(2).mean()

class HingeDiscriminatorLoss(nn.Module):
    def forward(self,real_scores,fake_scores):
        return F.relu(1-real_scores).mean()+F.relu(1+fake_scores).mean()

class HingeGeneratorLoss(nn.Module):
    def forward(self,fake_scores): return -fake_scores.mean()

class WassersteinCriticLoss(nn.Module):
    """WGAN critic minimization objective E[D(fake)]-E[D(real)]."""
    def forward(self,real_scores,fake_scores): return fake_scores.mean()-real_scores.mean()

class WassersteinGeneratorLoss(nn.Module):
    def forward(self,fake_scores): return -fake_scores.mean()

class RelativisticAverageDiscriminatorLoss(nn.Module):
    """RaGAN logistic discriminator objective."""
    def forward(self,real_scores,fake_scores):
        real_rel=real_scores-fake_scores.mean()
        fake_rel=fake_scores-real_scores.mean()
        return .5*(F.binary_cross_entropy_with_logits(real_rel,torch.ones_like(real_rel))+
                   F.binary_cross_entropy_with_logits(fake_rel,torch.zeros_like(fake_rel)))

class RelativisticAverageGeneratorLoss(nn.Module):
    def forward(self,real_scores,fake_scores):
        real_rel=real_scores-fake_scores.mean()
        fake_rel=fake_scores-real_scores.mean()
        return .5*(F.binary_cross_entropy_with_logits(real_rel,torch.zeros_like(real_rel))+
                   F.binary_cross_entropy_with_logits(fake_rel,torch.ones_like(fake_rel)))



# ---------------------------------------------------------------------------
# Additional segmentation / regression / SSL objectives
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    """Binary BCE + soft Dice loss."""
    def __init__(self,bce_weight=.5,dice_weight=.5,smooth=1e-6):
        super().__init__(); self.bw=bce_weight; self.dw=dice_weight; self.dice=DiceLoss(smooth=smooth)
    def forward(self,logits,target):
        return self.bw*F.binary_cross_entropy_with_logits(logits,target.to(logits.dtype))+self.dw*self.dice(logits,target)

class FocalDiceLoss(nn.Module):
    """Weighted sum of binary focal loss and soft Dice."""
    def __init__(self,focal_weight=.5,dice_weight=.5,alpha=.25,gamma=2.,smooth=1e-6):
        super().__init__(); self.fw=focal_weight; self.dw=dice_weight
        self.focal=BinaryFocalLoss(alpha,gamma); self.dice=DiceLoss(smooth=smooth)
    def forward(self,logits,target): return self.fw*self.focal(logits,target)+self.dw*self.dice(logits,target)

class SensitivitySpecificityLoss(nn.Module):
    """Squared-error sensitivity/specificity objective used in segmentation:
       lambda * mean(error^2 | foreground) + (1-lambda)*mean(error^2 | background).
    """
    def __init__(self,lam=.5,from_logits=True,eps=1e-7):
        super().__init__(); self.lam=lam; self.from_logits=from_logits; self.eps=eps
    def forward(self,pred,target):
        p=torch.sigmoid(pred) if self.from_logits else pred; y=target.to(p.dtype)
        e=(p-y).pow(2)
        fg=(e*y).sum()/y.sum().clamp_min(self.eps)
        bg=(e*(1-y)).sum()/(1-y).sum().clamp_min(self.eps)
        return self.lam*fg+(1-self.lam)*bg

class MCCLoss(nn.Module):
    """1 - differentiable Matthews correlation coefficient for binary masks."""
    def __init__(self,from_logits=True,eps=1e-7):
        super().__init__(); self.from_logits=from_logits; self.eps=eps
    def forward(self,pred,target):
        p=torch.sigmoid(pred) if self.from_logits else pred; y=target.to(p.dtype)
        tp=(p*y).sum(); tn=((1-p)*(1-y)).sum(); fp=(p*(1-y)).sum(); fn=((1-p)*y).sum()
        den=torch.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)).clamp_min(self.eps)
        return 1-(tp*tn-fp*fn)/den

class MAPE(nn.Module):
    def __init__(self,eps=1e-7,reduction="mean"): super().__init__(); self.eps=eps; self.reduction=reduction
    def forward(self,pred,target): return _reduce((pred-target).abs()/target.abs().clamp_min(self.eps),self.reduction)

class SMAPE(nn.Module):
    def __init__(self,eps=1e-7,reduction="mean"): super().__init__(); self.eps=eps; self.reduction=reduction
    def forward(self,pred,target):
        return _reduce(2*(pred-target).abs()/(pred.abs()+target.abs()+self.eps),self.reduction)

class MSLELoss(nn.Module):
    """Mean squared logarithmic error; requires pred,target > -1."""
    def forward(self,pred,target): return (torch.log1p(pred)-torch.log1p(target)).pow(2).mean()

class RMSLELoss(nn.Module):
    def __init__(self,eps=1e-12): super().__init__(); self.eps=eps
    def forward(self,pred,target): return torch.sqrt((torch.log1p(pred)-torch.log1p(target)).pow(2).mean()+self.eps)

class RelativeL1Loss(nn.Module):
    def __init__(self,eps=1e-7,reduction="mean"): super().__init__(); self.eps=eps; self.reduction=reduction
    def forward(self,pred,target): return _reduce((pred-target).abs()/(target.abs()+self.eps),self.reduction)

class RelativeL2Loss(nn.Module):
    def __init__(self,eps=1e-7,reduction="mean"): super().__init__(); self.eps=eps; self.reduction=reduction
    def forward(self,pred,target): return _reduce((pred-target).pow(2)/(target.pow(2)+self.eps),self.reduction)

class BarlowTwinsLoss(nn.Module):
    """Zbontar et al. 2021. Cross-correlation is computed after batch standardization."""
    def __init__(self,lambd=5e-3,eps=1e-5): super().__init__(); self.lambd=lambd; self.eps=eps
    def forward(self,z1,z2):
        z1=(z1-z1.mean(0))/(z1.std(0,unbiased=False)+self.eps)
        z2=(z2-z2.mean(0))/(z2.std(0,unbiased=False)+self.eps)
        c=z1.T@z2/z1.shape[0]
        on=(torch.diagonal(c)-1).pow(2).sum()
        off=(c-torch.diag(torch.diagonal(c))).pow(2).sum()
        return on+self.lambd*off

class VICRegLoss(nn.Module):
    """Bardes et al. 2022 VICReg = invariance + variance + covariance."""
    def __init__(self,sim_coeff=25.,std_coeff=25.,cov_coeff=1.,gamma=1.,eps=1e-4):
        super().__init__(); self.sim=sim_coeff; self.std=std_coeff; self.cov=cov_coeff; self.gamma=gamma; self.eps=eps
    @staticmethod
    def _offdiag(x):
        n=x.shape[0]
        return x.flatten()[:-1].view(n-1,n+1)[:,1:].flatten()
    def forward(self,x,y):
        repr_loss=F.mse_loss(x,y)
        xc=x-x.mean(0); yc=y-y.mean(0)
        if x.shape[0] < 2:
            # The reference VICReg variance/covariance estimator uses N-1.
            # A single sample has no sample variance; keep the objective finite.
            sx=torch.sqrt(torch.full_like(x.mean(0), self.eps))
            sy=torch.sqrt(torch.full_like(y.mean(0), self.eps))
        else:
            sx=torch.sqrt(x.var(0,unbiased=True)+self.eps)
            sy=torch.sqrt(y.var(0,unbiased=True)+self.eps)
        std_loss=.5*(F.relu(self.gamma-sx).mean()+F.relu(self.gamma-sy).mean())
        n=x.shape[0]
        if n>1:
            covx=(xc.T@xc)/(n-1); covy=(yc.T@yc)/(n-1)
            cov_loss=(self._offdiag(covx).pow(2).sum()+self._offdiag(covy).pow(2).sum())/(2*x.shape[1])
        else:
            cov_loss=x.sum()*0
        return self.sim*repr_loss+self.std*std_loss+self.cov*cov_loss

class NegativeCosineSimilarityLoss(nn.Module):
    """Common BYOL/SimSiam-style negative cosine objective."""
    def forward(self,p,z):
        return -F.cosine_similarity(p,z.detach(),dim=-1).mean()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LOSS_REGISTRY = {
    name: obj for name,obj in globals().copy().items()
    if isinstance(obj,type) and issubclass(obj,nn.Module) and obj is not nn.Module
}

CORE_PYTORCH_EXCLUSIONS = [
    "L1Loss","MSELoss","CrossEntropyLoss","CTCLoss","NLLLoss","PoissonNLLLoss",
    "GaussianNLLLoss","KLDivLoss","BCELoss","BCEWithLogitsLoss","MarginRankingLoss",
    "HingeEmbeddingLoss","MultiLabelMarginLoss","HuberLoss","SmoothL1Loss",
    "SoftMarginLoss","MultiLabelSoftMarginLoss","CosineEmbeddingLoss",
    "MultiMarginLoss","TripletMarginLoss","TripletMarginWithDistanceLoss"
]
