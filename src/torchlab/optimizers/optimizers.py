"""
missing_optimizers.py

Research optimizers that are not currently exposed as dedicated algorithms in
core torch.optim (PyTorch 2.13 docs checked August 2026).

Core torch.optim currently includes:
Adadelta, Adafactor, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, LBFGS,
Muon, NAdam, RAdam, RMSprop, Rprop, and SGD.

This module focuses on additional published optimizers and optimizer wrappers.
All implementations subclass torch.optim.Optimizer and support parameter groups.

Mathematical conventions are documented per class. The accompanying test suite
checks one-step equations, special-case identities, state recurrences, and
quadratic convergence smoke tests.

This is research-oriented reference code, not fused/foreach production kernels.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Sequence, Type

import torch
from torch import Tensor
from torch.optim import Optimizer


def _validate_betas(betas):
    if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
        raise ValueError("betas must be in [0,1)")

def _closure_loss(closure):
    if closure is None:
        return None
    with torch.enable_grad():
        return closure()

def _norm(x: Tensor) -> Tensor:
    return torch.linalg.vector_norm(x)

def _matrix_power_sym(a: Tensor, power: float, eps: float = 1e-12) -> Tensor:
    # symmetric PSD fractional matrix power
    vals, vecs = torch.linalg.eigh(a)
    vals = vals.clamp_min(eps).pow(power)
    return (vecs * vals.unsqueeze(0)) @ vecs.T


class Lion(Optimizer):
    """Lion: Symbolic Discovery of Optimization Algorithms (Chen et al. 2023).

    c_t = beta1*m_{t-1} + (1-beta1)*g_t
    theta_t = (1-lr*wd)*theta_{t-1} - lr*sign(c_t)
    m_t = beta2*m_{t-1} + (1-beta2)*g_t
    """
    def __init__(self, params, lr=1e-4, betas=(0.9,0.99), weight_decay=0.0):
        _validate_betas(betas)
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self, closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            lr,b1,b2,wd=group["lr"],*group["betas"],group["weight_decay"]
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad
                st=self.state[p]
                if not st: st["m"]=torch.zeros_like(p)
                m=st["m"]
                update=m.mul(b1).add(g,alpha=1-b1)
                if wd: p.mul_(1-lr*wd)
                p.add_(update.sign(),alpha=-lr)
                m.mul_(b2).add_(g,alpha=1-b2)
        return loss


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling (You, Gitman, Ginsburg 2017).

    Uses local_lr = trust * ||w|| / (||g|| + wd*||w|| + eps), then momentum SGD.
    Weight decay is coupled as in the original LARS formulation.
    """
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0.0,
                 trust_coefficient=0.001, eps=1e-8, nesterov=False):
        super().__init__(params, dict(lr=lr,momentum=momentum,weight_decay=weight_decay,
                                     trust_coefficient=trust_coefficient,eps=eps,nesterov=nesterov))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad
                wd=group["weight_decay"]
                wnorm=_norm(p); gnorm=_norm(g)
                if wnorm>0 and gnorm>0:
                    local=group["trust_coefficient"]*wnorm/(gnorm+wd*wnorm+group["eps"])
                else:
                    local=torch.ones((),device=p.device,dtype=p.dtype)
                d=g.add(p,alpha=wd) if wd else g.clone()
                d.mul_(local)
                mu=group["momentum"]
                if mu:
                    st=self.state[p]
                    if "buf" not in st: st["buf"]=torch.zeros_like(p)
                    b=st["buf"]; b.mul_(mu).add_(d)
                    d=d.add(b,alpha=mu) if group["nesterov"] else b
                p.add_(d,alpha=-group["lr"])
        return loss


class LAMB(Optimizer):
    """LAMB v3 (You et al. 2020), layer-wise adaptive large-batch optimizer.

    m_t = beta1*m + (1-beta1)g
    v_t = beta2*v + (1-beta2)g^2
    r_t = m_t/(sqrt(v_t)+eps) + wd*w
    phi(||w||) = clamp(||w||, max=clamp_value)
    trust = phi(||w||)/||r_t||, with trust=1 for zero norms.

    The published v3/reference implementation omits Adam bias correction by
    default. ``bias_correction=True`` is exposed for the commonly used variant.
    """
    def __init__(self,params,lr=1e-3,betas=(.9,.999),eps=1e-6,weight_decay=0.0,
                 clamp_value=10.0,bias_correction=False):
        _validate_betas(betas)
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,
                                     clamp_value=clamp_value,bias_correction=bias_correction))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            b1,b2=group["betas"]
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad; st=self.state[p]
                if not st: st["step"]=0; st["m"]=torch.zeros_like(p); st["v"]=torch.zeros_like(p)
                st["step"]+=1; t=st["step"]; m,v=st["m"],st["v"]
                m.mul_(b1).add_(g,alpha=1-b1); v.mul_(b2).addcmul_(g,g,value=1-b2)
                if group["bias_correction"]:
                    mh=m/(1-b1**t); vh=v/(1-b2**t)
                else:
                    mh=m; vh=v
                r=mh/(vh.sqrt()+group["eps"])
                if group["weight_decay"]: r=r.add(p,alpha=group["weight_decay"])
                wn=_norm(p).clamp(max=group["clamp_value"]); rn=_norm(r)
                trust=wn/rn if wn>0 and rn>0 else torch.ones_like(wn)
                p.add_(r,alpha=-group["lr"]*float(trust))
        return loss


class AdaBelief(Optimizer):
    """AdaBelief (Zhuang et al. 2020).
    m_t = beta1*m + (1-beta1)g
    s_t = beta2*s + (1-beta2)(g-m_t)^2 + eps
    Adam-like bias correction. Decoupled weight decay by default.
    """
    def __init__(self,params,lr=1e-3,betas=(.9,.999),eps=1e-16,weight_decay=0.,
                 decoupled_weight_decay=True):
        _validate_betas(betas)
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,
                                     decoupled_weight_decay=decoupled_weight_decay))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            b1,b2=group["betas"]
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad
                if group["weight_decay"] and not group["decoupled_weight_decay"]:
                    g=g.add(p,alpha=group["weight_decay"])
                st=self.state[p]
                if not st: st["step"]=0; st["m"]=torch.zeros_like(p); st["s"]=torch.zeros_like(p)
                st["step"]+=1; t=st["step"]; m,s=st["m"],st["s"]
                m.mul_(b1).add_(g,alpha=1-b1)
                residual=g-m
                s.mul_(b2).addcmul_(residual,residual,value=1-b2).add_(group["eps"])
                mh=m/(1-b1**t); sh=s/(1-b2**t)
                if group["weight_decay"] and group["decoupled_weight_decay"]:
                    p.mul_(1-group["lr"]*group["weight_decay"])
                p.addcdiv_(mh,sh.sqrt().add(group["eps"]),value=-group["lr"])
        return loss


class AdaBound(Optimizer):
    """AdaBound (Luo et al. 2019), practical Adam-style implementation.

    Elementwise Adam step sizes are dynamically clipped:
      lower = final_lr*(1 - 1/(gamma*t+1))
      upper = final_lr*(1 + 1/(gamma*t))
    after accounting for the base/final LR ratio.
    """
    def __init__(self,params,lr=1e-3,betas=(.9,.999),final_lr=.1,gamma=1e-3,
                 eps=1e-8,weight_decay=0.,amsbound=False):
        _validate_betas(betas)
        super().__init__(params,dict(lr=lr,betas=betas,final_lr=final_lr,gamma=gamma,
                                     eps=eps,weight_decay=weight_decay,amsbound=amsbound,base_lr=lr))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            b1,b2=group["betas"]
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad
                if group["weight_decay"]: g=g.add(p,alpha=group["weight_decay"])
                st=self.state[p]
                if not st:
                    st["step"]=0; st["m"]=torch.zeros_like(p); st["v"]=torch.zeros_like(p)
                    if group["amsbound"]: st["vmax"]=torch.zeros_like(p)
                st["step"]+=1; t=st["step"]; m,v=st["m"],st["v"]
                m.mul_(b1).add_(g,alpha=1-b1); v.mul_(b2).addcmul_(g,g,value=1-b2)
                vv=v
                if group["amsbound"]:
                    torch.maximum(st["vmax"],v,out=st["vmax"]); vv=st["vmax"]
                step_size=group["lr"]*math.sqrt(1-b2**t)/(1-b1**t)
                final_lr=group["final_lr"]*group["lr"]/group["base_lr"]
                lo=final_lr*(1-1/(group["gamma"]*t+1))
                hi=final_lr*(1+1/(group["gamma"]*t))
                bounded=torch.full_like(vv,step_size).div_(vv.sqrt().add(group["eps"])).clamp_(lo,hi)
                p.addcmul_(m,bounded,value=-1)
        return loss

class AMSBound(AdaBound):
    def __init__(self,*args,**kwargs):
        kwargs["amsbound"]=True
        super().__init__(*args,**kwargs)


class Yogi(Optimizer):
    """Yogi (Zaheer et al. 2018).
    v_t = v_{t-1} - (1-beta2)*sign(v_{t-1}-g^2)*g^2.
    """
    def __init__(self,params,lr=1e-2,betas=(.9,.999),eps=1e-3,weight_decay=0.):
        _validate_betas(betas)
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            b1,b2=group["betas"]
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad
                if group["weight_decay"]: g=g.add(p,alpha=group["weight_decay"])
                st=self.state[p]
                if not st: st["step"]=0; st["m"]=torch.zeros_like(p); st["v"]=torch.zeros_like(p)
                st["step"]+=1; t=st["step"]; m,v=st["m"],st["v"]
                m.mul_(b1).add_(g,alpha=1-b1)
                g2=g*g
                v.addcmul_(torch.sign(v-g2),g2,value=-(1-b2))
                mh=m/(1-b1**t); vh=v/(1-b2**t)
                p.addcdiv_(mh,vh.sqrt().add(group["eps"]),value=-group["lr"])
        return loss


class QHM(Optimizer):
    """Quasi-Hyperbolic Momentum (Ma & Yarats 2018).
    m = beta*m + (1-beta)*g
    update = (1-nu)*g + nu*m
    """
    def __init__(self,params,lr=1e-2,momentum=.9,nu=1.,weight_decay=0.):
        super().__init__(params,dict(lr=lr,momentum=momentum,nu=nu,weight_decay=weight_decay))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad
                if group["weight_decay"]: g=g.add(p,alpha=group["weight_decay"])
                st=self.state[p]
                if "m" not in st: st["m"]=torch.zeros_like(p)
                m=st["m"]; b=group["momentum"]; nu=group["nu"]
                m.mul_(b).add_(g,alpha=1-b)
                d=g.mul(1-nu).add(m,alpha=nu)
                p.add_(d,alpha=-group["lr"])
        return loss


class QHAdam(Optimizer):
    """QHAdam (Ma & Yarats 2018).
    numerator=(1-nu1)g + nu1*mhat
    denominator=sqrt((1-nu2)g^2 + nu2*vhat)+eps
    """
    def __init__(self,params,lr=1e-3,betas=(.9,.999),nus=(1.,1.),eps=1e-8,weight_decay=0.):
        _validate_betas(betas)
        super().__init__(params,dict(lr=lr,betas=betas,nus=nus,eps=eps,weight_decay=weight_decay))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            b1,b2=group["betas"]; n1,n2=group["nus"]
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad
                if group["weight_decay"]: g=g.add(p,alpha=group["weight_decay"])
                st=self.state[p]
                if not st: st["step"]=0; st["m"]=torch.zeros_like(p); st["v"]=torch.zeros_like(p)
                st["step"]+=1; t=st["step"]; m,v=st["m"],st["v"]
                m.mul_(b1).add_(g,alpha=1-b1); v.mul_(b2).addcmul_(g,g,value=1-b2)
                mh=m/(1-b1**t); vh=v/(1-b2**t)
                num=g.mul(1-n1).add(mh,alpha=n1)
                den=(g*g*(1-n2)+vh*n2).sqrt().add(group["eps"])
                p.addcdiv_(num,den,value=-group["lr"])
        return loss


class NovoGrad(Optimizer):
    """NovoGrad (Ginsburg et al. 2019), layer-wise second moment.

    v_t = beta2*v + (1-beta2)*||g||^2
    g_norm = g/(sqrt(v_t)+eps)
    m_t = beta1*m + (1-beta1)*(g_norm + wd*w)
    """
    def __init__(self,params,lr=1e-3,betas=(.95,.98),eps=1e-8,weight_decay=0.,
                 grad_averaging=False):
        _validate_betas(betas)
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,
                                     grad_averaging=grad_averaging))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            b1,b2=group["betas"]
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad; st=self.state[p]; g2=g.pow(2).sum()
                if not st: st["v"]=g2.clone(); st["m"]=torch.zeros_like(p)
                else: st["v"].mul_(b2).add_(g2,alpha=1-b2)
                d=g/(st["v"].sqrt()+group["eps"])
                if group["weight_decay"]: d=d.add(p,alpha=group["weight_decay"])
                if group["grad_averaging"]: d=d.mul(1-b1)
                st["m"].mul_(b1).add_(d)
                p.add_(st["m"],alpha=-group["lr"])
        return loss


class AdaMod(Optimizer):
    """AdaMod (Ding et al. 2019): Adam with EMA upper bound on adaptive LR."""
    def __init__(self,params,lr=1e-3,betas=(.9,.999),beta3=.999,eps=1e-8,weight_decay=0.):
        _validate_betas(betas)
        super().__init__(params,dict(lr=lr,betas=betas,beta3=beta3,eps=eps,weight_decay=weight_decay))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            b1,b2=group["betas"]; b3=group["beta3"]
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad
                if group["weight_decay"]: g=g.add(p,alpha=group["weight_decay"])
                st=self.state[p]
                if not st: st["step"]=0; st["m"]=torch.zeros_like(p); st["v"]=torch.zeros_like(p); st["s"]=torch.zeros_like(p)
                st["step"]+=1; t=st["step"]
                st["m"].mul_(b1).add_(g,alpha=1-b1); st["v"].mul_(b2).addcmul_(g,g,value=1-b2)
                step=group["lr"]*math.sqrt(1-b2**t)/(1-b1**t)
                eta=torch.full_like(p,step).div_(st["v"].sqrt().add(group["eps"]))
                st["s"].mul_(b3).add_(eta,alpha=1-b3)
                eta=torch.minimum(eta,st["s"])
                p.addcmul_(st["m"],eta,value=-1)
        return loss


class DiffGrad(Optimizer):
    """diffGrad (Dubey et al. 2019).
    Adam first/second moments; update multiplied by sigmoid(|g_t-g_{t-1}|).
    """
    def __init__(self,params,lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0.):
        _validate_betas(betas)
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            b1,b2=group["betas"]
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad
                if group["weight_decay"]: g=g.add(p,alpha=group["weight_decay"])
                st=self.state[p]
                if not st:
                    st["step"]=0; st["m"]=torch.zeros_like(p); st["v"]=torch.zeros_like(p); st["prev"]=torch.zeros_like(p)
                st["step"]+=1; t=st["step"]
                st["m"].mul_(b1).add_(g,alpha=1-b1); st["v"].mul_(b2).addcmul_(g,g,value=1-b2)
                dfc=torch.sigmoid((st["prev"]-g).abs())
                st["prev"].copy_(g)
                mh=st["m"]/(1-b1**t); vh=st["v"]/(1-b2**t)
                p.addcdiv_(mh*dfc,vh.sqrt().add(group["eps"]),value=-group["lr"])
        return loss


class AdaNorm(Optimizer):
    """AdaNorm gradient-norm correction applied to Adam (Dubey et al. 2022).

    e_t = gamma*e_{t-1} + (1-gamma)||g_t||
    scaled g = g * e_t/||g|| if e_t > ||g||, else g.
    Moments are updated from the scaled gradient.
    """
    def __init__(self,params,lr=1e-3,betas=(.9,.999),gamma=.95,eps=1e-8,weight_decay=0.):
        _validate_betas(betas)
        super().__init__(params,dict(lr=lr,betas=betas,gamma=gamma,eps=eps,weight_decay=weight_decay))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            b1,b2=group["betas"]
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad
                if group["weight_decay"]: g=g.add(p,alpha=group["weight_decay"])
                st=self.state[p]
                if not st: st["step"]=0; st["m"]=torch.zeros_like(p); st["v"]=torch.zeros_like(p); st["e"]=torch.zeros((),device=p.device,dtype=p.dtype)
                st["step"]+=1; t=st["step"]; gn=_norm(g)
                st["e"].mul_(group["gamma"]).add_(gn,alpha=1-group["gamma"])
                gs=g*(st["e"]/gn) if gn>0 and st["e"]>gn else g
                st["m"].mul_(b1).add_(gs,alpha=1-b1)
                # AdaNorm corrects the first-moment gradient only; v_t uses raw g_t^2.
                st["v"].mul_(b2).addcmul_(g,g,value=1-b2)
                mh=st["m"]/(1-b1**t); vh=st["v"]/(1-b2**t)
                p.addcdiv_(mh,vh.sqrt().add(group["eps"]),value=-group["lr"])
        return loss


class AggMo(Optimizer):
    """Aggregated Momentum (Lucas et al. 2018).
    Multiple velocity buffers with different decay rates; updates are averaged.
    """
    def __init__(self,params,lr=1e-2,betas=(0.,.9,.99),weight_decay=0.):
        super().__init__(params,dict(lr=lr,betas=tuple(betas),weight_decay=weight_decay))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            K=len(group["betas"])
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad
                if group["weight_decay"]: g=g.add(p,alpha=group["weight_decay"])
                st=self.state[p]
                if "bufs" not in st: st["bufs"]=[torch.zeros_like(p) for _ in group["betas"]]
                d=torch.zeros_like(p)
                for b,buf in zip(group["betas"],st["bufs"]):
                    buf.mul_(b).add_(g)
                    d.add_(buf,alpha=1/K)
                p.add_(d,alpha=-group["lr"])
        return loss


class Shampoo(Optimizer):
    """Shampoo (Gupta et al. 2018), matrix version for tensors flattened to 2D.

    L_t += G G^T, R_t += G^T G
    update = L^{-1/4} G R^{-1/4}.
    Vectors fall back to Adagrad-style preconditioning.
    """
    def __init__(self,params,lr=1e-2,eps=1e-4,weight_decay=0.,update_freq=1):
        super().__init__(params,dict(lr=lr,eps=eps,weight_decay=weight_decay,update_freq=update_freq))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad
                if group["weight_decay"]: g=g.add(p,alpha=group["weight_decay"])
                st=self.state[p]; st["step"]=st.get("step",0)+1
                if p.ndim<2:
                    if "v" not in st: st["v"]=torch.zeros_like(p)
                    st["v"].addcmul_(g,g)
                    d=g/(st["v"].sqrt()+group["eps"])
                else:
                    G=g.reshape(g.shape[0],-1); m,n=G.shape
                    if "L" not in st:
                        st["L"]=torch.eye(m,device=p.device,dtype=p.dtype)*group["eps"]
                        st["R"]=torch.eye(n,device=p.device,dtype=p.dtype)*group["eps"]
                        st["Li"]=torch.eye(m,device=p.device,dtype=p.dtype)
                        st["Ri"]=torch.eye(n,device=p.device,dtype=p.dtype)
                    st["L"].add_(G@G.T); st["R"].add_(G.T@G)
                    if st["step"]%group["update_freq"]==0:
                        st["Li"]=_matrix_power_sym(st["L"],-.25,group["eps"])
                        st["Ri"]=_matrix_power_sym(st["R"],-.25,group["eps"])
                    d=(st["Li"]@G@st["Ri"]).reshape_as(p)
                p.add_(d,alpha=-group["lr"])
        return loss


class Fromage(Optimizer):
    """Fromage (Bernstein et al. 2020).
    For each parameter tensor:
      d = g * ||w||/(||g||+eps)
      w <- (w - lr*d)/sqrt(1+lr^2)
    """
    def __init__(self,params,lr=1e-2,eps=1e-12):
        super().__init__(params,dict(lr=lr,eps=eps))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            lr=group["lr"]
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad; wn=_norm(p); gn=_norm(g)
                d=g*(wn/(gn+group["eps"])) if wn>0 and gn>0 else g
                p.add_(d,alpha=-lr).div_(math.sqrt(1+lr*lr))
        return loss


class MADGRAD(Optimizer):
    """MADGRAD (Defazio & Jelassi 2021), no momentum variant by default.

    s += lambda*g; v += lambda*g^2; z = x0 - s/(v^(1/3)+eps).
    With momentum, x <- (1-m)*x + m*z using paper/library convention.
    """
    def __init__(self,params,lr=1e-2,momentum=0.,weight_decay=0.,eps=1e-6):
        super().__init__(params,dict(lr=lr,momentum=momentum,weight_decay=weight_decay,eps=eps))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad
                if group["weight_decay"]: g=g.add(p,alpha=group["weight_decay"])
                st=self.state[p]
                if not st:
                    st["step"]=0; st["s"]=torch.zeros_like(p); st["v"]=torch.zeros_like(p); st["x0"]=p.detach().clone()
                st["step"]+=1
                lam=group["lr"]*math.sqrt(st["step"])
                st["s"].add_(g,alpha=lam); st["v"].addcmul_(g,g,value=lam)
                z=st["x0"]-st["s"]/(st["v"].pow(1/3)+group["eps"])
                mu=group["momentum"]
                if mu: p.mul_(1-mu).add_(z,alpha=mu)
                else: p.copy_(z)
        return loss


class Adan(Optimizer):
    """Adan (Xie et al. 2022).
    m = beta1*m + (1-beta1)g
    v = beta2*v + (1-beta2)(g-g_prev)
    n = beta3*n + (1-beta3)(g + beta2*(g-g_prev))^2
    update uses bias-corrected (m + beta2*v)/(sqrt(n)+eps).
    """
    def __init__(self,params,lr=1e-3,betas=(.98,.92,.99),eps=1e-8,weight_decay=0.):
        if any(not 0<=b<1 for b in betas): raise ValueError
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            b1,b2,b3=group["betas"]
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad; st=self.state[p]
                if not st:
                    st["step"]=0; st["m"]=torch.zeros_like(p); st["v"]=torch.zeros_like(p)
                    st["n"]=torch.zeros_like(p); st["prev"]=g.detach().clone()
                st["step"]+=1; t=st["step"]; diff=g-st["prev"]
                st["m"].mul_(b1).add_(g,alpha=1-b1)
                st["v"].mul_(b2).add_(diff,alpha=1-b2)
                q=g+b2*diff
                st["n"].mul_(b3).addcmul_(q,q,value=1-b3)
                mh=st["m"]/(1-b1**t); vh=st["v"]/(1-b2**t); nh=st["n"]/(1-b3**t)
                if group["weight_decay"]: p.div_(1+group["lr"]*group["weight_decay"])
                p.addcdiv_(mh+b2*vh,nh.sqrt().add(group["eps"]),value=-group["lr"])
                st["prev"].copy_(g)
        return loss



class AdaHessian(Optimizer):
    """AdaHessian (Yao et al. 2021).

    Uses Hutchinson's estimator of the diagonal Hessian:
      h_t = z_t * (H_t z_t), z_t in {-1,+1}^d
    and an EMA of h_t^2 as the second-order preconditioner.

    Important: gradients must be created with ``create_graph=True`` so the
    Hessian-vector product is differentiable. Example::

        loss.backward(create_graph=True)
        optimizer.step()

    ``n_samples`` averages independent Rademacher probes. This reference
    implementation estimates the Hessian on every call to ``step``.
    """
    def __init__(self, params, lr=.1, betas=(.9,.999), eps=1e-4,
                 weight_decay=0., hessian_power=1., n_samples=1):
        _validate_betas(betas)
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        super().__init__(
            params,
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                 hessian_power=hessian_power, n_samples=int(n_samples))
        )

    def _hutchinson_diagonal(self):
        params = []
        grads = []
        samples = None
        for group in self.param_groups:
            samples = group["n_samples"] if samples is None else samples
            for p in group["params"]:
                if p.grad is None:
                    continue
                if not p.grad.requires_grad:
                    raise RuntimeError(
                        "AdaHessian requires gradients computed with "
                        "loss.backward(create_graph=True)"
                    )
                params.append(p)
                grads.append(p.grad)

        if not params:
            return {}

        out = {p: torch.zeros_like(p) for p in params}
        # Parameter groups can technically request different n_samples; perform
        # the maximum count and only accumulate for active groups.
        max_samples = max(g["n_samples"] for g in self.param_groups)
        group_of = {p: g for g in self.param_groups for p in g["params"]}

        for sample_idx in range(max_samples):
            active = [p for p in params if sample_idx < group_of[p]["n_samples"]]
            if not active:
                continue
            active_grads = [p.grad for p in active]
            zs = [
                torch.empty_like(p).bernoulli_(.5).mul_(2).sub_(1)
                for p in active
            ]
            hvps = torch.autograd.grad(
                active_grads,
                active,
                grad_outputs=zs,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )
            for p, z, hv in zip(active, zs, hvps):
                out[p].add_(hv * z, alpha=1.0 / group_of[p]["n_samples"])
        return out

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        hdiag = self._hutchinson_diagonal()

        with torch.no_grad():
            for group in self.param_groups:
                b1, b2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad.detach()
                    h = hdiag[p].detach()
                    st = self.state[p]
                    if not st:
                        st["step"] = 0
                        st["m"] = torch.zeros_like(p)
                        st["h2"] = torch.zeros_like(p)

                    st["step"] += 1
                    t = st["step"]
                    st["m"].mul_(b1).add_(g, alpha=1-b1)
                    st["h2"].mul_(b2).addcmul_(h, h, value=1-b2)

                    mhat = st["m"] / (1-b1**t)
                    h2hat = st["h2"] / (1-b2**t)
                    denom = h2hat.pow(group["hessian_power"]/2).add(group["eps"])

                    if group["weight_decay"]:
                        p.mul_(1-group["lr"]*group["weight_decay"])
                    p.addcdiv_(mhat, denom, value=-group["lr"])

        return loss


class SophiaG(Optimizer):
    """Sophia-G (Liu et al. 2023), diagonal Hessian estimator supplied externally.

    `update_hessian()` should be called on Hessian-estimator steps after setting
    p.grad to an unbiased diagonal-Hessian estimate (e.g. Gauss-Newton-Bartlett).
    Ordinary `step()` uses current first moment and stored h:
      p -= lr * clamp(m / max(rho*h, eps), -1, 1)
    with decoupled weight decay.
    """
    def __init__(self,params,lr=1e-4,betas=(.965,.99),rho=.04,weight_decay=.1,eps=1e-12):
        _validate_betas(betas)
        super().__init__(params,dict(lr=lr,betas=betas,rho=rho,weight_decay=weight_decay,eps=eps))
    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            b2=group["betas"][1]
            for p in group["params"]:
                if p.grad is None: continue
                st=self.state[p]
                if "h" not in st: st["h"]=torch.zeros_like(p)
                st["h"].mul_(b2).add_(p.grad,alpha=1-b2)
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            b1,_=group["betas"]
            for p in group["params"]:
                if p.grad is None: continue
                st=self.state[p]
                if "m" not in st: st["m"]=torch.zeros_like(p)
                if "h" not in st: st["h"]=torch.zeros_like(p)
                st["m"].mul_(b1).add_(p.grad,alpha=1-b1)
                if group["weight_decay"]: p.mul_(1-group["lr"]*group["weight_decay"])
                denom=torch.maximum(group["rho"]*st["h"], torch.full_like(st["h"],group["eps"]))
                d=(st["m"]/denom).clamp(-1,1)
                p.add_(d,alpha=-group["lr"])
        return loss


class PNM(Optimizer):
    """Positive-Negative Momentum (Xie et al. 2021), SGD variant.

    The two interleaved momenta implement the paper recursion
        m_t = beta1^2 m_{t-2} + (1-beta1^2) g_t
    and the update
        u_t = ((1+beta0)m_t - beta0 m_{t-1}) /
              sqrt((1+beta0)^2 + beta0^2).

    Weight decay is coupled to the gradient, matching classical SGD-style PNM.
    """
    def __init__(self,params,lr=1e-2,betas=(.9,1.0),weight_decay=0.):
        beta1,beta0=betas
        if not 0 <= beta1 < 1 or beta0 < 0: raise ValueError("invalid PNM betas")
        super().__init__(params,dict(lr=lr,betas=betas,weight_decay=weight_decay))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            beta1,beta0=group["betas"]
            beta1_sq=beta1*beta1
            noise_norm=math.sqrt((1+beta0)**2+beta0**2)
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad
                if group["weight_decay"]: g=g.add(p,alpha=group["weight_decay"])
                st=self.state[p]
                if not st:
                    st["step"]=0
                    st["pos"]=torch.zeros_like(p)
                    st["neg"]=torch.zeros_like(p)
                st["step"]+=1
                cur,old=(st["pos"],st["neg"]) if st["step"]%2==1 else (st["neg"],st["pos"])
                cur.mul_(beta1_sq).add_(g,alpha=1-beta1_sq)
                d=((1+beta0)*cur-beta0*old)/noise_norm
                p.add_(d,alpha=-group["lr"])
        return loss


class AdaPNM(Optimizer):
    """AdaPNM (Xie et al. 2021): Adam/AMSGrad with PNM first moment.

    m_t = beta1^2 m_{t-2} + (1-beta1^2) g_t
    mhat_t = ((1+beta0)m_t-beta0*m_{t-1})/(1-beta1^t)
    v_t = beta2*v_{t-1} + (1-beta2) g_t^2
    vhat_t = max(v_t)/(1-beta2^t) when amsgrad=True
    u_t = mhat_t / (sqrt((1+beta0)^2+beta0^2)*(sqrt(vhat_t)+eps)).
    """
    def __init__(self,params,lr=1e-3,betas=(.9,.999,1.0),eps=1e-8,
                 weight_decay=0.,amsgrad=True,decoupled_weight_decay=True):
        b1,b2,b0=betas
        _validate_betas((b1,b2))
        if b0 < 0: raise ValueError("beta0 must be nonnegative")
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,
                                     amsgrad=amsgrad,decoupled_weight_decay=decoupled_weight_decay))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            b1,b2,b0=group["betas"]
            b1sq=b1*b1
            noise_norm=math.sqrt((1+b0)**2+b0**2)
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad
                if group["weight_decay"] and not group["decoupled_weight_decay"]:
                    g=g.add(p,alpha=group["weight_decay"])
                st=self.state[p]
                if not st:
                    st["step"]=0
                    st["pos"]=torch.zeros_like(p)
                    st["neg"]=torch.zeros_like(p)
                    st["v"]=torch.zeros_like(p)
                    st["vmax"]=torch.zeros_like(p)
                st["step"]+=1; t=st["step"]
                cur,old=(st["pos"],st["neg"]) if t%2==1 else (st["neg"],st["pos"])
                cur.mul_(b1sq).add_(g,alpha=1-b1sq)
                mhat=((1+b0)*cur-b0*old)/(1-b1**t)
                st["v"].mul_(b2).addcmul_(g,g,value=1-b2)
                vv=st["v"]
                if group["amsgrad"]:
                    torch.maximum(st["vmax"],vv,out=st["vmax"])
                    vv=st["vmax"]
                vhat=vv/(1-b2**t)
                if group["weight_decay"] and group["decoupled_weight_decay"]:
                    p.mul_(1-group["lr"]*group["weight_decay"])
                denom=vhat.sqrt().add(group["eps"])
                p.addcdiv_(mhat/noise_norm,denom,value=-group["lr"])
        return loss


class Lookahead(Optimizer):
    """Lookahead wrapper (Zhang et al. 2019).

    Fast weights are updated by `base_optimizer`; every k steps:
    slow <- slow + alpha*(fast-slow); fast <- slow.
    """
    def __init__(self,base_optimizer:Optimizer,k=5,alpha=.5):
        self.base_optimizer=base_optimizer; self.k=k; self.alpha=alpha; self._la_step=0
        # Initialize Optimizer internals while sharing base param groups.
        super().__init__(base_optimizer.param_groups,{})
        self.param_groups=base_optimizer.param_groups
        self.state=base_optimizer.state
        self.slow={p:p.detach().clone() for g in self.param_groups for p in g["params"]}
    def zero_grad(self,*a,**kw): return self.base_optimizer.zero_grad(*a,**kw)
    @torch.no_grad()
    def step(self,closure=None):
        loss=self.base_optimizer.step(closure); self._la_step+=1
        if self._la_step%self.k==0:
            for g in self.param_groups:
                for p in g["params"]:
                    s=self.slow[p]; s.add_(p-s,alpha=self.alpha); p.copy_(s)
        return loss
    def state_dict(self):
        d=self.base_optimizer.state_dict()
        d["lookahead_slow"]=[self.slow[p].clone() for g in self.param_groups for p in g["params"]]
        d["lookahead_step"]=self._la_step
        return d


class Ranger(Lookahead):
    """RAdam + Lookahead ('Ranger' family). RAdam itself is core PyTorch."""
    def __init__(self,params,lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0.,k=6,alpha=.5):
        base=torch.optim.RAdam(params,lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        super().__init__(base,k=k,alpha=alpha)


class SAM(Optimizer):
    """Sharpness-Aware Minimization (Foret et al. 2021).

    Usage:
      loss = closure(); loss.backward()
      optimizer.first_step(zero_grad=True)
      closure().backward()
      optimizer.second_step(zero_grad=True)

    `step(closure)` performs both passes when closure does backward itself.
    """
    def __init__(self,params,base_optimizer_cls:Type[Optimizer]=torch.optim.SGD,rho=.05,adaptive=False,**kwargs):
        params=list(params)
        self.rho=rho; self.adaptive=adaptive
        self.base_optimizer=base_optimizer_cls(params,**kwargs)
        super().__init__(self.base_optimizer.param_groups,{})
        self.param_groups=self.base_optimizer.param_groups
    @torch.no_grad()
    def _grad_norm(self):
        norms=[]
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    scale=p.abs() if self.adaptive else 1.0
                    norms.append(_norm(p.grad*scale))
        if not norms: return torch.tensor(0.)
        return torch.linalg.vector_norm(torch.stack(norms))
    @torch.no_grad()
    def first_step(self,zero_grad=False):
        norm=self._grad_norm()
        scale=self.rho/(norm+1e-12)
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                e=(p.pow(2) if self.adaptive else 1.0)*p.grad*scale
                self.state[p]["e_w"]=e
                p.add_(e)
        if zero_grad: self.zero_grad()
    @torch.no_grad()
    def second_step(self,zero_grad=False):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    def zero_grad(self,*a,**kw): self.base_optimizer.zero_grad(*a,**kw)
    def step(self,closure=None):
        if closure is None: raise RuntimeError("SAM requires a closure")
        with torch.enable_grad(): loss=closure()
        self.first_step(zero_grad=True)
        with torch.enable_grad(): closure()
        self.second_step()
        return loss


class ASAM(SAM):
    """Adaptive SAM (Kwon et al. 2021), p=2 elementwise normalization.

    With T_w = |w| + eta, the perturbation is
        epsilon = rho * T_w^2 g / ||T_w g||_2.

    This generic optimizer cannot infer parameter *names*, so T_w is applied to
    every supplied parameter. To reproduce the original weight-vs-bias policy,
    put bias parameters in a separate ordinary SAM/base-optimizer group.
    """
    def __init__(self,params,base_optimizer_cls:Type[Optimizer]=torch.optim.SGD,
                 rho=.5,eta=.01,**kwargs):
        self.eta=eta
        super().__init__(params,base_optimizer_cls=base_optimizer_cls,rho=rho,adaptive=False,**kwargs)
    @torch.no_grad()
    def _grad_norm(self):
        norms=[]
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    tw=p.abs().add(self.eta)
                    norms.append(_norm(p.grad*tw))
        if not norms:
            return torch.tensor(0.)
        return torch.linalg.vector_norm(torch.stack(norms))
    @torch.no_grad()
    def first_step(self,zero_grad=False):
        norm=self._grad_norm()
        scale=self.rho/(norm+1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                tw=p.abs().add(self.eta)
                e=tw.pow(2)*p.grad*scale
                self.state[p]["e_w"]=e
                p.add_(e)
        if zero_grad: self.zero_grad()


class SGDW(Optimizer):
    """SGD with decoupled weight decay (Loshchilov & Hutter 2019)."""
    def __init__(self,params,lr=1e-2,momentum=0.,weight_decay=0.,dampening=0.,nesterov=False):
        super().__init__(params,dict(lr=lr,momentum=momentum,weight_decay=weight_decay,dampening=dampening,nesterov=nesterov))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if group["weight_decay"]: p.mul_(1-group["lr"]*group["weight_decay"])
                d=p.grad; mu=group["momentum"]
                if mu:
                    st=self.state[p]
                    if "buf" not in st: st["buf"]=d.clone()
                    else: st["buf"].mul_(mu).add_(d,alpha=1-group["dampening"])
                    d=d.add(st["buf"],alpha=mu) if group["nesterov"] else st["buf"]
                p.add_(d,alpha=-group["lr"])
        return loss


class AdamP(Optimizer):
    """AdamP (Heo et al. 2021), faithful channel/layer projection form."""
    def __init__(self,params,lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0.,
                 delta=.1,wd_ratio=.1,nesterov=False):
        _validate_betas(betas)
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,
                                     delta=delta,wd_ratio=wd_ratio,nesterov=nesterov))
    @staticmethod
    def _channel_view(x): return x.reshape(x.shape[0],-1)
    @staticmethod
    def _layer_view(x): return x.reshape(1,-1)
    @classmethod
    def _project(cls,p,grad,perturb,delta,wd_ratio,eps):
        if p.ndim<=1: return perturb,1.0
        expand=(-1,)+(1,)*(p.ndim-1)
        for view in (cls._channel_view,cls._layer_view):
            pv=view(p); gv=view(grad)
            cos=(pv*gv).sum(1).abs()/((pv.norm(dim=1)+eps)*(gv.norm(dim=1)+eps))
            if cos.max() < delta/math.sqrt(pv.shape[1]):
                pn=p/(pv.norm(dim=1).reshape(expand)+eps)
                perturb=perturb-pn*view(pn*perturb).sum(1).reshape(expand)
                return perturb,wd_ratio
        return perturb,1.0
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            b1,b2=group["betas"]
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad; st=self.state[p]
                if not st: st["step"]=0; st["m"]=torch.zeros_like(p); st["v"]=torch.zeros_like(p)
                st["step"]+=1; t=st["step"]
                st["m"].mul_(b1).add_(g,alpha=1-b1)
                st["v"].mul_(b2).addcmul_(g,g,value=1-b2)
                bc1=1-b1**t; bc2=1-b2**t
                denom=st["v"].sqrt()/math.sqrt(bc2)+group["eps"]
                if group["nesterov"]:
                    perturb=(b1*st["m"]+(1-b1)*g)/denom
                else:
                    perturb=st["m"]/denom
                perturb,ratio=self._project(p,g,perturb,group["delta"],group["wd_ratio"],group["eps"])
                if group["weight_decay"]:
                    p.mul_(1-group["lr"]*group["weight_decay"]*ratio)
                p.add_(perturb,alpha=-group["lr"]/bc1)
        return loss


class SGDP(Optimizer):
    """SGDP (Heo et al. 2021), faithful projected momentum-SGD form."""
    def __init__(self,params,lr=1e-2,momentum=.9,weight_decay=0.,dampening=0.,
                 nesterov=False,eps=1e-8,delta=.1,wd_ratio=.1):
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov requires momentum > 0 and dampening == 0")
        super().__init__(params,dict(lr=lr,momentum=momentum,weight_decay=weight_decay,
                                     dampening=dampening,nesterov=nesterov,eps=eps,
                                     delta=delta,wd_ratio=wd_ratio))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            mu=group["momentum"]
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad; st=self.state[p]
                if "buf" not in st: st["buf"]=torch.zeros_like(p)
                st["buf"].mul_(mu).add_(g,alpha=1-group["dampening"])
                d=g.add(st["buf"],alpha=mu) if group["nesterov"] else st["buf"]
                d,ratio=AdamP._project(p,g,d,group["delta"],group["wd_ratio"],group["eps"])
                if group["weight_decay"]:
                    decay_denom=1-mu
                    if decay_denom<=0: raise ValueError("SGDP weight decay requires momentum < 1")
                    p.mul_(1-group["lr"]*group["weight_decay"]*ratio/decay_denom)
                p.add_(d,alpha=-group["lr"])
        return loss


class Padam(Optimizer):
    """Partially adaptive momentum estimation (Chen & Gu 2018).
    Adam/AMSGrad denominator raised to partial exponent p (typically 1/8).
    """
    def __init__(self,params,lr=1e-3,betas=(.9,.999),partial=.125,eps=1e-8,weight_decay=0.,amsgrad=True):
        _validate_betas(betas)
        super().__init__(params,dict(lr=lr,betas=betas,partial=partial,eps=eps,weight_decay=weight_decay,amsgrad=amsgrad))
    @torch.no_grad()
    def step(self,closure=None):
        loss=_closure_loss(closure)
        for group in self.param_groups:
            b1,b2=group["betas"]
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad
                if group["weight_decay"]: g=g.add(p,alpha=group["weight_decay"])
                st=self.state[p]
                if not st:
                    st["step"]=0; st["m"]=torch.zeros_like(p); st["v"]=torch.zeros_like(p); st["vmax"]=torch.zeros_like(p)
                st["step"]+=1; t=st["step"]
                st["m"].mul_(b1).add_(g,alpha=1-b1); st["v"].mul_(b2).addcmul_(g,g,value=1-b2)
                vv=st["v"]
                if group["amsgrad"]:
                    torch.maximum(st["vmax"],vv,out=st["vmax"]); vv=st["vmax"]
                mh=st["m"]/(1-b1**t); vh=vv/(1-b2**t)
                p.addcdiv_(mh,vh.pow(group["partial"]).add(group["eps"]),value=-group["lr"])
        return loss


class Eve(Optimizer):
    """Eve (Koushik, Hayashi & Neubig 2016), Adam with global LR feedback.

    For t>1:
      d_t = |f_t-f_{t-1}| / (min(f_t,f_{t-1}) - f_star)
      d_hat = clip(d_t, 1/c, c)
      d_tilde = beta3*d_tilde_prev + (1-beta3)*d_hat
      alpha_t = alpha_1 / d_tilde

    Requires a closure that recomputes the objective *and gradients*. The global
    optimum f_star defaults to 0, as in the paper's cross-entropy/MSE setting.
    """
    def __init__(self,params,lr=1e-3,betas=(.9,.999,.999),eps=1e-8,c=10.,
                 f_star=0.,weight_decay=0.):
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,c=c,f_star=f_star,
                                     weight_decay=weight_decay))
        self.f_prev=None; self.d_tilde=1.0
    def step(self,closure=None):
        if closure is None: raise RuntimeError("Eve requires a closure that computes loss and gradients")
        with torch.enable_grad():
            loss=closure()
        f=float(loss.detach())
        group0=self.param_groups[0]; b3=group0["betas"][2]
        if self.f_prev is not None:
            denom=min(f,self.f_prev)-group0["f_star"]
            raw=abs(f-self.f_prev)/max(denom,group0["eps"])
            dhat=min(max(raw,1/group0["c"]),group0["c"])
            self.d_tilde=b3*self.d_tilde+(1-b3)*dhat
        self.f_prev=f
        with torch.no_grad():
            for group in self.param_groups:
                b1,b2,_=group["betas"]
                for p in group["params"]:
                    if p.grad is None: continue
                    g=p.grad
                    if group["weight_decay"]: g=g.add(p,alpha=group["weight_decay"])
                    st=self.state[p]
                    if not st: st["step"]=0; st["m"]=torch.zeros_like(p); st["v"]=torch.zeros_like(p)
                    st["step"]+=1;t=st["step"]
                    st["m"].mul_(b1).add_(g,alpha=1-b1)
                    st["v"].mul_(b2).addcmul_(g,g,value=1-b2)
                    mh=st["m"]/(1-b1**t); vh=st["v"]/(1-b2**t)
                    p.addcdiv_(mh,vh.sqrt().add(group["eps"]),value=-group["lr"]/self.d_tilde)
        return loss


# Registry
OPTIMIZER_REGISTRY = {
    name: obj for name,obj in globals().copy().items()
    if isinstance(obj,type) and issubclass(obj,Optimizer) and obj is not Optimizer
    and name not in {"Lookahead"}  # wrapper kept separately but still exported
}
OPTIMIZER_REGISTRY["Lookahead"] = Lookahead

CORE_TORCH_OPTIMIZERS = [
    "Adadelta","Adafactor","Adagrad","Adam","AdamW","SparseAdam","Adamax",
    "ASGD","LBFGS","Muon","NAdam","RAdam","RMSprop","Rprop","SGD"
]
