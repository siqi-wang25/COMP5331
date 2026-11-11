"""Check numerical parity, input gradients, ablations and saved state."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from models.damps import DAMPS
from models.damps_projected import DAMPSProjected

torch.manual_seed(999)
x=torch.randn(40,64);y=torch.randn(40,64)
legacy=DAMPS(64,'cpu',x,y)
with torch.no_grad():
    legacy.image_trs.weight.copy_(torch.eye(64)); legacy.image_trs.bias.zero_()
    legacy.text_trs.weight.copy_(torch.eye(64)); legacy.text_trs.bias.zero_()
adapter=DAMPSProjected(x,y)
with torch.no_grad():
    for target,source in [('avrf_image','AVRF_image'),('avrf_text','AVRF_txt'),('phase_residual','psi'),('mix_logits','lambda_weights'),('phase_prior','avg_R')]:
        getattr(adapter,target).copy_(getattr(legacy,source))
for got,expected in zip(adapter(x,y),legacy(x,y)):
    torch.testing.assert_close(got,expected,rtol=1e-6,atol=1e-6)
for variant in ['full','no_apc','no_avrf','no_imcf']:
    m=DAMPSProjected(x,y,variant)
    a=x.clone().requires_grad_();b=y.clone().requires_grad_()
    out=m(a,b);sum(t.square().mean() for t in out).backward()
    assert all(t.shape==x.shape and torch.isfinite(t).all() for t in out)
    assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0
    for name,param in m.named_parameters():
        if param.requires_grad: assert param.grad is not None and torch.isfinite(param.grad).all(),(variant,name)
    reloaded=DAMPSProjected(x,y,variant);reloaded.load_state_dict(m.state_dict())
    for u,v in zip(m(x,y),reloaded(x,y)):torch.testing.assert_close(u,v)
    assert not torch.allclose(m(x+0.1,y)[0],m(x,y)[0])
# Document the published one-tap coherence degeneracy; do not silently replace it.
a=torch.fft.rfft(x);b=torch.fft.rfft(y)
coh=(a*b.conj()).abs().square()/(a.abs().square()*b.abs().square()+1e-8)
assert (coh-1).abs().max()<1e-4
print('PASS: legacy spectral parity, all four variants, input/parameter gradients, state reload, coherence diagnostic')
