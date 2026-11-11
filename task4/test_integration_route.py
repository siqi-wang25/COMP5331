"""Demonstrate the public author's owned-projection route versus our adapter."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from models.damps import DAMPS
from models.damps_projected import DAMPSProjected

torch.set_num_threads(2)
torch.manual_seed(999)
raw_i, raw_t = torch.randn(32, 12), torch.randn(32, 10)
author = DAMPS(16, 'cpu', raw_i, raw_t)
x, y = torch.randn(32, 16, requires_grad=True), torch.randn(32, 16, requires_grad=True)
first = author(x, y)
second = author(x + 3, y - 4)
for a, b in zip(first, second): torch.testing.assert_close(a, b, rtol=0, atol=0)
sum(z.square().mean() for z in first).backward()
assert x.grad is None and y.grad is None
assert author.image_trs.weight.grad.abs().sum() > 0
adapter = DAMPSProjected(x.detach(), y.detach())
calibrated = adapter(x, y)
assert not torch.allclose(calibrated[0], adapter(x + 3, y - 4)[0])
sum(z.square().mean() for z in calibrated).backward()
assert x.grad.abs().sum() > 0 and y.grad.abs().sum() > 0
print(json.dumps({'public_author_ignores_supplied_values': True,
                  'public_author_owned_projection_receives_gradient': True,
                  'projected_adapter_uses_inputs_and_returns_gradients': True,
                  'author_smore_integration_confirmed': False}))
