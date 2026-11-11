"""GPU check: raw SMORE retains upstream parameters, predictions, loss and gradients."""
import argparse, importlib.util, sys, tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--official',required=True);a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import scipy.sparse as sp
import torch
from models.smore import SMORE
spec=importlib.util.spec_from_file_location('official_smore',a.official);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class Dataset:
    def get_user_num(self):return 4
    def get_item_num(self):return 12
class Loader:
    dataset=Dataset()
    def inter_matrix(self,form):return sp.coo_matrix((np.ones(8),([0,0,1,1,2,2,3,3],[0,1,2,3,4,5,6,7])),shape=(4,12))
with tempfile.TemporaryDirectory() as d:
    data=Path(d)/'toy';data.mkdir();rng=np.random.default_rng(4)
    for f in ['image_feat.npy','text_feat.npy']:np.save(data/f,rng.normal(size=(12,8)).astype('float32'))
    c=dict(USER_ID_FIELD='userID',ITEM_ID_FIELD='itemID',NEG_PREFIX='neg_',train_batch_size=4,device=torch.device('cuda'),end2end=False,is_multimodal_model=True,data_path=d+'/',dataset='toy',vision_feature_file='image_feat.npy',text_feature_file='text_feat.npy',cl_loss=.01,n_ui_layers=2,embedding_size=8,n_layers=1,reg_weight=.0001,image_knn_k=3,text_knn_k=3,dropout_rate=0.,graph_cache_path=str(data),damps_variant='raw')
    torch.manual_seed(999);old=module.SMORE(c,Loader()).cuda()
    torch.manual_seed(999);new=SMORE(c,Loader()).cuda()
    for k,v in old.state_dict().items():torch.testing.assert_close(v,new.state_dict()[k],rtol=0,atol=0)
    for x,y in zip(old.forward(old.norm_adj),new.forward(new.norm_adj)):torch.testing.assert_close(x,y,rtol=0,atol=0)
    batch=torch.tensor([[0,1,2,3],[0,2,4,6],[10,10,10,10]],device='cuda')
    lo=old.calculate_loss(batch);ln=new.calculate_loss(batch);torch.testing.assert_close(lo,ln,rtol=0,atol=0)
    lo.backward();ln.backward()
    for (name,u),(_,v) in zip(old.named_parameters(),new.named_parameters()):
        if u.grad is not None:torch.testing.assert_close(u.grad,v.grad,rtol=1e-6,atol=1e-7)
    for variant in ['full','no_apc','no_avrf','no_imcf']:
        c['damps_variant']=variant;torch.manual_seed(999);m=SMORE(c,Loader()).cuda()
        for k,v in new.state_dict().items():torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)
        loss=m.calculate_loss(batch);assert torch.isfinite(loss);loss.backward()
        assert m.image_trs.weight.grad.abs().sum()>0 and m.text_trs.weight.grad.abs().sum()>0
print('PASS: raw initial weights/predictions/loss/gradients match official SMORE; all DAMPS variants preserve backbone initialization and projection gradients')
