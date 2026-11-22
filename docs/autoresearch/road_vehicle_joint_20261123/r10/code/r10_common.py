"""R10 immutable inputs, accepted original actor, and evaluation utilities."""
from pathlib import Path
import sys
R8=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_vehicle_joint_dggt_20261123_r8')
sys.path.insert(0,str(R8/'code'))
from joint_rgb_common import read,load,write,sha,source_gate,all4_and_sample,srgb_to_linear_numpy,linear_to_srgb_numpy
from joint_identity_adapter import JointIdentityAdapter


def contracts(path):
    cfg=read(path);base=read(cfg['baseline_contract']);old=read(cfg['original_R8_baseline_contract'])
    assert len(base['actors'])==1 and base['actors'][0]['sha256']==old['actors'][0]['sha256']==cfg['accepted_actor_sha256']
    assert base['actors'][0]['asset']==old['actors'][0]['asset'],'Original accepted vehicle asset is mandatory'
    assert cfg['existing_formal_updates']==400 and cfg['existing_representation_total_limit']==2000
    assert cfg['proposed_gate_optimization_max_updates']==600 and cfg['optimization_enabled'] is False
    formal=read(cfg['existing_formal_receipt']);assert formal['complete'] and formal['actual_optimizer_updates']==400
    assert cfg['ground_band_definition']==dict(x_pad=30,bottom_start=-12,bottom_end=100)
    return cfg,base


def npy(tensor):return tensor.detach().cpu().numpy()


class LocalGateAdapter(JointIdentityAdapter):
    """The diagnostic gate only multiplies selected existing native opacities."""
    def __init__(self,contract,output,gids=(),deleted_owner_ids=()):
        super().__init__(contract,output);torch=self.torch
        self.local_gate_ids=torch.as_tensor(gids,device='cuda',dtype=torch.long)
        assert not torch.isin(self.local_gate_ids,self.preserve).any()
        self.allowed_deleted_owners=tuple(deleted_owner_ids)

    def packet(self,raw,component='shared'):
        p=super().packet(raw,component)
        if component not in ('shared','context') or not len(self.local_gate_ids):return p
        torch=self.torch;_,group,owners,gids,_=self.context[raw]
        selected=(group==0)&torch.isin(gids,self.local_gate_ids)
        allowed=owners<=0
        for value in self.allowed_deleted_owners:allowed|=owners==value
        assert torch.all(allowed[selected]),'Cannot retire a retained vehicle or native selected actor'
        for spec,_ in self.actors:assert not (owners[selected]==spec['owner_id']).any()
        ids=torch.nonzero(selected).reshape(-1);q=dict(p);opacity=p['opacities'].clone();opacity[ids]=0;q['opacities']=opacity
        return q
