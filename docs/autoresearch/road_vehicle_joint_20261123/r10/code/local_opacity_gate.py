"""One target-free global-row opacity sidecar over a frozen local joint asset.

Only root-scheduled construction enables CUDA. No source, camera, geometry,
colour, canonical actor or local-Gaussian field is optimized by this adapter.
"""
from pathlib import Path
import numpy as np
from local_gaussian_asset import bank, read, write, sha
from local_joint_adapter import LocalJointAdapter


def pin(path):
    path=Path(path)
    return dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size)


class OpacityGateAdapter(LocalJointAdapter):
    def __init__(self,asset,output,ids,kinds,*,enable_gpu=False,values=None):
        super().__init__(asset,output,enable_gpu=enable_gpu)
        assert self.asset.manifest['native_gate']['mode']=='identity_no_retirement'
        assert len(self.asset.native_gate['global_row_ids'])==0
        torch=self.torch
        ids=np.asarray(ids);kinds=np.asarray(kinds)
        assert ids.dtype==np.int64 and np.array_equal(ids,np.unique(ids)) and len(ids)>0
        assert kinds.shape==ids.shape and np.isin(kinds,[0,1]).all()
        self.opacity_ids=torch.as_tensor(ids,device='cuda',dtype=torch.long)
        self.opacity_kinds=torch.as_tensor(kinds,device='cuda',dtype=torch.long)
        initial=np.ones(len(ids),np.float32) if values is None else np.asarray(values)
        assert initial.dtype==np.float32 and initial.shape==ids.shape
        assert np.isfinite(initial).all() and ((initial>=0)&(initial<=1)).all()
        self.opacity_parameter=torch.nn.Parameter(torch.as_tensor(initial,device='cuda').clone())
        self.trainable=[self.opacity_parameter]
        self.local_initial_dc=self.local.dc.detach().clone()
        self.local_initial_active=self.local.active.detach().clone()
        self.native_end=None
        # A candidate must retain the same prefix position and group in all
        # five immutable cameras. Missing rows are never interpreted as zero.
        for raw in self.frames:
            _,group,owners,gids,offset=self.base.context[raw]
            if self.native_end is None:self.native_end=int(offset)
            assert self.native_end==int(offset)
            pos=torch.searchsorted(gids,self.opacity_ids)
            assert (pos<len(gids)).all() and torch.equal(gids[pos],self.opacity_ids)
            assert torch.equal(group[pos].long(),self.opacity_kinds)
            native=self.opacity_kinds==0
            assert (owners[pos[native]]<=0).all()
            assert not torch.isin(self.opacity_ids[native],self.base.preserve).any()
            road=~native
            assert ((self.opacity_ids[road]-offset)>=0).all()
            assert ((self.opacity_ids[road]-offset)<len(self.base.road_active)).all()

    def _multiply(self,packet,gids):
        torch=self.torch
        positions=torch.searchsorted(self.opacity_ids,gids)
        safe=positions.clamp_max(len(self.opacity_ids)-1)
        selected=(positions<len(self.opacity_ids))&(self.opacity_ids[safe]==gids)
        rows=torch.nonzero(selected).reshape(-1)
        result=dict(packet);opacity=packet['opacities'].clone()
        opacity[rows]=opacity[rows]*self.opacity_parameter[safe[rows],None]
        result['opacities']=opacity
        return result

    def _native_gate(self,raw,packet):
        if 'opacity_parameter' not in self.__dict__:
            return super()._native_gate(raw,packet)
        return self._multiply(packet,self.base.context[raw][3])

    def packet(self,raw,component='shared'):
        if component in ('road','existing_road') and 'opacity_parameter' in self.__dict__:
            road=self.base.packet(raw,'road')
            gids=self.torch.arange(len(road['opacities']),device='cuda')+self.native_end
            road=self._multiply(road,gids)
            return road if component=='existing_road' else self.base.scene._bindings().merge_rows(road,self.local_packet(raw))
        return super().packet(raw,component)

    def assert_opacity_only(self):
        self.assert_fixed();torch=self.torch
        assert torch.equal(self.local.dc,self.local_initial_dc)
        assert torch.equal(self.local.active,self.local_initial_active)
        assert torch.isfinite(self.opacity_parameter).all()
        assert ((self.opacity_parameter>=0)&(self.opacity_parameter<=1)).all()
        assert all(not p.requires_grad for p in self.local.parameters())
        assert all(not p.requires_grad for _,f in self.base.actors for p in f.parameters())

    def export_sidecar(self,directory,record):
        self.assert_opacity_only();directory=Path(directory)
        directory.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(directory/'opacity_gate.npz',global_row_ids=self.opacity_ids.cpu().numpy(),
            group_kind=self.opacity_kinds.cpu().numpy().astype(np.int8),
            gate=self.opacity_parameter.detach().cpu().numpy(),road_active_world_rows=self.base.road_active)
        result=dict(schema='R10.target_free_global_opacity_sidecar.v1',complete=True,
            parent_asset_manifest=pin(self.asset.directory/'manifest.json'),
            native_global_row_count=self.native_end,gate=pin(directory/'opacity_gate.npz'),
            actor_parent_sha256=self.asset.manifest['actor_parent_sha256'],
            source_field_sha256=self.asset.manifest['source']['field']['sha256'],
            rows=len(self.opacity_ids),group_names=['unowned_native','existing_RoadRows'],
            only_opacity_multiplier_changed=True,all_parent_files_unmodified=True,
            load_reads_targets=False,training_record=record)
        write(directory/'manifest.json',result)
        return result


def load_opacity_sidecar(directory,output,*,enable_gpu=False):
    """Fresh loader: parent asset + global gate only; no fit contract/targets."""
    directory=Path(directory);m=read(directory/'manifest.json')
    assert m['complete'] and m['schema']=='R10.target_free_global_opacity_sidecar.v1'
    assert m['load_reads_targets'] is False
    parent=Path(m['parent_asset_manifest']['path'])
    assert sha(parent)==m['parent_asset_manifest']['sha256']
    path=directory/'opacity_gate.npz';assert sha(path)==m['gate']['sha256']
    z=bank(path)
    rt=OpacityGateAdapter(parent.parent,output,z['global_row_ids'],z['group_kind'],enable_gpu=enable_gpu,values=z['gate'])
    assert np.array_equal(rt.base.road_active,z['road_active_world_rows'])
    assert rt.native_end==m['native_global_row_count']
    assert rt.asset.manifest['source']['field']['sha256']==m['source_field_sha256']
    return rt
