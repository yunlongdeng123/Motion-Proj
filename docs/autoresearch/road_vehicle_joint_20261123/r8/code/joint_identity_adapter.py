"""Persistent read-only same-instance channels; no fitting inputs are needed."""
from joint_native_adapter import JointNativeAdapter


class JointIdentityAdapter(JointNativeAdapter):
    def packet(self,raw,component='shared'):
        p=super().packet(raw,component);p=dict(p);features=p['features'].clone()
        assert 5+len(self.actors)<=features.shape[1]
        if component in ('shared','context'):
            _,group,owners,_,_=self.context[raw];nc=len(owners)
            for index,(spec,field) in enumerate(self.actors):
                features[:nc,5+index]=((group==0)&(owners==spec['owner_id'])).float()
            if component=='shared':
                cursor=nc
                for index,(_,field) in enumerate(self.actors):
                    features[cursor:cursor+len(field.xyz),5+index]=1;cursor+=len(field.xyz)
        elif component=='actor':
            cursor=0
            for index,(_,field) in enumerate(self.actors):
                features[cursor:cursor+len(field.xyz),5+index]=1;cursor+=len(field.xyz)
        p['features']=features;return p

    def raster_packet(self,raw,packet,tag,environment):
        cam=self.cameras[raw];out=self.scene._bindings().raster(cam,self.scene.pipe,0,packet,self.scene._background)
        self.calls+=1;self.trace.append(dict(call=self.calls,raw=raw,tag=tag,single_native_raster=True))
        rgb=out['rgb']
        if environment:
            rgb=rgb+(1-out['alpha'])*self.scene.environment(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        return dict(rgb=rgb,alpha=out['alpha'],depth_sum_m=out['depth_sum']/self.meter_to_model,
            road_alpha=out['extra'][0],actor_alpha=out['extra'][1],context_alpha=out['extra'][2],
            unified_actor_alpha=out['extra'][3:3+len(self.actors)])
