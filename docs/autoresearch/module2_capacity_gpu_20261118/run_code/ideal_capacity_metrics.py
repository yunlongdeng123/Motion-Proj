"""固定D_color与独立解析标线的CPU评价；不调用renderer，不用未知标线相位作GT。"""
import importlib.util
import math
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion,distance_transform_edt


def rgb_metric(prediction,target,mask):
    if not mask.any():return dict(pixels=0,RMSE=None,MAE=None)
    delta=prediction[:,mask].astype(np.float64)-target[:,mask].astype(np.float64)
    return dict(pixels=int(mask.sum()),RMSE=float(np.sqrt(np.mean(delta**2))),MAE=float(np.mean(np.abs(delta))))


class DetailMetrics:
    def __init__(self,data):
        self.data=data
        path=data.root.parent/'ideal_code/ideal_world_teacher.py'
        spec=importlib.util.spec_from_file_location('capacity_independent_world_field',path)
        self.field=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.field)
        self.cache={}

    def prepare(self,name):
        if name in self.cache:return
        case=next(c for c in self.data.field['cases'] if c['name']==name)
        targets=self.data.target(name);rows={}
        for raw in self.data.masks:
            z=targets[raw];geo=self.data.geometry[raw];M=self.data.masks[raw]
            xyz=geo['surface_world_xyz'];safe=np.where(np.isfinite(xyz),xyz,np.r_[self.data.field['world_frame']['origin_xy'],0.])
            base_case=dict(case,marking=None)
            asphalt,_=self.field.evaluate_world(safe,base_case,self.data.field)
            u,_,_,_=self.field.curve_coordinates(safe,self.data.field)
            mask=z['detail_evaluation_mask']&M
            edge_valid=binary_erosion(mask,iterations=1)
            paint=z['paint_fraction'];binary=paint>=.5
            edge=(binary&~binary_erosion(binary))&edge_valid
            groups=[]
            if case.get('marking'):
                mark=case['marking'];ids=np.rint((u-mark['phase_m'])/mark['period_m']).astype(np.int64)
                for key in np.unique(ids[mask&(paint>=.5)]):
                    g=mask&(ids==key)
                    if not np.any(g&(paint>=.5)):continue
                    groups.append((int(key),g))
            rows[raw]=dict(mask=mask,edge_valid=edge_valid,target_edge=edge,asphalt=asphalt,
                paint=paint,core=mask&(paint>=.9),flank=mask&(paint<=.1),groups=groups,target=z['target_rgb_srgb'])
        self.cache[name]=(case,rows)

    def registration(self):
        out={}
        for name in self.data.cases:
            self.prepare(name);case,rows=self.cache[name]
            out[name]={raw:dict(detail_pixels=int(z['mask'].sum()),edge_pixels=int(z['target_edge'].sum()),
                paint_core=int(z['core'].sum()),asphalt_flank=int(z['flank'].sum()),known_dash_fragments=len(z['groups'])) for raw,z in rows.items()}
        return dict(masks='pre-frozen target detail_sampling mask intersect D_color, one-pixel erosion for edges only',
            edge_matching_tolerance_px=1.,center_error_aggregation='maximum weighted paint-centroid shift per known dash fragment and view',
            contrast='known paint core minus known asphalt flank; native premult contrast projection, only alpha>.9',
            lowfreq_contrast='all-five fixed-alpha mean-centered target modulation projection gain',cases=out)

    def evaluate(self,name,images):
        self.prepare(name);case,rows=self.cache[name];result=[]
        if not case.get('marking'):
            if name=='constant':return dict(status='not_applicable_constant',passed=True,frames=[])
            numerator=denominator=0.
            for raw in rows:
                M=self.data.masks[raw];T=rows[raw]['target'][:,M].T.astype(np.float64)
                a=self.data.geometry[raw]['fixed_alpha'][M].astype(np.float64)[:,None]
                P=images[raw][:,M].T.astype(np.float64);mu=T.mean(0)
                signal=a*(T-mu);numerator+=float((signal*(P-a*mu)).sum());denominator+=float((signal**2).sum())
            gain=numerator/denominator if denominator>0 else None
            return dict(status='lowfreq_contrast',passed=gain is not None and gain>=.9,modulation_gain=gain,frames=[])
        for raw,z in rows.items():
            alpha=self.data.geometry[raw]['fixed_alpha'].astype(np.float64)
            P=images[raw].transpose(1,2,0).astype(np.float64);A=z['asphalt'];d=np.asarray(self.data.field['paint_srgb'])-A
            den=alpha*np.sum(d*d,axis=2)
            fraction=np.zeros(alpha.shape);np.divide(np.sum((P-alpha[...,None]*A)*d,axis=2),den,out=fraction,where=den>1e-12)
            fraction=np.clip(fraction,0,1);binary=fraction>=.5
            pe=(binary&~binary_erosion(binary))&z['edge_valid'];ge=z['target_edge']
            if ge.any() and pe.any():
                precision=float((distance_transform_edt(~ge)[pe]<=1.).mean())
                recall=float((distance_transform_edt(~pe)[ge]<=1.).mean());F1=2*precision*recall/(precision+recall) if precision+recall else 0.
            else:precision=recall=F1=0.
            yy,xx=np.mgrid[:alpha.shape[0],:alpha.shape[1]];errors=[];missing=0
            for key,mask in z['groups']:
                gt=z['paint'][mask].astype(np.float64);pr=fraction[mask]
                if gt.sum()<=0:continue
                if pr.sum()<=1e-12:missing+=1;continue
                coords=np.c_[xx[mask],yy[mask]]
                errors.append(float(np.linalg.norm((coords*gt[:,None]).sum(0)/gt.sum()-(coords*pr[:,None]).sum(0)/pr.sum())))
            center=max(errors) if errors else None
            if z['core'].any() and z['flank'].any():
                truth=float(z['paint'][z['core']].mean()-z['paint'][z['flank']].mean())
                contrast=float((fraction[z['core']].mean()-fraction[z['flank']].mean())/truth) if truth>0 else None
            else:contrast=None
            passed=F1>=.9 and center is not None and center<=1. and missing==0 and contrast is not None and contrast>=.9
            result.append(dict(raw=raw,detail_pixels=int(z['mask'].sum()),edge_precision=precision,edge_recall=recall,edge_F1=F1,
                known_dash_fragments=len(z['groups']),missing_predicted_fragments=missing,line_center_error_max_px=center,
                contrast_retention=contrast,passed=passed))
        stress=case['gate_role'].startswith('diagnostic')
        return dict(status='stress_diagnostic_only' if stress else ('pass' if all(r['passed'] for r in result) else 'not_met_or_not_evaluable'),
            passed=False if stress else all(r['passed'] for r in result),never_negative_capacity_evidence=stress,frames=result)
