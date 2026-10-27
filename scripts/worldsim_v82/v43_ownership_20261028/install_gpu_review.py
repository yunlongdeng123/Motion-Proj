"""Install verified native experiments in the SAME first-scene workspace."""
from pathlib import Path
import argparse,json,shutil,copy,base64,zlib
import numpy as np
B=Path(__file__).resolve().parents[2];O=B/'outputs/V43_Ownership_20261027/gpu_v43';S=B/'outputs/GS_Scene_Diagnostics_20261027/14705303724557273004_3105_000_3125_000'
def wrap(p):
    rel=p.relative_to(S).as_posix();p.with_name(p.name+'.js').write_text('window.IMAGE_DATA['+json.dumps(rel)+']='+json.dumps('data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode())+';',encoding='utf8')
def copy_images(source,target):
    shutil.copytree(source,target,dirs_exist_ok=True)
    for f in target.rglob('*.png'):wrap(f)
def unpack(p):
    data=zlib.decompress(base64.b64decode(p['data']));return {k:np.frombuffer(data[a['offset']:a['offset']+a['bytes']],np.dtype(a['dtype'])).reshape(a['shape']).copy() for k,a in p['arrays'].items()}
def pack(arrays):
    b=b'';meta={}
    for k,v in arrays.items():
        v=np.ascontiguousarray(v);raw=v.tobytes();meta[k]=dict(offset=len(b),bytes=len(raw),dtype=v.dtype.str,shape=list(v.shape));b+=raw
    return dict(arrays=meta,data=base64.b64encode(zlib.compress(b,6)).decode())
def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['ablation','opacity'],required=True);p.add_argument('--boundary-only',action='store_true');a=p.parse_args();D=O/('first_review_'+('boundary' if a.boundary_only else a.part)+'_delivery_v1');assert json.loads((D/'delivery_validation.json').read_text())['archive_and_files_verified'];v=json.loads((S/'v43_groups_data.json').read_text());groups={g['id']:g for g in v['groups']};added=[]
    def put(g,geom,evidence):
        v['groups']=[old for old in v['groups'] if old['id']!=g['id']]+[g];v['geometry'][g['id']]=geom;(S/'evidence'/(g['id']+'_metrics.json')).write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf8');added.append(g['id'])
    if a.part=='ablation':
        summary=json.loads((D/'summary.json').read_text());modes=dict(hide_unknown='去unknown',hide_vehicle='去车辆14–19',hide_both='去车辆＋unknown')
        for radius in [0,5]:
            for mode,title in modes.items():
                rel=Path('assets/v43_review/ablation')/(str(radius)+'px')/mode;copy_images(D/(str(radius)+'px')/mode,S/rel);geom=copy.deepcopy(v['geometry']['v43_motion_no_edit']);frames={}
                for raw,q in geom.items():
                    q['semantic_render_filter']=dict(mode=mode,radius=radius,fraction=.2,static_unknown=True);q['note']='实际原生RGB剥离；固定'+str(radius)+'px与20%规则，含全局unknown/其他车辆影响；未补全。右侧按同一冻结行标签筛选，不受预览带宽改动。';frames[raw]=dict(channels=dict(rgb=(rel/('%03d.png'%int(raw))).as_posix(),alpha=(rel/('%03d_alpha.png'%int(raw))).as_posix()))
                gid='v43_remove_'+mode+'_'+str(radius)+'px';put(dict(id=gid,title='原生剥离 · '+title+' / '+str(radius)+'px',category='v43',kind='v43_semantic_ablation',frames=frames,summary='evidence/'+gid+'_metrics.json'),geom,summary)
    else:
        names=dict(boundary='内外分区alpha控制') if a.boundary_only else dict(legacy='原归一化控制',balanced='两域alpha平衡',field='共享局部校准')
        for arm,title in names.items():
            source=D/arm;rel=Path('assets/v43_review/opacity')/arm;copy_images(source,S/rel);cloud=copy.deepcopy(v['clouds']['v43:10000']);data=unpack(cloud['data']);opacity=np.load(source/'effective_opacity.npz')['opacity'];assert len(opacity)==len(data['ids']);data['opacity']=opacity.astype('float32');cloud['data']=pack(data);cloud.pop('opacity_density_scale',None);ck='v43:opacity_'+arm;v['clouds'][ck]=cloud
            for case in ['no_edit','turn_left15','translate_lateral_1m']:
                base='v43_actor_10000' if case=='no_edit' else 'v43_motion_'+case;geom=copy.deepcopy(v['geometry'][base]);frames={}
                for raw,q in geom.items():
                    for obj in q['objects']:obj['cloud']=ck;obj['label']='V43 '+arm
                    q['note']='冻结车身10K起点后，'+title+'额外10K；点数/几何/DC固定。源语义仍取原10K观测，不是新opacity的贡献标签。';f=dict(channels=dict(rgb=(rel/case/'rgb'/('%03d.png'%int(raw))).as_posix(),dynamic=(rel/case/'actor'/('%03d.png'%int(raw))).as_posix()))
                    if case!='no_edit':
                        key='evidence/v43_review_maps/'+arm+'/'+case+'/'+raw+'.js';dest=S/key;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/case/('%03d_maps.js'%int(raw)),dest);f['maps']=key
                    frames[raw]=f
                gid='v43_opacity_'+arm+'_'+case;put(dict(id=gid,title=title+' · 再10K / '+dict(no_edit='原视角',turn_left15='转向',translate_lateral_1m='横移')[case],category='v43',kind='v43_opacity_calibration',frames=frames,summary='evidence/'+gid+'_metrics.json'),geom,json.loads((source/'contract.json').read_text()))
    v['badcase_available']=True;(S/'v43_groups_data.json').write_text(json.dumps(v,separators=(',',':')),encoding='utf8');(S/'v43_groups.js').write_text('window.V43_BADCASE_AVAILABLE=true;window.V43_DELIVERY='+json.dumps(v,separators=(',',':'))+';window.SCENE_DATA.groups.push(...window.V43_DELIVERY.groups);Object.assign(window.EDIT_GEOMETRY.groups,window.V43_DELIVERY.geometry);Object.assign(window.FOREIGN_CLOUDS,window.V43_DELIVERY.clouds);',encoding='utf8');print(json.dumps(dict(installed=added,models_downloaded=False)))
if __name__=='__main__':main()
