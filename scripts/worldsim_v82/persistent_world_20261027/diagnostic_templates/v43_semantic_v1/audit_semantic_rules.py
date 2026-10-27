"""Bounded CPU parameter diagnostics on the exact saved display sample, 3 x 49 frames."""
from pathlib import Path
import json,base64,zlib,time,html
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];H=B/'outputs/GS_Scene_Diagnostics_20261027';O=B/'outputs/V43_Ownership_20261027/semantic_audit';start=time.monotonic()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def wrapped(p):return json.loads(p.read_text(encoding='utf8').split('=',1)[1].rstrip(';\n'))
def unpack(p):
 b=zlib.decompress(base64.b64decode(p['data']));return {k:np.frombuffer(b[v['offset']:v['offset']+v['bytes']],v['dtype']).reshape(v['shape']) for k,v in p['arrays'].items()}
source=read(O/'observations.json');old=read(O.parent/'initialization.json');result={'scope':'deterministic saved display sample only; not full population or pixel contribution','GPU_used':False,'model_modified':False,'manual_reviewed':False,'profiles':{'projection':'mask center only, weak 2D association','depth_candidate':'mask + nearest observed LiDAR <=2px, z difference <=0.5m, max sigma <=0.5m','strict_proxy':'mask interior >=2px+2sigma projected radius, exact pixel LiDAR z difference <=0.15m, max sigma <=0.2m; z-only proxy is weaker than original Euclidean test'},'contexts':[]};cards=[]
for entry in source['contexts']:
 ctx=entry['context'];folder=H/ctx;d=read(folder/'scene.json');cloud=wrapped(folder/'cloud.js');frames=list(d['frames'].items());summary={'context':ctx,'frames':49,'parts':{},'original_full_rule_counts':next(x['parts'] for x in old['contexts'] if x['context']==ctx)}
 summary['original_full_rule_counts']={p:{k:a[k] for k in ['counts','any_positive_rows','any_non_target_rows']} for p,a in summary['original_full_rule_counts'].items()}
 overlays={i:Image.open(folder/d['gt'][frames[i][0]]).convert('RGB') for i in [14,32,47]}
 for part in ['static','dynamic']:
  c=cloud[part];a=unpack(c['data']);a={k:(v.ravel() if k in ['ids','t','sigma_t','opacity'] else v) for k,v in a.items()};n=len(a['ids']);sigma=a['scales'].max(1);freq=2*np.pi/c['cycle'];hits={p:np.zeros(n,int) for p in result['profiles']};bins={p:np.zeros(n,np.uint8) for p in hits};active=np.zeros(n,int);eligiblebins=np.zeros(n,np.uint8);frows=[]
  for fi,(raw,f) in enumerate(frames):
   obs=unpack(entry['frames'][raw]['observations']);mask=obs['mask'];depth=obs['lidar_z'];interior=obs['interior'];finite=np.isfinite(depth).reshape(320,480);dist,indices=distance_transform_edt(~finite,return_indices=True);near=depth.reshape(320,480)[tuple(indices)].ravel()
   t=f['model_time'];dt=t-a['t'];marg=np.exp(-.5*(dt/a['sigma_t'])**2);enabled=(marg>.05)&(a['opacity']*marg>=.005);active+=enabled;bit=1<<min(2,fi//17);eligiblebins[enabled]|=bit
   xyz=a['xyz']+a['velocity']*(np.sin(dt*freq)/freq)[:,None];M=np.array(f['c2w']);cam=(xyz-M[:3,3])@M[:3,:3];K=f['K'];uv=cam[:,:2]/np.maximum(cam[:,2:],1e-8)*[K[0],K[1]]+[K[2],K[3]];good=enabled&(cam[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<480)&(uv[:,1]>=0)&(uv[:,1]<320);ids=np.flatnonzero(good);ij=np.floor(uv[ids]).astype(int);idx=ij[:,1]*480+ij[:,0];mh=mask[idx]>0
   tests={'projection':mh,'depth_candidate':mh&(dist.ravel()[idx]<=2)&(np.abs(cam[ids,2]-near[idx])<=.5)&(sigma[ids]<=.5),'strict_proxy':mh&(interior[idx]>=2+2*sigma[ids]*max(K[:2])/cam[ids,2])&(np.abs(cam[ids,2]-depth[idx])<=.15)&(sigma[ids]<=.2)}
   for p,test in tests.items():hits[p][ids[test]]+=1;bins[p][ids[test]]|=bit
   frows.append({'raw':int(raw),'active':int(enabled.sum()),'front':len(ids),'mask_projection':int(mh.sum()),'depth_candidate':int(tests['depth_candidate'].sum()),'strict_proxy':int(tests['strict_proxy'].sum())})
   if fi in overlays:
    draw=ImageDraw.Draw(overlays[fi]);color='#ff4fd2' if part=='static' else '#42e8a2'
    for u,v in ij[tests['depth_candidate']]:draw.rectangle((int(u),int(v),int(u)+1,int(v)+1),fill=color)
  pop=np.array([bin(i).count('1') for i in range(8)]);ps={p:{'any':int((v>0).sum()),'at_least_3_frames':int((v>=3).sum()),'3_frames_2_bins':int(((v>=3)&(pop[bins[p]]>=2)).sum())} for p,v in hits.items()}
  summary['parts'][part]={'sample_rows':n,'population_rows':c['total'],'eligible_3_frames':int((active>=3).sum()),'eligible_2_time_bins':int((pop[eligiblebins]>=2).sum()),'profiles':ps,'per_frame':frows}
 for fi,im in overlays.items():im.resize((960,640)).save(O/(ctx[:6]+'_raw'+frames[fi][0]+'.png'))
 result['contexts'].append(summary)
 parts=''.join('<tr><td>'+p+'</td><td>'+str(v['sample_rows'])+'</td>'+''.join('<td>'+str(v['profiles'][pr]['any'])+' / '+str(v['profiles'][pr]['3_frames_2_bins'])+'</td>' for pr in result['profiles'])+'</tr>' for p,v in summary['parts'].items())
 cards.append('<section><h2>'+ctx+'</h2><a href="../../GS_Scene_Diagnostics_20261027/'+ctx+'/index.html#workspace">进入共享工作台</a><p>下表每格 = 至少一次支持 / ≥3帧且跨2时间段。样本行数不同，不能当全量或像素比例。</p><table><tr><th>分支</th><th>真实采样行数</th><th>仅掩码投影</th><th>深度候选</th><th>严格条件 z 代理</th></tr>'+parts+'</table>'+''.join('<figure><img src="'+ctx[:6]+'_raw'+frames[fi][0]+'.png"><figcaption>raw'+frames[fi][0]+'：原实拍＋深度候选中心投影；洋红静态，绿色动态。二维重合不证明身份正确或三维落框。</figcaption></figure>' for fi in overlays)+'</section>')
result['wall_seconds']=time.monotonic()-start
(O/'rule_sweep.json').write_text(json.dumps(result,indent=2),encoding='utf8')
svg='<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="170"><rect width="1000" height="170" fill="#101c2d"/><g fill="#1e3b50" stroke="#83a9c7"><rect x="10" y="45" width="230" height="90"/><rect x="275" y="45" width="220" height="90"/><rect x="530" y="45" width="220" height="90"/><rect x="785" y="45" width="205" height="90"/></g><g font-family="sans-serif" font-size="17" fill="white" text-anchor="middle"><text x="125" y="80">RGB / existing mask</text><text x="125" y="110">LiDAR / camera / rows</text><text x="385" y="80">CPU projection + depth</text><text x="385" y="110">candidate semantics</text><text x="640" y="80">semantic x legacy branch</text><text x="640" y="110">frame / row inspection</text><text x="887" y="80">manual review</text><text x="887" y="110">GPU attribution pending</text><text x="258" y="95">→</text><text x="513" y="95">→</text><text x="768" y="95">→</text></g></svg>'
(O/'architecture.svg').write_text(svg,encoding='utf8')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V43 CPU 语义与归属规则复测</title><style>body{background:#09121d;color:#dce8f6;font:16px/1.65 Microsoft YaHei,sans-serif;max-width:1160px;margin:30px auto}a{color:#8cc9ff}section{padding:20px;border:1px solid #34516d;margin:24px 0}img{max-width:100%}figure{display:inline-block;width:30%;vertical-align:top;margin:1%}table{border-collapse:collapse}td,th{border:1px solid #456;padding:7px}figcaption{font-size:13px}</style><h1>CPU 语义投影、分支交叉与规则复测</h1><a href="../index.html">返回 V43 架构</a><p>语义和旧渲染归属分开保存：类别已知不代表具体身份已核验。仅已审阅目标车的 DEVA 掩码提供 vehicle 候选，其余保持 unknown。蓝色车辆框只用于审计，未生成掩码或点标签。当前不是完整场景的语义分割，也未用中心投影冒充高斯渲染贡献。</p><img src="architecture.svg"><p>语义 map：0 unknown；1 vehicle；2 road；3 sidewalk；4 building；5 vegetation；6 sky；7 pole/sign；8 person/cycle。当前数据仅含 0、1，其余是预留映射。unknown 不等于背景，vehicle×static 不等于错误（停放车辆也可以静止）。洋红提示所选目标车的掩码语义可能跨入静态分支，须结合深度、RGB、可见性和实际贡献核验。</p><p>真实复测：3个开发场景各49帧、相同确定性显示样本、3组参数。严格条件只使用 LiDAR z 差的代理，弱于原始三维距离测试；旧全模型严格标签保持不变。本次没有通过调门限把候选升级成可信身份，没有把 unknown 少当成质量提高。全量行的逐帧候选可在工作台改变采样率查看；此表仅是样本统计。</p><p>GPU 待办：SAM/SegFormer 等密集语义推理、alpha×T贡献投射与遮挡排除、跨帧实例一致性、人工评审后校准。当前无卡且心跳暂停，未启动这些阶段。</p><a href="rule_sweep.json">逐帧条件计数与全部参数</a>'''+''.join(cards)+'</html>'
(O/'index.html').write_text(page,encoding='utf8')
print(json.dumps({'wall_seconds':result['wall_seconds'],'contexts':[{'context':x['context'],'parts':x['parts']} for x in []],'report':str(O/'index.html')}))
