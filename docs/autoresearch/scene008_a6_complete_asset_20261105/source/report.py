"""Package exact first complete-asset control, including negative results."""
import json,shutil,hashlib,csv,gzip
from pathlib import Path
from collections import defaultdict
import numpy as np
from full_pvd import RUN,FixedAsset
from metrics import pool,SOURCE_LIMITS

def read(path):return json.loads(path.read_text())
def write(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def flatten(x,prefix=''):
    if isinstance(x,dict):
        if {'sum','count','mean'}.issubset(x):yield prefix,x
        else:
            for key,value in x.items():yield from flatten(value,prefix+'.'+key if prefix else key)
def aggregate(rows):
    all_values=defaultdict(list)
    for row in rows:
        for key,value in flatten(row):all_values[key].append(value)
    return {key:pool(values) for key,values in all_values.items()}
def gauss_ply(npz,path):
    asset=FixedAsset.from_geometry(npz);xyz=asset.xyz.numpy();normal=asset.normal.numpy();rgb=asset.rgb.numpy()
    q=np.stack((1+normal[:,2],-normal[:,1],normal[:,0],np.zeros(2048)),1)
    invalid=np.linalg.norm(q,axis=1)<1e-5;q[invalid]=[0,1,0,0];q/=np.linalg.norm(q,axis=1)[:,None]
    scales=np.stack((asset.sigma.numpy(),asset.sigma.numpy(),asset.thickness.numpy()),1)
    a=asset.opacity.numpy();columns=np.c_[xyz,normal,(rgb-.5)/.28209479177387814,np.log(a/(1-a)),np.log(scales),q].astype('<f4')
    names=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    header='ply\nformat binary_little_endian 1.0\ncomment canonical metres; SH0; complete fixed PVD geometry\nelement vertex 2048\n'+''.join('property float '+name+'\n' for name in names)+'end_header\n'
    path.write_bytes(header.encode()+columns.tobytes())
    offset=len(header.encode());restored=np.frombuffer(path.read_bytes()[offset:],dtype='<f4').reshape(2048,-1)
    assert np.array_equal(restored,columns)

def main():
    evaluation=read(RUN/'evaluation/metrics.json');assert evaluation['all49_complete']
    fit=read(RUN/'appearance/fit_receipt.json');receipt=read(RUN/'evaluation/receipt.json')
    groups={'all49':evaluation['rows'],'BUILD40':[r for r in evaluation['rows'] if r['split']=='BUILD'],
            'internal_HELD9':[r for r in evaluation['rows'] if r['split']=='internal_HELD']}
    summary={name:aggregate(rows) for name,rows in groups.items()}
    dest=RUN/'delivery';dest.mkdir(exist_ok=False);(dest/'assets').mkdir();(dest/'evidence').mkdir()
    shutil.copytree(RUN/'evaluation/frames',dest/'frames',ignore=shutil.ignore_patterns('*_replay.npz'))
    shutil.copytree(RUN/'old_work_closeout',dest/'old_work_closeout')
    shutil.copytree(RUN/'source',dest/'source',ignore=shutil.ignore_patterns('__pycache__','*.bak*'))
    shutil.copyfile(RUN/'source/architecture.svg',dest/'architecture.svg')
    for name in ('complete_a6_fitted.npz','complete_a6_prior.npz'):
        shutil.copyfile(RUN/'appearance'/name,dest/'assets'/name)
    gauss_ply(dest/'assets/complete_a6_fitted.npz',dest/'assets/complete_a6_fitted.ply')
    for source in ('contract.json','geometry_provenance.json','baseline_parity.json','capacity_audit.json','smoke/receipt.json',
                   'appearance/fit_receipt.json','appearance/collection.json','evaluation/receipt.json'):
        shutil.copyfile(RUN/source,dest/'evidence'/source.replace('/','_'))
    for name in ('geometry_support_audit.json','next_unified_asset_contract.json'):
        if (RUN/name).exists():shutil.copyfile(RUN/name,dest/'evidence'/name)
    with gzip.open(dest/'evidence/all49_metrics.json.gz','wt',encoding='utf-8') as f:json.dump(evaluation,f,ensure_ascii=False)
    write(dest/'summary.json',summary)
    with (dest/'fixed_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['split','metric','count','mean','max','above_002','above_005'])
        for split,data in summary.items():
            for key,value in data.items():writer.writerow([split,key]+[value[k] for k in ('count','mean','max','above_002','above_005')])
    def mean(path,group='all49'):return summary[group][path]['mean']
    def fmt(value):return '—' if value is None else f'{value:.5f}'
    rows=''
    for scope,group in [('完整49帧','all49'),('内部外观留出9帧','internal_HELD9')]:
        for label,kind in [('原生 a6','solo_native'),('完整PVD＋参考色','solo_prior'),('完整PVD＋观测拟合','solo_fitted')]:
            values=[mean(kind+'.a6.'+key,group) for key in ('rgb_error','measured_depth_error','visible_coverage','reviewed_below_half')]
            rows+='<tr><td>'+scope+'</td><td>'+label+'</td>'+''.join('<td>'+fmt(v)+'</td>' for v in values)+'</tr>'
    preservation=''
    for state,label in [('keep','a37在场'),('delete','删除a37')]:
        prefix='scene.'+state
        for region,regionlabel in [('road_visible','可见道路'),('outside','非目标保护区')]+([('a37','可见a37')] if state=='keep' else []):
            value=summary['all49'][prefix+'.preservation.'+region+'.parent_rgb_change']
            preservation+=f'<tr><td>{label}</td><td>{regionlabel}</td><td>{value["count"]}</td><td>{fmt(value["mean"])}</td><td>{value["above_005"]}</td></tr>'
    n=mean('solo_native.a6.rgb_error');f=mean('solo_fitted.a6.rgb_error')
    nd=mean('solo_native.a6.measured_depth_error');fd=mean('solo_fitted.a6.measured_depth_error')
    quality=dict(quality_accepted=False,reason='Current full2048 fixed basis with SH0 appearance does not preserve measured quality; geometry, discretization and material capacity are not isolated by this control',
       native_solo_RGB=n,fitted_solo_RGB=f,native_solo_depth_m=nd,fitted_solo_depth_m=fd,
       engineering_pass=receipt['status'],appearance_holdout_not_geometry_generalization=True)
    write(dest/'quality_decision.json',quality)
    limits=''.join('<li>'+s+'</li>' for s in SOURCE_LIMITS)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scene008 单一完整 a6 资产：首轮对照</title>
<style>body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#203847;background:#f6f8fa;margin:0}main{max-width:1480px;margin:auto;padding:28px}h1{font-size:30px}h2{margin-top:38px}section{background:white;padding:22px;margin:20px 0;border-radius:10px}img{width:100%;display:block}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #d9e2e8;padding:8px;text-align:left}th{background:#edf3f6}a{color:#006b82}.note{background:#fff3dd;padding:14px;border-left:4px solid #d69a32}button{padding:7px 16px;margin:4px;cursor:pointer}input{width:55%;vertical-align:middle}.scroll{overflow-x:auto}.scroll img{min-width:1050px}.small{font-size:14px;color:#4b6575}code{overflow-wrap:anywhere}</style>
<main><h1>Scene008：从并存车身，转向单一完整 a6 资产</h1><p>旧裁剪分支已收尾。本轮从完整 2,048 点 PVD 几何出发，冻结几何、尺度和透明度，用 a6 自己的真实可见 RGB 校正一份共享外观，然后整体替换原生 a6。无编辑与删除共用同一份资产。</p>
<section><img src="architecture.svg" alt="输入、恢复、单一资产、统一合成架构"><p class="note">这是完整资产替换的第一组强对照，尚未通过画质验收。工程上消除了原生 a6 与 PVD 并存的竞争；能否保住这辆车的已观测细节，必须另外看 RGB、深度与覆盖。此轮没有修改几何，也没有实现 MOG-2 不透明首交渲染。</p></section>
<section><h2>先看对象本身</h2><p>每帧各列共用同一裁剪。第一列只展示真实可见 a6，其余未知处黑底；第二列为原生 a6；第三列为完整 PVD 的观测条件参考色；第四列为同一几何拟合后的资产；第五列为固定几何的透明度。后窗、尾灯、轮廓和孔洞需直接对照。</p>
<div><b id="rawlabel">raw113</b> <input id="slider" type="range" min="89" max="137" value="113"> <button onclick="setRaw(105)">105</button><button onclick="setRaw(113)">113</button><button onclick="setRaw(130)">130</button></div><div class="scroll"><img id="objects" src="frames/113_objects.jpg" alt="黑底对象五列对照"></div><p class="small">推断外观保留来源标记：646 点达到至少两帧且累计贡献≥1像素的支持门槛，1,402 点仍标记为弱监督／推断。没有把隐藏车身纹理当成真实细节。</p>
<h2>同一资产放回冻结街景</h2><p>依次为原生场景a37在场、新资产a37在场、原生场景删a37、新资产删a37。新资产两种状态都不含原生 a6。背景为原生10K＋道路1K，另复用已保存的固定11×9世界道路颜色场；各对照完全相同。</p><div class="scroll"><img id="street" src="frames/113_street.jpg" alt="街景保留删除配对"></div><h2>分组成像贡献</h2><p>依次为在场时新资产、a37，删除后新资产、旧静态层。全部来自同一次联合排序；这些图描述遮挡，不作为隐藏几何准确率。</p><div class="scroll"><img id="groups" src="frames/113_groups.jpg" alt="统一合成分组贡献"></div></section>
<section><h2>固定分母：已观测 a6</h2><p>RGB为sRGB通道平均绝对误差；深度单位米；覆盖为对象alpha；缺口为完整reviewed可见轮廓中alpha&lt;0.5的比例。完整核心39308像素、实测深度8930点，所有方法相同；没有按新资产alpha缩小评价分母。</p><table><thead><tr><th>范围</th><th>资产</th><th>RGB MAE</th><th>深度 MAE / m</th><th>可见核心覆盖</th><th>轮廓低覆盖比例</th></tr></thead><tbody>'''+rows+'''</tbody></table><p>内部留出仅表示这9帧未参与本轮外观拟合。原生参考和PVD几何来自同场景上下文，PVD的200个条件点中有14点来自这些帧，不能宣称几何独立测试或跨场景泛化。</p></section>
<section><h2>保留域：相对原生场景的变化</h2><p>这里报告同状态新资产与原生基线的每像素RGB变化；超过0.05的像素数是局部损伤尾部，不能只用全图平均掩盖。变化不自动等于改善或损伤，需结合原始RGB和图像审查。</p><table><thead><tr><th>状态</th><th>区域</th><th>固定像素</th><th>平均RGB变化</th><th>变化&gt;0.05像素</th></tr></thead><tbody>'''+preservation+'''</tbody></table></section>
<section><h2>实验合同与下一步</h2><p>40帧BUILD，仅真实可见a6内部RGB；不读取被a37遮挡处的RGB。每个固定高斯使用一组共享SH0颜色，共6144个有界参数，稀疏岭回归λ=0.001。无观测区域使用对象内部全局低容量颜色延续。全资产几何哈希贯穿拟合、导出、重载与49帧评价保持一致。</p><p>首轮完整提案对照用于定位固定PVD的表达限制。后续主框架应让可靠实测表面与先验共同约束同一份资产：先区分固定几何分辨率不足与表面对齐误差，再单独开放实测约束下的受限几何细化。规则依赖观测可靠性和表面距离，不依赖遮挡发生在下部、左部、右部或上部。</p><p>3帧真实GPU smoke覆盖联合渲染、分组贡献守恒、非均匀随机颜色的稀疏设计一致性，以及资产保存／恢复／后续入口。原道路1K接口重放最大RGB差5.96×10⁻⁸。实际外观求解约62秒；没有启动长程神经训练。AutoDL保持开机。</p></section>
<section><h2>资产与可复核记录</h2><p><a href="assets/complete_a6_fitted.npz">完整资产NPZ（几何、颜色、来源标记）</a> · <a href="assets/complete_a6_fitted.ply">标准SH0高斯PLY</a> · <a href="source/full_pvd.py">独立资产加载／渲染接口</a> · <a href="summary.json">全部固定指标</a> · <a href="fixed_metrics.csv">指标CSV</a> · <a href="quality_decision.json">验收状态</a> · <a href="old_work_closeout/index.html">旧分支收尾</a></p><p><a href="evidence/contract.json">预定合同</a> · <a href="evidence/geometry_provenance.json">完整PVD来源</a> · <a href="evidence/appearance_fit_receipt.json">外观求解</a> · <a href="evidence/all49_metrics.json.gz">49帧逐项指标</a></p><details><summary>观测口径与限制</summary><ul>'''+limits+'''</ul></details><p class="small">相关工作：<a href="https://research.nvidia.com/labs/sil/projects/axolotl3d/">Axolotl3D</a>使用图像、可见性、相机与部分点云约束完整形状；<a href="https://gaussianobject.github.io/">GaussianObject</a>结合稀疏视图、几何先验与生成式修复细化对象。这里的待验证贡献落在街景对象一致性、观测保持与编辑可用性。</p></section></main>
<script>function setRaw(v){v=Number(v);document.getElementById('slider').value=v;document.getElementById('rawlabel').textContent='raw'+v;let r=String(v).padStart(3,'0');for(let k of ['objects','street','groups'])document.getElementById(k).src='frames/'+r+'_'+k+'.jpg'}document.getElementById('slider').oninput=e=>setRaw(e.target.value);</script></html>'''
    html=html.replace('首轮完整提案对照用于定位固定PVD的表达限制。','首轮完整提案对照未通过已观测质量验收。原生资产有100,358颗高斯、三阶SH，新资产只有2,048颗高斯、SH0，因此当前差距同时包含空间分辨率、表面对齐与外观容量因素，不能唯一归因于PVD几何。完整核心覆盖已达0.9852，内部缺覆盖也不是充分解释。')
    html=html.replace('先区分固定几何分辨率不足与表面对齐误差，再单独开放实测约束下的受限几何细化。','依次检验固定几何下的外观容量、统一表面的采样分辨率，最后单独开放实测约束下的受限几何校正。每一步保留独立对照。')
    html=html.replace('<p>3帧真实GPU smoke','<p><a href="evidence/geometry_support_audit.json">观测支持／固定设计审计</a> · <a href="evidence/capacity_audit.json">原生与新资产容量审计</a> · <a href="evidence/next_unified_asset_contract.json">下一阶段单一资产实验合同（设计，未运行）</a></p><p>原生a6对象层的100,358行已全部退出新场景输入。冻结的旧静态层仍可能有残留贡献，本轮没有宣称解决所有外部静态遮挡。</p><p>3帧真实GPU smoke')
    (dest/'index.html').write_text(html,encoding='utf-8')
    (dest/'assets/README.txt').write_text('Canonical object metres: +x forward, +y left, +z up. NPZ contains one complete 2048-row SH0 Gaussian asset, no native a6 inference dependency. Apply the supplied actor trajectory once before joint rasterization. Full geometry is a proposal, not verified full ground truth. RGB marked colour_inferred has weak/no direct evidence. PLY uses standard Gaussian splatting opacity logits, log-scales, wxyz quaternion and SH0 DC; it contains the same single asset. full_pvd.py FixedAsset.from_geometry(path).rows(model_from_local,camera,pipe,channels,precomputed=False).\n',encoding='utf-8')
    print(json.dumps(dict(delivery=str(dest),quality=quality),ensure_ascii=False))

if __name__=='__main__':main()
