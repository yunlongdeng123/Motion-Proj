"""Package measured asset controls and the separate inherited road-tint diagnosis."""
import argparse,csv,gzip,json,shutil,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

RUN=Path(__file__).resolve().parent.parent;PARENT=RUN.parent/'scene008_a6_complete_asset_20261105'
def read(path):return json.loads(path.read_text())
def write(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def img(a):
    if a.ndim==3 and a.shape[0] in (1,3):a=a.transpose(1,2,0)
    return Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5))
def strip(values,labels,crop,height=240):
    tiles=[]
    for a,label in zip(values,labels):
        p=img(a).crop(crop);p=p.resize((round(p.width*height/p.height),height),Image.Resampling.NEAREST)
        t=Image.new('RGB',(p.width,height+31),(242,247,249));t.paste(p,(0,31));ImageDraw.Draw(t).text((7,8),label,fill=(24,42,52));tiles.append(t)
    result=Image.new('RGB',(sum(p.width for p in tiles),height+31));left=0
    for p in tiles:result.paste(p,(left,0));left+=p.width
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='delivery');args=parser.parse_args()
    dest=RUN/args.output;dest.mkdir(exist_ok=False);(dest/'frames').mkdir();(dest/'assets').mkdir();(dest/'evidence').mkdir()
    summaries={arm:read(RUN/arm/'evaluation/summary.json') for arm in ('m1','m2','m3')}
    original=read(PARENT/'delivery/summary.json');evaluations={arm:read(RUN/arm/'evaluation/metrics.json') for arm in summaries}
    assert all(v['all49_complete'] for v in evaluations.values())
    for raw in range(89,138):
        with np.load(PARENT/'evaluation/frames'/f'{raw:03d}_replay.npz') as old:
            native=old['native_solo'].astype(float);m0=old['asset_solo'].astype(float);alpha=old['alpha'][0].astype(float)
        values=[]
        for arm in summaries:
            with np.load(RUN/arm/'evaluation/frames'/f'{raw:03d}_replay.npz') as saved:
                values.append(saved['solo'].astype(float));alpha=np.maximum(alpha,saved['alpha'][0])
            shutil.copyfile(RUN/arm/'evaluation/frames'/f'{raw:03d}_street.jpg',dest/'frames'/f'{raw:03d}_{arm}_street.jpg')
        with np.load(RUN.parent/'scene008_instances_20261105_r1/observations/frames'/f'{raw:03d}.npz') as obs:
            rgb=obs['rgb'].astype(float)
            if rgb.max()>1:rgb=rgb/255.
            assert .1<float(rgb.max())<=1.,'Daylight Scene008 display must use correctly normalized RGB'
            mask=obs['visible'][0];truth=rgb*mask[:,:,None]
        support=(alpha>.01)|(native.max(0)>.01)|mask;y,x=np.where(support)
        crop=(max(0,int(x.min())-10),max(0,int(y.min())-10),min(480,int(x.max())+11),min(320,int(y.max())+11))
        strip([truth,native,m0]+values,['Observed RGB','Native a6','M0 2048 SH0','M1 2048 SH3','M2 18432 SH0','M3 measured SH0'],crop).save(dest/'frames'/f'{raw:03d}_objects.jpg',quality=95)
    for arm,filename in [('m1','complete_a6_SH3.npz'),('m2','complete_a6.npz'),('m3','complete_a6.npz')]:
        shutil.copyfile(RUN/arm/filename,dest/'assets'/f'{arm}_complete_a6.npz')
        for name in ('contract.json','collection.json','fit_receipt.json','smoke/receipt.json','evaluation/summary.json'):
            if (RUN/arm/name).exists():shutil.copyfile(RUN/arm/name,dest/'evidence'/(f'{arm}_'+name.replace('/','_')))
        with gzip.open(dest/'evidence'/f'{arm}_all49_metrics.json.gz','wt',encoding='utf-8') as f:json.dump(evaluations[arm],f)
    for name in ('surface_sampling_audit.json','measured_correction_audit.json','measured_correction_contract.json','operator_audit.json'):
        shutil.copyfile(RUN/name,dest/'evidence'/name)
    for name in ('sh_asset_cpu_test.json','m3_review.json'):
        if (RUN/'source'/name).exists():shutil.copyfile(RUN/'source'/name,dest/'evidence'/name)
    shutil.copytree(RUN/'source',dest/'source',ignore=shutil.ignore_patterns('__pycache__','*.bak*'))
    shutil.copyfile(PARENT/'source/full_pvd.py',dest/'source/full_pvd.py')
    shutil.copyfile(RUN/'source/architecture.svg',dest/'architecture.svg')
    tint=dest/'tint';tint.mkdir()
    for pattern in ('*_controls.png','metrics.json','findings.json'):
        for path in (RUN/'tint_diagnostic').glob(pattern):shutil.copyfile(path,tint/path.name)
    write(dest/'summary.json',summaries)
    rows=[]
    for split in ('all49','HELD'):
        parent=original['all49' if split=='all49' else 'internal_HELD9']
        for name,prefix in [('Native','solo_native'),('M0','solo_fitted')]:
            rows.append(dict(split=split,arm=name,RGB=parent[prefix+'.a6.rgb_error']['mean'],
                depth_m=parent[prefix+'.a6.measured_depth_error']['mean'],coverage=parent[prefix+'.a6.visible_coverage']['mean']))
        for arm,summary in summaries.items():
            data=summary[split];rows.append(dict(split=split,arm=arm.upper(),RGB=data['solo_new.a6.rgb_error']['mean'],
               depth_m=data['solo_new.a6.measured_depth_error']['mean'],coverage=data['solo_new.a6.visible_coverage']['mean']))
    with (dest/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    table=''.join('<tr>'+''.join('<td>'+ (f'{row[k]:.5f}' if isinstance(row[k],float) else str(row[k]))+'</td>' for k in ('split','arm','RGB','depth_m','coverage'))+'</tr>' for row in rows)
    preservation=''
    for arm,summary in summaries.items():
        for state in ('keep','delete'):
            for region in ('a37','road_visible','outside'):
                key=f'incremental_scene.{state}.preservation.{region}.parent_rgb_change'
                if key not in summary['all49']:continue
                v=summary['all49'][key];preservation+=f'<tr><td>{arm}</td><td>{state}</td><td>{region}</td><td>{v["mean"]:.7f}</td><td>{v["above_005"]}</td></tr>'
    correction=read(RUN/'measured_correction_audit.json');fit_seconds=sum(read(RUN/a/'fit_receipt.json')['seconds'] for a in summaries)
    decision=dict(scientific_quality_accepted=False,reason='Observed RGB still worse than native; hidden texture has no independent truth; local correction limited support',
       latest_results=rows,background_frozen=True,road_tint_diagnosis='Inherited road-layer seam; broad cool tone already present in original/native; three-frame diagnostic only',
       candidate_assets_are_experimental=True,formal_appearance_steps_per_arm=2000,extra_failed_smoke_and_QC_recorded=True,
       next='Assess reliable observed object surface construction, tangential alignment and material support; no part-specific clipping or automatic threshold expansion')
    write(dest/'quality_decision.json',decision)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scene008 完整资产：材质、采样与实测校正</title><style>body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#f5f8fa;color:#193b4a}main{max-width:1500px;margin:auto;padding:26px}section{background:white;border-radius:10px;padding:22px;margin:20px 0}h1{font-size:29px}h2{margin-top:30px}img{width:100%;display:block}.scroll{overflow:auto}.scroll img{min-width:1600px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:7px;border-bottom:1px solid #d8e3e9;text-align:left}th{background:#eef4f7}a{color:#006b83}button,select{padding:7px 13px;margin:4px}input{width:50%;vertical-align:middle}.note{background:#fff3dc;border-left:4px solid #ce9531;padding:13px}.small{font-size:14px}</style><main>
<h1>Scene008：完整 a6 的材质、采样与实测校正</h1><p>所有对照始终只使用一份完整资产；原生 a6 只提供观测关联和可靠外观参考，退出新场景成像。车辆轨迹、a37、旧静态、环境与已有道路恢复全部冻结。没有按下半部、左右或上部设置接管规则。</p>
<section><img src="architecture.svg" alt="架构组件与三组单因素对照"><p class="note">这是机制验证，尚未达到原生已观测外观质量。M1只改视角外观容量；M2保留SH0、改变全车高斯离散；M3在M2相同点数和材质容量上，只增加实测约束的中心校正。各组独立保存与完整49帧评价。仍使用原生高斯合成，没有MOG-2不透明表面首交保证。</p></section>
<section><h2>同一裁剪的黑底对象</h2><p>依次为真实可见RGB、原生a6、M0完整2048点SH0、M1相同几何SH3、M2完整18432点SH0、M3实测中心校正SH0。未知部位的低容量颜色仍为推断外观。</p><div><b id="raw">raw113</b> <input id="range" type="range" min="89" max="137" value="113"><button onclick="show(105)">105</button><button onclick="show(113)">113</button><button onclick="show(130)">130</button></div><div class="scroll"><img id="objects" src="frames/113_objects.jpg" alt="六列单一对象对照"></div><h2>同一资产在场 / 删除配对</h2><select id="arm"><option value="m1">M1 材质对照：参考为M0</option><option value="m2" selected>M2 采样对照：参考为M0</option><option value="m3">M3 实测校正：参考为M2</option></select><p>左两列为参考/新资产且a37在场，右两列为参考/新资产且删除a37。M3与M2直接配对，避免把采样收益误算为几何校正收益。</p><div class="scroll"><img id="street" src="frames/113_m2_street.jpg" alt="街景统一合成保留删除对照"></div></section>
<section><h2>固定已观测分母</h2><p>all49使用39308个原始可见核心像素和8930个LiDAR深度测量；HELD9为新增外观/校正步骤的内部留出。RGB为sRGB通道平均绝对误差，深度为渲染alpha加权camera-z误差（米），覆盖为对象alpha。没有按新模型alpha缩小评价域。</p><table><tr><th>范围</th><th>资产</th><th>RGB MAE</th><th>深度 MAE / m</th><th>可见核心覆盖</th></tr>'''+table+'''</table><p>M1的alpha与depth_sum相对M0逐像素完全相同。M2确实改变中心离散、尺度、透明度及合成深度，不能叫逐像素几何等价；它的收益不自动证明PVD真实形状更准确。M3仅改变有可靠多帧支持的局部中心；其拟合残差降低不能代替渲染评价。</p><p>BUILD深度7311点、内部HELD深度1619点。父级PVD和native已有同场景信息；200个PVD条件点含14个来自HELD帧。内部留出不能声称独立几何真值、隐藏表面准确率或跨场景泛化。</p></section>
<section><h2>实测约束到底改了什么</h2><p>只读取BUILD40确认可见的原始LiDAR样本；通过跨帧稳健平面、交替帧互证、法向一致性与时序一致性后，沿固定局部法向施加有界位移。未知中心位移严格为零，normal、scale、opacity、点数、SH0容量及颜色拟合起点保持一致。支持半径依据采样间距，用一套全车规则。</p><p>原始可见标签允许20cm关联容差，因此又做同源投影/深度一致核对。该核对使用同一LiDAR来源，不是独立精度证据。工程容差2cm不等于已校准的传感器噪声。<a href="evidence/measured_correction_audit.json">实际支持、位移与不变量</a> · <a href="evidence/measured_correction_contract.json">固定合同</a></p></section>
<section><h2>保留域变化与尾部</h2><p>相对各组直接参考的RGB变化。M1/M2参考M0，M3参考M2；均值很小也不能掩盖少量大变化。outside可能包含完整资产新增显露的区域，变化不自动等同于其他真实表面受损，需结合可见a37、道路与原图判断。</p><table><tr><th>对照</th><th>状态</th><th>固定区域</th><th>平均RGB变化</th><th>变化&gt;0.05像素数</th></tr>'''+preservation+'''</table></section>
<section><h2>路面色块与全图偏蓝的单独诊断</h2><p><strong>路面割裂主要来自恢复道路插入后与原生路面的颜色衔接。</strong>在raw105/113/130，关闭后加的世界道路颜色场，接缝仍在；颜色场反而略减边界跳变。完整a6替换对该边界梯度指标影响为零。颜色场在road贡献&lt;1e−6之外的最大RGB影响≤2.98e−8，因此不能解释全图偏蓝。</p><p><strong>原始RGB本身就偏冷，native背景也延续了这种色调。</strong>三帧原图蓝指数B−(R+G)/2为0.0756–0.0789，native为0.0785–0.0811。环境项是(1−alpha)×方向背景，主要在低覆盖背景中贡献冷色；它不是给不透明车体或整张图施加蓝色滤镜。均值含天空，这些数字不足以认定整段视频白平衡错误。</p><p>颜色衔接的更早根因还没有唯一拆分为道路参考纹理、阴影监督、静态gate或方向背景拟合。车辆对照继续冻结道路；这次没有通过全图调色掩盖接缝。</p><select id="tintRaw"><option>105</option><option selected>113</option><option>130</option></select><select id="tintState"><option value="delete" selected>删除a37</option><option value="keep">保留a37</option></select><img id="tint" src="tint/113_delete_controls.png" alt="原图、原生、道路与颜色场消融及环境贡献"><p><a href="tint/findings.json">归因、逐帧数字、源码行与限制</a> · <a href="tint/metrics.json">完整诊断指标</a></p></section>
<section><h2>复核与资产</h2><p>每组正式外观优化2000步，额外smoke与工程重放单列。真实GPU smoke覆盖保存、完整状态恢复后的下一步逐值相等、资产重载、联合合成和下一阶段入口。第一次SH3 smoke发现通用稀疏GPU归约不确定，改为固定树FP64积累后通过；M2 smoke发现旧Runtime导入遮蔽，在正式拟合前修复。正式数值合同与预算保持不变。</p><p><a href="evidence/operator_audit.json">独立SciPy/伴随/确定性/原SH0求解器核对</a> · <a href="comparison.csv">比较表CSV</a> · <a href="summary.json">完整汇总</a> · <a href="quality_decision.json">验收状态</a> · <a href="assets/m1_complete_a6.npz">M1资产</a> · <a href="assets/m2_complete_a6.npz">M2资产</a> · <a href="assets/m3_complete_a6.npz">M3资产</a></p><p class="small">独立加载接口为source/sh_asset.py；同目录full_pvd.py只提供几何字段、序列化与哈希。资产不依赖原生a6成像。原场景外部残留静态层仍保留，没有宣称全部静态遮挡已解决。AutoDL保持开机。</p></section></main>
<script>function show(r){r=Number(r);document.getElementById('range').value=r;document.getElementById('raw').textContent='raw'+r;let a=document.getElementById('arm').value;document.getElementById('objects').src='frames/'+r+'_objects.jpg';document.getElementById('street').src='frames/'+r+'_'+a+'_street.jpg'}document.getElementById('range').oninput=e=>show(e.target.value);document.getElementById('arm').onchange=()=>show(document.getElementById('range').value);function tint(){document.getElementById('tint').src='tint/'+document.getElementById('tintRaw').value+'_'+document.getElementById('tintState').value+'_controls.png'}document.getElementById('tintRaw').onchange=tint;document.getElementById('tintState').onchange=tint;</script></html>'''
    # Include the actual localized correction count without inventing an expanded support.
    paragraph='<p>实际通过校正的中心 '+str(correction['eligible_centers'])+' / '+str(correction['rows'])+'；未知区保持不动。完整数值见审计。</p>'
    html=html.replace('<h2>实测约束到底改了什么</h2>','<h2>实测约束到底改了什么</h2>'+paragraph)
    html=html.replace('支持半径依据采样间距，用一套全车规则。','支持半径依据采样间距，用一套全车规则。通过支持检查的中心是至多25cm邻域平面的插值或有限外推，不是这些中心本身被真实直接观测的证书；本对照不能校正切向误差、法向、拓扑或未观测整车。')
    html=html.replace('<h2>固定已观测分母</h2>','<h2>固定已观测分母</h2><p>M1受约束SH3只小幅改善；M2统一细采样的RGB与深度收益更明显。M3本轮234点实测法向校正没有继续改善RGB，不放宽门槛追逐收益。当前优先保留M2作后续单一资产研究基线，所有组仍作为实验资产保存。</p>')
    html=html.replace("let a=document.getElementById('arm').value;", "let a=document.getElementById('arm').value;let filename=String(r).padStart(3,'0');")
    html=html.replace("'frames/'+r+'_objects.jpg'", "'frames/'+filename+'_objects.jpg'")
    html=html.replace("'frames/'+r+'_'+a+'_street.jpg'", "'frames/'+filename+'_'+a+'_street.jpg'")
    (dest/'index.html').write_text(html,encoding='utf-8')
    print(json.dumps(dict(delivery=str(dest),results=rows),ensure_ascii=False))

if __name__=='__main__':main()
