"""Archive verified physical-layer controls and preserve other repository work."""
import gzip
import json
from pathlib import Path
import shutil

RUN=Path(__file__).resolve().parent.parent
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
OUT=REPO/'docs/autoresearch/scene008_a6_layered_handoff_20261105'

def read(p):return json.loads(p.read_text())

def main():
    summary=read(RUN/'summary.json');verification=read(RUN/'verification.json')
    assert verification['exact_saved_metric_replay'] and len(verification['checked'])==324
    assert (RUN/'delivery/index.html').is_file()
    OUT.mkdir(exist_ok=False);(OUT/'source').mkdir()
    for p in (RUN/'source').glob('*.py'):
        shutil.copyfile(p,OUT/'source'/p.name)
    for p in RUN.glob('*.npz'):shutil.copyfile(p,OUT/p.name)
    for p in RUN.glob('*.json'):
        if p.name.endswith('_evaluation.json') or p.name in ('retired_depth_audit.json','smoke.json'):
            with gzip.open(OUT/(p.name+'.gz'),'wb') as f:f.write(p.read_bytes())
        else:shutil.copyfile(p,OUT/p.name)
    for name in ('architecture.svg','fixed_metrics.csv'):
        shutil.copyfile(RUN/'delivery'/name,OUT/name)
    shutil.copyfile(RUN/'delivery.tar.gz.sha256',OUT/'delivery.tar.gz.sha256')
    rows=[];applicable=0
    for group in range(3):
        for layer in ('near','same','far'):
            r=read(RUN/f'{layer}_group{group}_evaluation.json')
            fit=read(RUN/f'{layer}_group{group}_fit.json')['choice']
            applicable+=fit['reason']!='NO_LOST_GROUP_SUPPORT'
            h=r['pools']['held']['black']
            p,c=h['parent']['group'],h['coupled']['group']
            rows.append(f'|{group+1}|{layer}|{p["rgb"]["mean"]:.6f}->{c["rgb"]["mean"]:.6f}|'
                f'{p["depth"]["mean"]:.5f}->{c["depth"]["mean"]:.5f}|{c["coverage_error"]["mean"]:.6f}|'
                f'{fit["reason"]}|{r["method_accepted"]}|')
    n=len(summary['accepted'])
    text=f'''# Scene008：近/同/远物理支撑对照

固定三组实测面，对旧原生a6做近/同/远法向支撑对照，9个case全部按合同执行，{applicable}个存在可归因退役量，{n}个通过全部方法验收。324个保存状态指标精确重放，另有81条实际选中状态的重载验证。最终资产选择见summary与各selected.npz，失败case明确保存零操作。

![组件与数据流](architecture.svg)

本轮接续[薄壳/贡献登记失败](../scene008_a6_coupled_handoff_20261105/README.md)。保留7/5/5个实测代理面、755/175/310个登记原生行，固定几何/颜色/尺度、PVD、道路和轨迹。物理区域沿固定车体参考法向分为camera侧>12cm、同面±12cm、反侧>12cm；完整normal范围来自原车框每边+.28m的8角投影，原tangent盒与测量分母不变。BUILD相机侧必须一致。精确条件质量crop只作用于登记a6支撑，不整行退役。

物理法向层与原生Gaussian中心排序深度是不同概念；本轮不改变排序或实现不透明首交。真实GPU smoke核对三层之和等于完整物理柱，最大误差约1e-7，零操作所有输出通道与父资产精确一致。

每层四臂：parent、retirement only、proxy only、coupled。退役.5固定，仅一个BUILD标量加入系数，在0至0.35内二分10步匹配组贡献预算。没有额外阈值扫描，没有神经训练。零责任层NO_LOST_GROUP_SUPPORT属于不适用，不作为反证。

BUILD40帧，内部留出9帧89/95/101/105/113/121/125/130/137。分母与上一轮一致：BUILD/留出141/26、49/9、65/6。系数拟合不用留出；留出veto参与最终candidate/zero操作选择。留出已多轮诊断，不是未触碰独立测试或泛化证据。

严格parent目标RGB不恶化、深度改善>1e-5m只在black验证；相对同退役分支要求RGB/深度改善、组贡献绝对误差<=.01。全49帧black/keep/delete继承保护。额外非目标深度保真仅覆盖group外固定a6测点，各帧各状态mean absolute change<=.005m；其他对象/道路通过继承RGB保护，不宣称全场测量深度保真。

|组|层|留出RGB parent->coupled|留出深度m parent->coupled|贡献绝对误差|退役预算状态|接受|
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

retired_depth_audit.json.gz记录BUILD父transmittance下的first-order pre-retirement alpha*T归因，按实际合成中心depth相对测量±12cm重新分类；校验等于.5倍parent层native coverage。不等同于退役后真实alpha*T差，不是隐藏几何真值。完整观察与后续设计见交互报告。

## 入口与复现

目录：{RUN}

Python /root/autodl-tmp/envs/drivingeditor/bin/python；Torch2.2.2/cu118；RTX4080SUPER。

source/smoke_layered.py -> source/run_layered.py -> source/verify_layered.py -> source/retired_depth_audit.py -> source/make_delivery.py。父规则依赖scene008_a6_registered_handoff_20261105与coupled/heuristic/depth_order等保留目录，输入观测与模型未覆盖。

本地：outputs/scene008_layered_handoff/delivery/index.html；远端：{RUN}/delivery/index.html。包含9case四臂黑底/街景、全部5292行指标、几何登记、选择状态NPZ与源码。原始字节由SHA256包和MANIFEST保存；Git文本规范化LF，不改变数值/二进制。AutoDL保持开机，子代理均sol6.1/max。
'''
    (OUT/'README.md').write_text(text,encoding='utf-8')
    entry=f'2026-11-05，Scene008完成近/同/远物理法向支撑9个对照，{n}个通过全部方法验收；324个保存指标重放及81条实际选择状态重载。物理支撑位置与原生中心排序深度需区分。AutoDL保持开机。[实验与代码](autoresearch/scene008_a6_layered_handoff_20261105/README.md)。\n\n'
    backups=RUN/'git_backups';backups.mkdir()
    for name in ('RESEARCH_STATUS.md','EXPERIMENTS.md'):
        p=REPO/'docs'/name;shutil.copyfile(p,backups/name)
        original=p.read_text(encoding='utf-8')
        if name=='RESEARCH_STATUS.md':
            header,body=original.split('\n',1);new=header+'\n\n'+entry+body.lstrip('\n')
        else:new=original.rstrip()+'\n\n'+entry
        p.write_text(new.rstrip()+'\n',encoding='utf-8')
    for p in OUT.rglob('*'):
        if p.is_file() and p.suffix in ('.py','.md','.json','.csv','.svg','.sha256'):
            p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n').rstrip()+b'\n')
    print(OUT)

if __name__=='__main__':main()
