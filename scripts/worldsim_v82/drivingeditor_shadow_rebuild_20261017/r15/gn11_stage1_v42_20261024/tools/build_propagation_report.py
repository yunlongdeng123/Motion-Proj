"""Build a self-contained 49-frame CPU audit report; overlays are NOT renders."""
import argparse,json,html,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ARCH='''<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="350" viewBox="0 0 1320 350"><defs><marker id="a" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0 0L9 3L0 6" fill="none" stroke="#526377"/></marker></defs><rect width="1320" height="350" fill="#f5f8fc"/><g font-family="Arial,Microsoft YaHei" font-size="20" text-anchor="middle" fill="#16324e"><rect x="20" y="20" width="280" height="70" rx="10" fill="#dcecdf"/><text x="160" y="62">原 V1 主副本</text><rect x="340" y="20" width="310" height="70" rx="10" fill="#dcecdf"/><text x="495" y="62">原生完整重建输出</text><text x="950" y="62">完整通路不加载本轮候选或编辑状态</text><rect x="20" y="165" width="230" height="85" rx="10" fill="#e0e9f7"/><text x="135" y="200">49 帧现有证据</text><text x="135" y="229">RGB / LiDAR / 道路种子</text><rect x="290" y="165" width="250" height="85" rx="10" fill="#e0e9f7"/><text x="415" y="200">平面传播与射线求交</text><text x="415" y="229">生成新位置 / 法线</text><rect x="580" y="165" width="290" height="85" rx="10" fill="#e0e9f7"/><text x="725" y="200">可见来源几何 + 色块核验</text><text x="725" y="229">另组来源验证；不取车身 RGB</text><rect x="910" y="165" width="380" height="85" rx="10" fill="#e0e9f7"/><text x="1100" y="200">候选点文件 + 固定区域覆盖报告</text><text x="1100" y="229">仅供独立编辑副本；尚未训练 / 渲染</text><text x="660" y="310" font-size="18">无观测的候选保留几何先验身份；不能当作真实颜色监督或成功补全</text></g><g stroke="#526377" stroke-width="2" marker-end="url(#a)"><path d="M300 55H340"/><path d="M250 207H290"/><path d="M540 207H580"/><path d="M870 207H910"/></g></svg>'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'images').mkdir(exist_ok=True)
    s=json.loads((a.results/'summary.json').read_text());meta=json.loads((a.inputs/'manifest.json').read_text())
    (a.output/'architecture.svg').write_text(ARCH,encoding='utf8');shutil.copy2(a.results/'summary.json',a.output/'summary.json');shutil.copy2(a.results/'protocol.json',a.output/'protocol.json')
    methods=['legacy','fixed_same_verifier','image_plane','metric_plane','measured_refined'];names=['V4.1 原标签','固定曲面 · 同核验','图像邻近平面 · CPU 对照','米制道路平面 · CPU 候选','可见来源深度修正 · CPU 候选']
    detail=json.loads((a.results/'audit_detail.json').read_text());shutil.copy2(a.results/'audit_detail.json',a.output/'audit_detail.json');shutil.copy2(a.results/'critical_hypotheses.json',a.output/'critical_hypotheses.json')
    for fid in meta['frame_order']:
        f=dict(np.load(a.inputs/f'{fid}.npz'));z=dict(np.load(a.results/'frames'/f'{fid}.npz'));pix=z['pixels'];rgb=np.clip(f['rgb']*255,0,255)
        for name in methods:
            canvas=rgb.copy();mask=z[name];p=pix[mask];canvas[p[:,1],p[:,0]]=.4*canvas[p[:,1],p[:,0]]+.6*np.array([0,230,130])
            if name!='legacy':
                p=pix[z[name]&~z[name+'_validation']];canvas[p[:,1],p[:,0]]=.4*rgb[p[:,1],p[:,0]]+.6*np.array([245,185,40])
                p=pix[z[name+'_validation']&~z['supported']];canvas[p[:,1],p[:,0]]=.25*rgb[p[:,1],p[:,0]]+.75*np.array([245,80,220])
            im=Image.fromarray(canvas.astype('uint8'))
            if fid=='0008102':
                d=ImageDraw.Draw(im)
                for x,y in [(255,240),(256,219)]:d.ellipse((x-3,y-3,x+3,y+3),outline='yellow',width=1)
            im.save(a.output/'images'/f'{fid}_{name}.webp',quality=90)
    head='<tr><th>区域</th><th>完整分母</th><th>原标签</th><th>固定曲面同核验</th><th>图像平面</th><th>米制平面</th><th>米制相对固定新增 / 原无支撑</th></tr>'
    rows=''
    def num(v,d):return f'{v:,} ({100*v/max(d,1):.2f}%)'
    for act,r in s['actors'].items():
        rows+=f'<tr><td>a{act}</td><td>{r["denominator"]:,}</td><td>{num(r["legacy"],r["denominator"])}</td><td>{num(r["fixed_same_verifier"],r["denominator"])}</td><td>{num(r["image_plane"],r["denominator"])}</td><td>{num(r["metric_plane"],r["denominator"])}</td><td>{r["metric_added_over_fixed"]:,} / {r["metric_added_unsupported"]:,}</td></tr>'
    validation='<tr><th>区域</th><th>固定：另组验证通过</th><th>图像平面：验证通过</th><th>米制平面：验证通过</th><th>实测修正：拟合 / 另组可验证 / 通过</th><th>修正新增旧标签外 / 原无支撑</th></tr>'
    for act,r in detail['actors'].items():
        v=r['measured_refined'];validation+=f'<tr><td>a{act}</td><td>{r["fixed_same_verifier"]["validation_pass"]:,}</td><td>{r["image_plane"]["validation_pass"]:,}</td><td>{r["metric_plane"]["validation_pass"]:,}</td><td>{v["fit"]:,} / {v["testable"]:,} / {v["validation_pass"]:,}</td><td>{v["validated_added_vs_legacy"]:,} / {v["validated_unsupported"]:,}</td></tr>'
    critical=''.join(f'<p><b>raw102 {r["pixel"]}，a{r["actor_id"]}</b>：'+ '；'.join(f'{names[methods.index(k)]}：几何提议 {v["proposal_exists"]}，拟合通过 {v["fit_accepted"]}，另组验证 {v["validated"]}' for k,v in r['methods'].items())+'</p>' for r in detail['critical'])
    intro=(a.results/'interpretation.html').read_text(encoding='utf8') if (a.results/'interpretation.html').exists() else '<p>以下为 CPU 提议与核验统计，未测量最终道路贡献或渲染质量。</p>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V4.2 · GaussianPro 机制 CPU 检验</title><style>body{font:17px/1.7 system-ui,"Microsoft YaHei";margin:0;background:#f3f6fa;color:#18314e}main{max-width:1300px;margin:auto;padding:30px}h1{font-size:34px}section{background:white;border:1px solid #cdd8e4;border-radius:12px;padding:24px;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border:1px solid #d4dee8;padding:9px;text-align:left}th{background:#eef4fb}select,button{font:inherit;padding:8px;margin:5px}img{max-width:100%}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.note{background:#fff2d8;padding:15px;border-radius:8px}.muted{color:#5e6e83}pre{white-space:pre-wrap}a{color:#1663a8}</style><main><h1>V4.2：平面传播能否补到失败区域？</h1>'''+intro+f'''<p class="muted">同49帧；CPU；优化更新0；quality_accepted=false。结果不是新训练的渲染图。</p><section><img src="architecture.svg" alt="V42完整通路与CPU编辑候选组件图"></section><section><h2>机制与公平对照</h2><p>从已有可靠道路 Gaussian 的位置、法线提出局部平面，用目标射线求交产生新位置。图像邻近平面与米制邻近平面各两个候选，使用完全相同的核验器。固定曲面对照也使用这一核验器，但保留原有几何支撑范围、不沿用1919点资格限制。</p><p>核验使用真实地面小三角形深度、2像素/1%回投检查和可见来源平面色块；目标车身RGB不参与。拟合使用交替5帧块中的两份来源；另组来源只作验证，不选平面。低纹理不能靠NCC确认深度，单列其对实测几何的依赖。</p><p class="note">这实现了 GaussianPro 的“平面假设→色块选择→几何核验→新位置”CPU机制对照，未复现官方 CUDA ACMH 搜索、训练内增密及平面损失。因此不能写成“已经超过 GaussianPro”。</p></section><section><h2>五车固定轮廓：49帧累计像素</h2><table>{head+rows}</table><p>分母保持整个原车辆轮廓；统计不是世界道路面积或最终补全率。原标签与新核验强度不同，不能仅凭二者大小归因；算法增益重点看同核验固定曲面、图像平面、米制平面的差异。另设实测修正：只用拟合来源的地面深度修正平面两轮，每轮法向位移≤0.1m、累计≤0.2m，最终核验门限不变；这一结果保留为单独方法，不覆盖首轮负结果。</p><table>{validation}</table><p>另组验证是本开发集内一致性检查，种子几何仍来源于全49帧，不是独立测试集。</p></section><section><h2>关键失败像素</h2>{critical}<p>无观测的中央空洞不能因传播出了几何位置，就计为真实道路颜色监督；普通静态污染也不会被新增点自动清除。</p></section><section><h2>49帧实际覆盖图</h2><p>绿色：拟合及另组颜色验证通过；黄色：仅拟合通过但缺另组确认或验证失败；紫色：另组验证通过且原本无曲面支撑。旧V4.1图的绿色沿用原标签，规则不同。底图是原始有车RGB；车身位置的色块表示删车后可获得其它帧路面参考，不是拿车身颜色当道路标签。黄色圆圈：指定失败点。没有着色表示未核验通过。</p><button id="prev">上一帧</button><select id="frame"></select><button id="next">下一帧</button><div class="grid">'''
    for name,label in zip(methods,names):page+=f'<div><h3>{label}</h3><img id="{name}" alt="{label}覆盖图"></div>'
    page+='''</div><pre id="stats"></pre></section><section><h2>代码、依据与复核入口</h2><p><a href="summary.json">逐帧与逐车完整统计</a> · <a href="protocol.json">运行前固定协议</a> · <a href="audit_detail.json">验证、关键点与最差帧</a> · <a href="critical_hypotheses.json">两个失败点的所有假设</a> · <a href="https://arxiv.org/abs/2402.14650">GaussianPro 论文</a> · <a href="https://github.com/kcheng1021/GaussianPro/tree/version1.0">官方实现</a></p><p>候选点XYZ/法线/来源/RGB另存 refined_validated_birth_proposals.npz，未写入V1或原R checkpoint。无GPU训练、无渲染质量结论；a6和中心残影需按真实结果继续处理。</p></section></main><script>'''
    page+='const data='+json.dumps(s,ensure_ascii=True)+';const frames='+json.dumps(meta['frame_order'])+';const methods='+json.dumps(methods)+';'
    page+='''const sel=document.getElementById('frame');frames.forEach((f,i)=>{let o=new Option('raw'+Number(f.slice(-3)),i);sel.add(o)});sel.value=13;function show(){let i=Number(sel.value),f=frames[i];methods.forEach(m=>document.getElementById(m).src='images/'+f+'_'+m+'.webp');document.getElementById('stats').textContent=JSON.stringify(data.per_frame[i],null,2)}sel.onchange=show;document.getElementById('prev').onclick=()=>{sel.value=(Number(sel.value)+48)%49;show()};document.getElementById('next').onclick=()=>{sel.value=(Number(sel.value)+1)%49;show()};show();</script></html>'''
    (a.output/'index.html').write_text(page,encoding='utf8')
    # Verify every generated display file decodes; no browser/UI claim.
    for path in (a.output/'images').glob('*.webp'):
        with Image.open(path) as im:im.load();assert im.size==(480,320)
    print(json.dumps({'report':str(a.output/'index.html'),'decoded_images':len(list((a.output/'images').glob('*.webp')))}))
if __name__=='__main__':main()
