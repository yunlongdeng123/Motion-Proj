"""汇总已审核DGGT诊断与官方证据入口，不运行模型或重编码媒体。"""
from pathlib import Path
import argparse, html, json
from html.parser import HTMLParser
from urllib.parse import urlsplit
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args(); root=a.output_root.resolve()
    pair=root/'alpha_compositing_pair_r1'; layer=root/'deleted_layers_frame000_r1'
    runs=[json.loads((d/'run.json').read_text(encoding='utf-8')) for d in (pair,layer)]
    reviews=[json.loads((d/'assistant_review.json').read_text(encoding='utf-8')) for d in (pair,layer)]
    assert all(r['status']=='complete' for r in runs)
    assert all(r['human_verdict'] is None for r in reviews)
    values=runs[1]['layer_statistics']
    names={'static_before':'静态层 · 删除前','static_delete':'静态层 · 删除后','dynamic_before':'动态层 · 删除前','dynamic_delete':'动态层 · 删除后','joint_delete':'联合删除'}
    rows=''.join(f'<tr><td>{names[k]}</td><td>{v["hole_alpha_mean"]:.4f}</td><td>{v["hole_black_luma_mean"]:.4f}</td></tr>' for k,v in values.items())
    def fig(path,label):
        return f'<figure><img src="{html.escape(path)}" alt="{html.escape(label)}"><figcaption>{html.escape(label)}</figcaption></figure>'
    top=''.join(fig(path,label) for path,label in [
        ('alpha_compositing_pair_r1/assets/input/000.png','输入 · 目标车仍在'),
        ('alpha_compositing_pair_r1/assets/official/delete/000.png','原生删除 · 灰黑缺口'),
        ('deleted_layers_frame000_r1/assets/joint_delete/white.png','同一删除高斯 · 纯白背景')])
    layers=''.join(fig('deleted_layers_frame000_r1/assets/'+k+'/black.png',names[k]+' · 累计颜色G') for k in ('static_before','static_delete','dynamic_before','dynamic_delete'))
    doc='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DGGT · 编辑质量判断</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#0b1018;color:#e7edf5}body{max-width:1500px;margin:auto;padding:28px}h1{font-size:34px}p{color:#b8c7d7;line-height:1.7}a{color:#93d4ff}.card{background:#142131;border:1px solid #30445b;border-radius:14px;padding:20px;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.grid.two{grid-template-columns:repeat(2,minmax(0,1fr))}figure{margin:0}img{width:100%;display:block;border-radius:8px}figcaption{font-size:13px;color:#b4c5d6;padding:8px 0}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #30445b;text-align:left}.flow{display:flex;flex-wrap:wrap;gap:10px;align-items:center}.flow span{border:1px solid #456080;border-radius:8px;padding:10px}.verdict{border-left:4px solid #edbd73;padding-left:16px}@media(max-width:800px){.grid,.grid.two{grid-template-columns:1fr}}
</style></head><body><p>DGGT / Waymo scene128 / 2026-10-11 / 助手审核 · 人工 verdict 未填写</p><h1>重建可辨，不等于删除后有可信道路</h1>
<div class="card verdict"><strong>当前判断：DGGT 的公开证据不足以支持高保真车辆编辑主线。</strong><p>本地首帧分层支持“删除后可用背景覆盖不足”为优先问题；重复透明度衰减只造成小幅压暗。仍不能证明所有残余高斯都选对、背景从未被拍到，或所有前馈方法都受同一限制。官方公开演示也有明显残留，论文扩散后较好示例仍有模糊与暗痕。</p></div>
<nav><a href="index.html">原始四种编辑总览</a> · <a href="deleted_layers_frame000_r1/index.html">静态/动态/Alpha/深度完整页</a> · <a href="alpha_compositing_pair_r1/index.html">四帧透明度配对页</a> · <a href="reference_refinement_r1/index.html">既有参考修复对照</a></nav>
<div class="card flow"><span>RGB → 预测高斯与相机</span>→<span>固定原实例选择 → 删除</span>→<span>静态 / 动态 / 联合渲染</span>→<span>G、Alpha、深度与背景合成</span>→<span>配对与逐图审核</span></div>
<section class="card"><h2>白底暴露的是缺口</h2><div class="grid">TOP</div><p>白底使用同一 G 与 Alpha，仅将背景换成纯白；亮起来的空白不是模型恢复的道路。原生灰黑带不应直接按形状认定为“漏删的车”。下部仍保留暗片，其身份尚未证实。</p></section>
<section class="card"><h2>静态与动态层的删除前后</h2><div class="grid two">LAYERS</div><p>单独层是独立渲染，不能当成联合遮挡中的真实颜色贡献。固定洞域为首帧11,198像素；平均Alpha不是被覆盖像素比例。</p><table><tr><th>层</th><th>洞内平均 Alpha</th><th>黑底平均亮度（0–1）</th></tr>ROWS</table><p>静态层删除前后几乎不变，未显示连续车后道路。动态层中可辨车身删除后消失。联合删除仍然低覆盖；不把有数值的预测深度等同于真实道路几何。</p></section>
<section class="card"><h2>重复乘 Alpha：有颜色影响，但没修好编辑</h2><p>官方分支与旧结果及官方trace精确重放。gsplat颜色G已按透明度累积，外层再乘A会继续压暗；去掉后，四帧固定洞域平均亮度只增加3.51–4.09/255，灰黑带与不连续道路仍在。该诊断未替换官方权重/源码/旧输出。</p></section>
<section class="card"><h2>官方公开结果能说明什么</h2><p><a href="https://xiaomi-research.github.io/dggt/">官方项目页</a>与<a href="https://xiaomi-research.github.io/dggt/resources/video/edit2_1.mp4">原删除车辆视频</a>已直接查看；视频末帧仍有灰色弧顶和黑块。视频本身没有 raw/refined 标签，因此不能认定它是 Difix 最终结果。</p><p><a href="https://arxiv.org/html/2512.03004v1#S4.SS3">论文 v1 Figure 5</a>明确区分扩散前后；<a href="https://arxiv.org/html/2512.03004v1/images/3.3_.png">扩散后删除例</a>是已核对的较好公开样例，但红框道路仍软化、有局部暗痕。<a href="https://arxiv.org/html/2512.03004v1/images/3.2.png">同例修复前</a>近处已经有道路，不是从本地这种整车宽洞中补出道路。</p><p>论文重建/NVS的PSNR、SSIM、LPIPS不是删除洞的质量指标。已核对材料未提供编辑专属背景真值、删除成功率或时序一致性量化；无法给出官方编辑的可靠质量上限。CVPR最终PDF未成功读取，以上按项目页与arXiv v1判断。</p></section>
<section class="card"><h2>投入建议与保留边界</h2><p>将DGGT保留为快速重建/几何初始化及编辑速度对照；当前不再把它当作已经接近可交付的高保真编辑主线。停止继续重复Difix、参考图或透明度调参。这个决策针对DGGT当前公开实现与已测输入，并不排除前馈初始化加场景优化、显式背景建模或其他路线；尚未启动这些新方向。</p><p>定时任务已删除。Seen-to-Scene保持暂停，最新完整14000断点已验证保留；最后日志14386，386次尾部更新未保存。服务器保持开机。</p></section>
<p>页面只汇总实测与已核对公开材料；原始失败、模型及浮点trace保留远端。未做本页浏览器视觉QA。</p></body></html>'''.replace('TOP',top).replace('LAYERS',layers).replace('ROWS',rows)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.refs=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ('href','src') and v:self.refs.append((tag,k,v))
    links=Links();links.feed(doc);local=[];decoded=0
    for tag,k,v in links.refs:
        if urlsplit(v).scheme:continue
        target=(root/v).resolve();assert target.is_relative_to(root) and target.is_file(),v
        local.append(v)
        if tag=='img':
            with Image.open(target) as im:im.load()
            decoded+=1
    (root/'diagnosis_review.html').write_text(doc,encoding='utf-8')
    audit={'local_links':len(local),'missing_links':0,'decoded_png_references':decoded,'browser_visual_qa':False,'new_gpu_calls':0}
    (root/'diagnosis_review_validation.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit))


if __name__=='__main__':main()
