from pathlib import Path
import json
W=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-12/w');O=W/'outputs/drivingeditor_red_GN11_30k'
r=json.loads((O/'engineering-visual-review.json').read_text(encoding='utf-8-sig'))
assert r['review_complete'] is True and r['human_verdict'] is None
p=O/'results-report.md';t=p.read_text(encoding='utf-8')
start=t.index('## 实际视觉复核的当前状态');end=t.index('## 可复核证据与结论边界',start)
new='''## 实际分层与固定时间视觉复核

实际逐帧审查覆盖全部49观察帧的GT/合成/静态/动态/raw alpha_d，以及固定pose0029079的49时间静态RGB/raw alpha_s，共343张必需图；补充alpha_s/body的实际覆盖以工程coverage记录为准。图像以原480×320像素排入contact sheets，无局部滤波、逐图归一化或body预遮罩。人工质量判定仍为`human_verdict=null`。

红车总体形状、红色车身与横穿方向和GT相符，raw98–100右边缘截断可见，没有建立整个车体分支持续崩坏。观察帧raw71–98的静态道路仍留有黑色车底、轮胎或阴影条块，动态分支并未完整承担接地阴影。这些是实际质量代价；完整body opacity监督不等于已实现shadow分离。

固定pose下，raw71–100的黑条块随模型时间由左向右移动，raw101–117仍有右侧路边黑块；树影和道路纹理也随时间变化。raw105–117右侧天空/STOP附近出现细绿线，raw113–117前景道路有局部浅粉斑块。root还直接核对raw113–116的原尺寸静态RGB/alpha contact sheet，确认该反事实伪影；观察49帧没有同样明显的长绿线。固定pose与其它模型时间没有配对GT，因此不能赋予反事实GT PSNR；上述缺陷说明静态分支未达到绝对time invariant。

离场后的17观察帧body为空，动态/alpha_d展示没有明显车体大小的残留；这不证明所有像素严格为0，低阈值支持计数以工程记录为准。两个新scene没有各自C00，不能从这些绝对缺陷推导SGCM/NPS相对回退或模块因果收益，也不把原scene17的1 dB配对gate套用到该scene。

真实浏览器检查覆盖观察序列End=49/raw117及空body表现、共享ROI、播放1→41后暂停、固定时间End=49、raw alpha/body诊断、7个展示图的480×320加载与error[]。页面最终停在raw79原图，静态/动态分支截图保留。[实际视觉记录](engineering-visual-review.json)与[浏览器QA](root_browser_QA.json)分别记录像素判断和交互检查。

补充SAM资源记录覆盖模型加载与三帧真实推理：allocated 5.740 GiB、reserved 6.107 GiB、PID采样6.412 GiB、device采样6.422 GiB，11个NV样本、错误0。SAM的2.757秒只计已构造predictor后的三帧处理，不代表含加载的整个任务wall。包含sky/DEVA/SAM的本轮最大预处理观测仍为DEVA设备10.279 GiB，属采样下界。

'''
p.write_text(t[:start]+new+t[end:],encoding='utf-8')
(W/'work/drivingeditor_resume/heartbeat_20261013_1056/red_results_report.md').write_text(p.read_text(encoding='utf-8'),encoding='utf-8')
print('RED_WRITEUP_REAL_VISUAL_SECTION_FINALIZED')
