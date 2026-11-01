from pathlib import Path
import json,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
W=Path(__file__).resolve().parent;OUT=W.parent.parent/'outputs/V43_R4_FiveCase_GPU_20261101'
data=json.loads((OUT/'review/data.js').read_text(encoding='utf-8')[len('window.FIVE_DATA='):-1])
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
html=(OUT/'mechanisms/index.html').read_text(encoding='utf-8')
for c in data:
    f=c['frames'][24]['frame_id'];folder=OUT/'mechanisms'/c['case']
    observed=Image.open(folder/(f+'_observed.png')).convert('RGB');assert observed.size==(480,320)
    z=np.load(folder/'regions'/(f+'.npz'))
    a=np.load(OUT/'fp32_controls'/c['case']/'h000/none'/(f+'.npz'))
    native=Image.fromarray(np.uint8(np.clip(a['final'].transpose(1,2,0),0,1)*255+.5))
    panels=[('原始观测 · 原始帧 '+str(c['frames'][24]['raw']),observed),('冻结模型 · 无编辑',native)]
    for label,key,color in [
        ('目标实测深度代理','observed_target_depth_proxy',(50,230,240)),
        ('实测与平均深度不一致代理','mean_depth_disagreement_proxy',(255,145,60)),
        ('稀疏历史地面见证代理','historical_ground_witness_proxy',(65,230,115)),
        ('尚无核实历史见证','background_without_verified_witness',(255,95,145))]:
        mask=z[key];base=np.asarray(observed).copy();base[mask]=np.uint8(.25*base[mask]+.75*np.array(color))
        panels.append((label+' · '+str(int(mask.sum()))+'像素',Image.fromarray(base)))
    fig=Image.new('RGB',(970,1070),'#101b2b');draw=ImageDraw.Draw(fig)
    for i,(label,im) in enumerate(panels):
        x=(i%2)*490;y=(i//2)*355;draw.text((x+5,y+4),label,(224,236,246),font=font);fig.paste(im,(x,y+30))
    file=f+'_region_review.jpg';fig.save(folder/file,quality=95,subsampling=0)
    marker='<h2>Waymo · scene-'+c['scene_key'].split('-')[-1]+' / '+c['actor']+'</h2>'
    addon='<p>固定中锚点对照。着色只标证据代理及待核实区，不表示真实点归属。平均深度不是首交面；历史地面见证不是道路 RGB 真值。本例所有区域的归属训练标签均禁用。</p><img loading="lazy" src="'+c['case']+'/'+file+'" alt="原始观测、无编辑渲染与四类证据代理区域">'
    if file not in html:html=html.replace(marker,marker+addon)
(OUT/'mechanisms/index.html').write_text(html,encoding='utf-8')
shutil.copy2(W/'video_contact.png',OUT/'fixed_delete_endpoints.png')
p=OUT/'report.html';html=p.read_text(encoding='utf-8')
if 'fixed_delete_endpoints.png' not in html:
    marker='<h2>结果可以说明什么</h2>'
    html=html.replace(marker,marker+'<p>以下是事先固定的中锚点删除对照：左为原绑定，右为完整来源候选全部重绑定。五例没有呈现一致的残留清除；部分右侧显露亮色残留，并改变其他区域。这些是范围与来源控制，不能直接命名为好答案，也不能从一个端点失败推断全部逐点绑定都不可行。</p><img src="fixed_delete_endpoints.png" alt="五例固定中帧删除端点对照">')
p.write_text(html,encoding='utf-8')
shutil.copy2(__file__,OUT/'source/add_region_figures.py')
print('5 region figures and endpoint overview added')
