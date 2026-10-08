"""用户v2文档的R001参数化适配；几何标注不等于真实轮廓。"""
from pathlib import Path
import json
from PIL import Image,ImageDraw

p=Path(__file__).parent/'evidence_pack'
roi=json.loads((p/'roi.json').read_text(encoding='utf-8'))
case=json.loads((p/'source/case.json').read_text(encoding='utf-8'))
b=next(a for a in case['frames'][5]['actors'] if a['instance_token']==roi['roles']['PROTECTED_B'])
x0,y0=roi['roi_full_xyxy_exclusive'][:2]
control=Image.open(p/'B_editmask_ONU_control.png').convert('RGB');d=ImageDraw.Draw(control)
hull=[(x-x0,y-y0) for x,y in b['hull']]
d.line(hull+[hull[0]],fill=(255,217,51),width=2)
d.text((hull[0][0]+4,hull[0][1]),'B / 45.6m',fill=(255,217,51))
control.save(p/'B_instance_control.png')
c=Image.open(p/'C_protected_B_same_track.png').convert('RGB')
canvas=Image.new('RGB',(512,384),(90,90,90));canvas.paste(c,(0,64))
ImageDraw.Draw(canvas).text((8,12),'B: DARK CAR / SAME TRACK / +1.35s / NOT ALIGNED',fill='white')
canvas.save(p/'C_B_anchor.png')
prompt='''任务：局部DELETE补景，只输出图A所对应的RGB局部图。图A原尺寸488×400，输出976×800，恰好2倍；保持同一取景、透视、物体位置和比例，不输出拼图、标注、图框或文字。

输入角色：
图A是唯一目标时刻R001 f05的原始RGB ROI；图B与图A逐像素对齐，是编辑范围和几何控制；图C是后方保护车B在+1.35秒的同track真实参考，不是当前视角，也不是输出模板。
删除对象A：图A中央偏左的近处银色SUV。仅删除它及编辑范围内属于它的阴影、薄膜、残片。不能把A补回来。
保留对象B：图C左侧的远处深色车辆。右侧相邻浅色车辆只是参考上下文，不是B。不要互换身份，不要直接复制参考车列到目标帧。

范围与几何：
图B红框是唯一允许修改的EDIT_MASK。在原ROI坐标x=146..340、y=120..279。红框外必须保留图A原始RGB；工程端会做严格硬合成。
图B黄色小六边形才是B在当前f05的投影包络：原ROI约x=152..216、y=153..186，深度约45.6米。这是3D框代理，不是精确silhouette。B是小而远的保护对象，不得放到红框中央、不得放大成近车，也不得重画整排车辆。
绿色O为多个保留物体的投影框内核，不能把每个绿块都当作B；蓝色N是实测背景LiDAR落点；灰色U是未知。框外或蓝点之间不等于已知空背景。不要把这些颜色、点、框、文字画入输出。

证据优先级：当前mask外真实像素 > 同track邻帧真实外观 > 当前投影几何 > 场景结构 > 模型先验。
图C没有可靠warp到图A，因此只取它的身份、颜色、车身比例和可辨部件；当前B的位置/尺度必须服从图B黄色包络和图A可见像素。前后端若证据不足，不强行编造具体车型、灯组、牌照或完整侧面。细节保持原始车载画面的模糊程度。
只在已确认道路的位置沿mask边界延续真实标线、沥青、透视与纹理。未知区域保持保守，不擅自判定全部是道路，也不靠新车填满mask。

硬性质量要求：删除A的银色车壳、半透明轮廓、悬空黑片和A自身影子；保住B而不是换一辆车；标线在红框四边连续；保留白色邻车、建筑和灯杆原位置，不改变曝光、颜色、锐度、天气和镜头。不美化，不新增车辆，不输出控制图。
这张输出仅作为生成伪标签候选，不是真实隐藏区真值；宁可保持低纹理的保守细节，不画高置信但无证据的车辆结构。
'''
(p/'prompt_v2_zh.txt').write_text(prompt,encoding='utf-8')
roi['B_proxy_hull_roi']=hull;roi['B_depth_m']=b['depth'];roi['generation_output_wh']=[976,800]
(p/'roi_v2.json').write_text(json.dumps(roi,ensure_ascii=False,indent=2),encoding='utf-8')
print('PROMPT_V2_READY')
