from pathlib import Path
from html import escape
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs' / 'vehicle_road_joint_20261121'
SOURCE = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
W, H = 2560, 1600
parts = []

def add(s):
    parts.append(s)

def rect(x,y,w,h,fill='#FFFFFF',stroke='none',rx=16,sw=1.5,**extra):
    attrs=' '.join(f'{k.replace("_","-")}="{escape(str(v))}"' for k,v in extra.items())
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {attrs}/>')

def line(x1,y1,x2,y2,color='#94A3B8',sw=2,marker=False,dash=None):
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" stroke-linecap="round"'+(' marker-end="url(#arrow)"' if marker else '')+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')

def text(x,y,s,size=25,fill='#142B43',weight=400,anchor='start',**extra):
    attrs=' '.join(f'{k.replace("_","-")}="{escape(str(v))}"' for k,v in extra.items())
    add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}" {attrs}>{escape(s)}</text>')

def path(d,fill='none',stroke='#2563EB',sw=3,**extra):
    attrs=' '.join(f'{k.replace("_","-")}="{escape(str(v))}"' for k,v in extra.items())
    add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" {attrs}/>')

def circle(x,y,r,fill,stroke='none',sw=2):
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

def pill(x,y,w,label,fg,bg='#FFFFFF',fs=20):
    rect(x,y,w,34,bg,fg,17,1.2)
    text(x+w/2,y+24,label,fs,fg,600,'middle')

def arrow(cx,y1,y2,color='#94A3B8'):
    line(cx,y1,cx,y2,color,3,True)

def car(x,y,s=1,color='#0D9488',fill='#E2F7F3'):
    add(f'<g transform="translate({x},{y}) scale({s})" aria-label="车辆">')
    path('M 4 34 L 9 18 Q 12 12 20 12 H 56 Q 64 12 67 18 L 72 34 Q 77 37 77 44 V 57 H 0 V 44 Q 0 37 4 34 Z',fill,color,2.5)
    path('M 16 20 H 60 L 65 32 H 11 Z','#FFFFFF',color,2)
    rect(8,40,13,7,'#FFFFFF',color,2,1.5)
    rect(56,40,13,7,'#FFFFFF',color,2,1.5)
    rect(5,54,13,10,color,'none',3)
    rect(59,54,13,10,color,'none',3)
    add('</g>')

def camera(x,y,s=1,color='#2563EB'):
    add(f'<g transform="translate({x},{y}) scale({s})">')
    rect(0,10,65,44,'#EDF4FF',color,7,2.6)
    path('M 17 10 L 22 2 H 43 L 48 10','#EDF4FF',color,2.5)
    circle(33,32,13,'#FFFFFF',color,2.6)
    circle(33,32,5,'#DBEAFE')
    add('</g>')

def points(x,y,s=1,color='#2563EB'):
    add(f'<g transform="translate({x},{y}) scale({s})">')
    for i,(dx,dy) in enumerate([(2,26),(11,9),(18,38),(27,21),(35,1),(42,33),(52,16),(61,42),(67,6),(74,28)]):
        circle(dx,dy,3.2+(i%3)*.5,color)
    add('</g>')

def road(x,y,s=1,color='#2563EB'):
    add(f'<g transform="translate({x},{y}) scale({s})">')
    path('M 20 0 H 54 L 78 64 H -4 Z','#E8F0FB',color,2.5)
    line(37,2,37,12,color,3)
    line(37,23,37,36,color,3)
    line(37,48,37,61,color,3)
    add('</g>')

def layers(x,y,s=1,color='#2563EB'):
    add(f'<g transform="translate({x},{y}) scale({s})">')
    path('M 0 15 L 30 0 L 60 15 L 30 30 Z','#EDF4FF',color,2.5)
    path('M 0 27 L 30 42 L 60 27',stroke=color,sw=2.5)
    path('M 0 39 L 30 54 L 60 39',stroke=color,sw=2.5)
    add('</g>')

def axis(x,y,s=1,color='#2563EB'):
    add(f'<g transform="translate({x},{y}) scale({s})">')
    line(15,45,61,45,color,2.8,True)
    line(15,45,15,1,color,2.8,True)
    line(15,45,-4,62,color,2.8,True)
    text(65,50,'x',17,color,600)
    text(11,-3,'z',17,color,600)
    text(-15,72,'y',17,color,600)
    add('</g>')

def gaussian_car(x,y,s=1,color='#0D9488'):
    add(f'<g transform="translate({x},{y}) scale({s})">')
    for yy in range(4):
        for xx in range(9):
            dx=xx*8 + (yy%2)*3
            dy=yy*8 + (5 if xx<2 or xx>6 else 0)
            add(f'<ellipse cx="{dx}" cy="{dy}" rx="3.8" ry="2.2" fill="{color}" opacity="{.48+(xx%3)*.18}" transform="rotate(-15 {dx} {dy})"/>')
    for xx in range(5):
        add(f'<ellipse cx="{18+xx*8}" cy="{-8}" rx="4" ry="2.4" fill="{color}" opacity=".6"/>')
    circle(14,35,7,'#FFFFFF',color,2.2)
    circle(60,35,7,'#FFFFFF',color,2.2)
    add('</g>')

def panel(x,w,num,title,en,color,light,status,status_w=116,status_color=None):
    add(f'<g id="module-{num}" aria-label="{escape(title)}">')
    rect(x,168,w,1286,light,color,22,1.5)
    rect(x,168,w,104,color,'none',22)
    rect(x,220,w,52,color,'none',0)
    circle(x+52,218,32,'#FFFFFF')
    text(x+52,230,str(num),37,color,700,'middle')
    text(x+101,212,title,31,'#FFFFFF',700)
    text(x+101,248,en,21,'#FFFFFF',500)
    pill(x+w-status_w-16,284,status_w,status,status_color or color,'#FFFFFF',19)
    add('</g>')

def card(x,y,w,h,title,lines_,color,fill='#FFFFFF',icon=None):
    add(f'<g class="component" aria-label="{escape(title)}">')
    rect(x,y,w,h,fill,color,15,1.2)
    tx=x+24
    if icon:
        icon(x+22,y+(h-54)/2,.76,color)
        tx=x+103
    text(tx,y+38,title,27,weight=700)
    for i,s in enumerate(lines_):
        text(tx,y+73+i*31,s,23,'#4A5F73')
    add('</g>')

add(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title description">
<title id="title">从驾驶观测到可编辑三维街景</title>
<desc id="description">四模块架构：观测与固定测距道路基础；已实现并完成少帧验证的道路补全；a20对象局部刚体高斯与624个下半车体/轮胎形状先验高斯；统一深度与alpha合成、分阶段后联合优化及75行局部归属修复。删除目标为a9白Mazda，保留补全目标为a20黑皮卡，a9/a20为场景可视别名。原生30K步终态高斯在第110帧作逆姿态变换与SH0颜色烘焙；底部1/4持续遮挡，实测盒和可见车宽约束隐部先验，隐藏形状、轮胎与材质为推断。200步车辆及200步联合更新、保存、恢复和五帧重载路径已验证，原训练封包保留。投影盒与掩码分区虽使物理a20盒与roadlegal零交集，继承大足迹仍在prior raw86/110道路教师域的457/482像素产生大于1e-3车辆贡献，实际贡献隔离未通过。主要白块残影、先验轮胎生硬及局部退化仍在，最终画质与贡献隔离待改进。道路两轮累计7239真新增高斯、3121锚点启用，历史892锚点启用另列。</desc>
<defs>
  <marker id="arrow" markerWidth="9" markerHeight="9" refX="7.3" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M 0 0 L 9 4.5 L 0 9 Z" fill="#64748B"/></marker>
  <filter id="shadow" x="-10%" y="-10%" width="120%" height="125%"><feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#193653" flood-opacity=".06"/></filter>
</defs>
<style>text {{font-family:'Microsoft YaHei','Noto Sans CJK SC','Arial',sans-serif;}} .component {{filter:url(#shadow);}}</style>
''')
rect(0,0,W,H,'#F6F9FD',rx=0)
text(W/2,76,'从驾驶观测到可编辑三维街景',56,'#0B2540',700,'middle')
text(W/2,122,'固定道路几何  ·  持久三维资产  ·  对象级车辆  ·  统一可编辑渲染',25,'#536B83',500,'middle')

BLUE,ORANGE,TEAL,NAVY='#2563EB','#DC600F','#0D9488','#335776'
panel(32,610,1,'观测与几何基础','Observations & fixed geometry',BLUE,'#EEF5FF','共享输入')
panel(660,610,2,'道路补全与资产化','Road recovery & persistent GS',ORANGE,'#FFF4E9','已实现并完成少帧验证',236,'#19814B')
panel(1288,610,3,'车辆补全与资产化','Object-centric vehicle completion',TEAL,'#ECF9F6','训练/保存/重载已验证',236,'#2563EB')
panel(1916,612,4,'联合合成与编辑','Joint rendering & editing',NAVY,'#EFF5F8','训练/保存/重载已验证',236,'#2563EB')

# Module 1: shared inputs and fixed geometry.
x,w=50,574
card(x,332,w,100,'多帧 RGB 与相机标定',['相机内外参 · 帧间位姿'],BLUE,icon=camera)
card(x,448,w,100,'LiDAR / 深度',['真实测距支撑与几何约束'],BLUE,icon=points)
card(x,564,w,116,'车辆实例与轨迹',['a9 白 Mazda · a20 黑皮卡','实例 mask · 逐帧刚体姿态'],BLUE,icon=car)
card(x,696,w,112,'原始静态场与对象初始 GS',['保留原场景的几何与可见支持'],BLUE,icon=layers)
arrow(337,818,842)
card(x,850,w,108,'时间与坐标对齐',['世界坐标与对象局部坐标一致'],BLUE,icon=axis)
arrow(337,968,990)
card(x,998,w,134,'固定测距支撑道路',['由射线—道路交点建立支撑','锁定道路位置、法向与厚度'],BLUE,icon=road)
arrow(337,1142,1166)
rect(x,1174,w,216,'#E2EEFF',BLUE,16,1.6)
text(x+24,1216,'共享几何基座',29,'#17458E',700)
rect(x+20,1242,258,105,'#FFFFFF','#AFC9F2',12)
rect(x+294,1242,258,105,'#FFFFFF','#AFC9F2',12)
road(x+42,1265,.66,BLUE)
text(x+119,1283,'道路支撑',25,'#17458E',700)
text(x+119,1317,'固定几何',21,'#4A5F73')
points(x+315,1271,.65,BLUE)
text(x+391,1283,'对象初始 GS',24,'#17458E',700)
text(x+391,1317,'局部表达',21,'#4A5F73')
text(x+24,1373,'作为道路与车辆恢复的共同输入',22,'#3E5D81')

# Module 2: verified road pipeline.
x=678
card(x,332,w,124,'冻结二维去车 / 去影教师',['SD1.5 + LoRA · 1152 × 768','删除 a9；合法道路区排除完整 a20 盒'],ORANGE)
arrow(965,466,484)
card(x,492,w,88,'α 缺口检测',['定位可恢复道路的覆盖不足'],ORANGE)
arrow(965,590,608)
card(x,616,w,88,'固定道路射线交点',['在固定道路支撑上确定补点位置'],ORANGE)
arrow(965,714,732)
rect(x,740,w,130,'#FFFFFF',ORANGE,15,1.2)
text(x+24,778,'按需扩充覆盖',27,weight=700)
text(x+24,814,'先启用闲置锚点 → 再自适应新增 GS',23,'#9B470E',600)
text(x+24,848,'仅在必要处采样 · 跨视角合并',23,'#4A5F73')
arrow(965,880,898)
rect(x,906,w,122,'#FFFFFF',ORANGE,15,1.2)
text(x+24,944,'受约束覆盖优化',27,weight=700)
text(x+24,978,'优化 α 与切向尺度',23,'#4A5F73')
text(x+24,1008,'位置冻结 · 法向厚度冻结',22,'#9B470E',600)
arrow(965,1038,1056)
card(x,1064,w,88,'真实 3DGS 外观拟合',['RGB + 高频细节（HF）+ 边缘'],ORANGE)
arrow(965,1162,1180)
rect(x,1188,w,202,'#FFE7CE',ORANGE,16,1.6)
text(x+24,1227,'持久道路资产',29,'#9B470E',700)
text(x+24,1261,'固定几何 · 可渲染覆盖 · 跨视角共享',23,'#735740')
rect(x+20,1281,258,66,'#FFFFFF','#EAB78A',10)
rect(x+294,1281,258,66,'#FFFFFF','#EAB78A',10)
text(x+42,1325,'7,239',34,'#A54305',700)
text(x+161,1324,'真新增 GS',22,'#6E4F36',600)
text(x+313,1325,'3,121',34,'#A54305',700)
text(x+430,1324,'锚点启用',22,'#6E4F36',600)
text(x+24,1376,'两轮累计；历史 892 锚点启用另列',21,'#7B5A3C')

# Module 3: target a20 is behind deletion target a9. Current integration, not a verified result.
x=1306
card(x,332,w,134,'a20 黑皮卡的跨帧可见证据',['真实 RGB · 实例 mask · LiDAR / 轨迹','底部 1/4 持续遮挡 · 视角变化仅 0.47°'],TEAL)
arrow(1593,476,498)
rect(x,506,w,118,'#FFFFFF',TEAL,15,1.2)
text(x+24,545,'原生 30K 步终态 → canonical GS',27,weight=700)
text(x+24,580,'第 110 帧逆姿态变换 · SH0 颜色烘焙',22,'#4A5F73')
text(x+24,611,'3D 盒筛除污染点 · 对象局部坐标',22,'#4A5F73')
arrow(1593,634,656)
rect(x,664,w,118,'#FFFFFF',TEAL,15,1.2)
text(x+24,702,'逐帧刚体变换',27,weight=700)
text(x+24,738,'局部 → 世界：x世界 = Rₜ x局部 + tₜ',23,'#087D72',600)
text(x+24,769,'HUGSIM 路线 · 固定相机与轨迹',22,'#4A5F73')
arrow(1593,792,814)
rect(x,822,w,142,'#FFFFFF',TEAL,15,1.2)
text(x+24,860,'下半车体 / 轮胎形状先验',27,weight=700)
text(x+24,895,'624 新增 GS：面板 / 保险杠 / 轮胎',22,'#4A5F73')
text(x+24,924,'实测盒与可见车宽 · 车漆中值初始化',22,'#4A5F73')
text(x+24,952,'隐藏形状 / 轮胎 / 材质为先验推断',21,'#2563EB',500)
arrow(1593,974,996)
rect(x,1004,w,130,'#FFFFFF',TEAL,15,1.2)
text(x+24,1042,'车辆优化 200 步（本轮）',27,weight=700)
text(x+24,1078,'颜色；可见支持行 α / scale 有界适配',22,'#4A5F73')
text(x+24,1113,'固定 xyz / pose · RGB / 完整 mask 监督',22,'#4A5F73')
arrow(1593,1144,1166)
rect(x,1174,w,104,'#DFF5EF',TEAL,16,1.6)
text(x+24,1213,'a20 持久车辆资产',28,'#086D64',700)
text(x+24,1252,'局部 GS / 轨迹 / 身份 · 视觉未通过',22,'#375E59')
rect(x,1304,w,86,'#FCFAFF','#A78BFA',14,1.5,stroke_dasharray='8 6')
text(x+24,1338,'后续可选：连续曲面 · Surface LOD',23,'#7455A5',600)
text(x+24,1371,'a9 / a20 为场景可视别名',21,'#7455A5')

# Module 4: shared renderer and explicit supervision isolation.
x=1934
rect(x,332,w,204,'#FFFFFF',NAVY,15,1.2)
text(x+24,370,'统一场景输入',27,weight=700)
for yy,label,color,kind in [(399,'原静态场',BLUE,'layers'),(451,'持久道路资产',ORANGE,'road'),(503,'a20 补全车辆资产',TEAL,'car')]:
    if kind=='layers': layers(x+25,yy-23,.46,color)
    elif kind=='road': road(x+25,yy-23,.45,color)
    else: car(x+24,yy-27,.46,color)
    text(x+82,yy+6,label,25,color,600)
arrow(2221,546,570)
rect(x,578,w,132,'#E7EFF5',NAVY,15,1.5)
text(x+24,618,'统一 depth / α 合成',29,'#294D6B',700)
text(x+24,655,'静态 / 道路 / a20 GS 共同深度排序',23,'#4A5F73')
text(x+24,688,'KEEP / DELETE 共用 renderer 与 loss',23,'#4A5F73')
arrow(2221,720,744)
rect(x,752,w,194,'#FFFFFF',NAVY,15,1.2)
text(x+24,790,'投影盒 + 掩码分区',27,weight=700)
circle(x+29,825,5,TEAL)
text(x+45,833,'a20：真实可见 RGB / 完整 mask',23,'#375E59')
circle(x+29,865,5,ORANGE)
text(x+45,873,'道路：按投影盒 + mask 许可教师目标',22,'#8D4B20')
circle(x+29,905,5,NAVY)
text(x+45,913,'越界足迹仍进入道路教师域：待修复',22,'#A35417')
arrow(2221,956,980)
rect(x,988,w,138,'#FFFFFF',NAVY,15,1.2)
text(x+24,1026,'分阶段后联合优化',27,weight=700)
text(x+24,1063,'道路完成 → 车辆 200 步 → 联合 200 步',23,'#4A5F73')
text(x+24,1100,'联合：a20 SH0 颜色 + 1,191 道路 RGB',22,'#4A5F73')
arrow(2221,1136,1152)
rect(x,1160,w,112,'#FFFFFF',NAVY,15,1.2)
text(x+24,1198,'旧 a9 残留归属修复（局部）',26,weight=700)
text(x+24,1230,'贡献核验 → 75 行归属修复 → 删车退出',22,'#4A5F73')
text(x+24,1258,'KEEP 保持 · 主要白块残影仍在',21,'#8D4B20')
arrow(2221,1282,1292)
rect(x,1300,w,90,'#E7F2F4',NAVY,16,1.6)
text(x+24,1336,'联合结果：画质与贡献隔离待改进',27,'#A35417',700)
rect(x+24,1348,226,30,'#FFFFFF','#AAC3CE',9)
rect(x+268,1348,282,30,'#FFFFFF','#AAC3CE',9)
text(x+137,1371,'DELETE · a9 白 Mazda',20,'#294D6B',600,'middle')
text(x+409,1371,'KEEP / 补全 · a20 黑皮卡',20,'#294D6B',600,'middle')

# Single concise overview: road and vehicle assets are parallel branches.
rect(32,1480,2496,88,'#FFFFFF','#C6D6E5',16,1.4)
text(58,1534,'整体数据流',26,'#294D6B',700)
for xx,ww,label,col,bg in [(245,350,'观测与固定几何',BLUE,'#EEF5FF'),(659,458,'道路资产 / 车辆资产',TEAL,'#ECF9F6'),(1181,350,'统一合成与优化',NAVY,'#EFF5F8'),(1595,328,'a9 删除 · a20 补全',NAVY,'#EFF5F8')]:
    rect(xx,1498,ww,51,bg,col,12,1.2)
    text(xx+ww/2,1532,label,25,col,600,'middle')
for xx in [608,1130,1544]: line(xx,1524,xx+36,1524,'#64748B',2.8,True)
line(1960,1497,1960,1551,'#C6D6E5',1.3)
circle(1990,1510,6,'#19814B')
text(2006,1518,'道路已实现 / 少帧验证',19,'#3C5368',600)
circle(2240,1510,6,'#2563EB')
text(2256,1518,'训练 / 保存 / 重载验证',19,'#3C5368',600)
rect(1984,1534,18,13,'none','#A78BFA',3,1.5,stroke_dasharray='4 3')
text(2013,1547,'后续可选，尚未实现',20,'#725590')
add('</svg>')

svg='\n'.join(parts)
(OUT/'architecture.svg').write_text(svg,encoding='utf-8')
(SOURCE/'render.html').write_text(f'<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;padding:0;width:{W}px;height:{H}px;overflow:hidden;}}svg{{display:block;}}</style></head><body>{svg}</body></html>',encoding='utf-8')
metadata = {
    'title':'从驾驶观测到可编辑三维街景', 'canvas':[W,H],
    'road_true_added_gs':7239, 'road_activated_anchors':3121,
    'historical_892':'separate historical anchor activation only',
    'vehicle_status':'training, saving and reload verified; visual quality and render-contribution isolation failed',
    'vehicle_representation':'native valid GS from the 30K-step endpoint, raw110 inverse-pose canonical transform, object SH0 color bake, 3D-box contamination filtering, canonical rigid GS',
    'hidden_surface_prior':'624 added lower-body panels, bumpers and tire-prior GS, constrained by measured 3D box and visible width, visible dark-paint median initialization; hidden shape, tire shape and material are inferred',
    'vehicle_steps':200,
    'vehicle_parameters':'SH0 color and visible-renderer-VJP-supported bounded alpha/scale; canonical xyz/pose frozen',
    'joint_steps':200, 'joint_parameters':'a20 SH0 color and 1191 road-GS RGB only',
    'verified_paths':'restore, both KEEP/DELETE gradient updates, freezing, fresh disk five-frame reload passed',
    'postfit_handoff':'75 verified unknown-static rows assigned a9 ownership in sidecar; KEEP/other-car/a20-visible maintained, outside ROI at most one gray-level code; no added training, original joint_fit retained',
    'visual_result':'major a9 white-block ghost remains, inferred tires are rigid-looking, three observation frames and some other-car regions degraded',
    'contribution_isolation':{'physical_a20_box_roadlegal_overlap':0, 'prior_roadlegal_vehicle_contribution_gt_1e_3_pixels':{'86':457,'110':482}, 'passed':False},
    'human_verdict':None,
    'delete_target':'Scene065 / a9 / white Mazda / NTjq-OhZ_97OMJ4xwJAxXg',
    'vehicle_completion_target':'Scene065 / a20 / black pickup / bzFbWRrSGOgEaXxnuGV0tw',
    'font':'Microsoft YaHei',
    'notes':'a9/a20 are scene visual aliases. HUGSIM route refers to canonical rigid Gaussian mechanisms, not a complete HUGSIM system port. Lower one quarter is occluded throughout the sequence; view variation is 0.47 degrees. Optional continuous surface and LOD are not implemented.'
}
(SOURCE/'content.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
print(OUT/'architecture.svg')
print(SOURCE/'render.html')
