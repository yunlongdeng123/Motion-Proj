"""Build the separate Chinese delivery note; never link it into the workbench."""
from pathlib import Path
from html import escape
from hashlib import sha256
import json
import shutil
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[3]
WORK = BASE / "work/actor_diagnostics_20261101"
OUT = BASE / "outputs/Actor_Diagnostics_20261101"
GPU = BASE / "outputs/V43_R4_GPU_20261031"

def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def write_json(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

def architecture():
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",28)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",23)
    image = Image.new("RGB",(1600,390),"#f5f8fc");d=ImageDraw.Draw(image)
    boxes=[(25,95,345,280,"原始片段","RGB · 标定 · 轨迹 · LiDAR"),(430,95,750,280,"只读提取与核对","完整帧 · 三维框 · 量测点"),(835,95,1155,280,"逐车证据检查","几何 · 视向 · 背景 · 运动"),(1240,95,1575,280,"候选浏览","同步视频 · 轨迹 · 几何对照")]
    for x0,y0,x1,y1,title,sub in boxes:
        d.rounded_rectangle((x0,y0,x1,y1),radius=16,fill="white",outline="#7e9fc2",width=3)
        d.text(((x0+x1)/2,y0+45),title,font=font,anchor="mm",fill="#163653")
        d.text(((x0+x1)/2,y0+110),sub,font=small,anchor="mm",fill="#315c7d")
    for x0,x1 in [(345,430),(750,835),(1155,1240)]:
        d.line((x0+8,187,x1-13,187),fill="#315c7d",width=5)
        d.polygon([(x1-10,187),(x1-27,177),(x1-27,197)],fill="#315c7d")
    d.text((800,338),"只使用编辑前输入；独立编辑真值和学生结果不进入筛选",font=small,anchor="mm",fill="#315c7d")
    image.save(OUT/"architecture.png")

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    evidence=OUT/"evidence";evidence.mkdir(exist_ok=True)
    review=WORK/"review"
    for name in ["source_only_second_screen_facts.json","source_only_screen_protocol.json","actor_eligibility.json","preparation45_source_diagnostics.csv","badcase_workbook_extract.json","media212_cross_runtime_numeric_audit.json"]:
        if (review/name).exists():shutil.copy2(review/name,evidence/name)
    facts=read(review/"source_only_second_screen_facts.json")
    names=read(GPU/"actor_candidates/actor_aliases.json")["scenes"]
    generated=WORK/"media/generated/scenes"
    media=[]
    for path in sorted(generated.glob("*/manifest.json")):
        record=read(path)
        if record.get("full_sequence"):media.append(record)
    geom=read(OUT/"geometry/comparison.json") if (OUT/"geometry/comparison.json").exists() else {"rows":[]}
    qa=read(OUT/"delivery_verification.json") if (OUT/"delivery_verification.json").exists() else {"state":"not_run"}
    scene_rows=[]
    for scene in facts["scenes"]:
        context=scene["context_id"];entry=names[context]
        built=next((r for r in media if r["context_id"]==context),{})
        scene_rows.append("<tr>"+"".join("<td>"+escape(str(v))+"</td>" for v in [entry["display_name"],scene["raw_frames"],scene["tracks"],built.get("raw_frame_count","待生成"),built.get("measured_box_rows","待生成")])+"</tr>")
    status = {"schema":"actor_diagnostics_delivery_v1","task_id":"WS-V43-ACTOR-DIAGNOSTICS-20261101","date_local":"2026-11-01",
        "scenes":15,"registered_vehicle_tracks":1228,"unchanged_preparation_queue":45,"media_completed_scenes":len(media),
        "complete_video_frames":sum(r["raw_frame_count"] for r in media),"measured_sensor_points":sum(r["measured_sensor_points"] for r in media),
        "measured_box_rows":sum(r["measured_box_rows"] for r in media),"GS_comparison_actors":len({(r["context_id"],r["track_id"]) for r in geom["rows"]}),
        "GS_anchor_attempts":len(geom["rows"]),"GS_anchor_comparisons":sum(bool(r.get("valid")) for r in geom["rows"]),
        "GS_missing_pose_anchors":sum(not r.get("valid") for r in geom["rows"]),"formal_eligibility_true":0,"formal_eligibility_false":0,"formal_eligibility_unknown":1228,
        "counterfactual_edited_reference_used_for_screen":False,"student_output_used_for_screen":False,"training_updates":0,"GPU_queries":0,"human_verdict":None,
        "strict_control_pass_scenes":14,"strict_control_fail_scenes":1,
        "facts":facts["all"],"preparation45":facts["preparation45"],"media_scenes":media,
        "failure_ledger_refs":["V82-F09","V82-F10"],"failure_ledger_delta":["V82-F09","V82-F10"],"interactive_browser_QA":False}
    write_json(OUT/"summary.json",status)
    architecture()
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>车辆候选时序与几何核查</title>
<style>body{{margin:0;background:#f4f7fb;color:#19334d;font:16px/1.7 "Microsoft YaHei",sans-serif}}main{{max-width:1120px;margin:auto;padding:30px 24px 80px}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:32px}}a{{color:#176bb0}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:9px 12px;border-bottom:1px solid #d5e0ec;text-align:left}}.note{{background:#fff4dc;border-left:4px solid #d18b21;padding:14px 18px}}img{{max-width:100%;height:auto}}code{{overflow-wrap:anywhere}}.muted{{color:#506a80}}</style><main>
<h1>车辆候选时序与几何核查</h1><p class="muted">2026-11-01 · 独立说明，未加入共享平台导航</p>
<p>候选页黑屏来自标注画布的不透明背景遮住原图，已改为透明；缺图和加载失败现在有明确提示。原场景号和 aN 车辆代号保持稳定，完整原始 ID 可展开核对。</p>
<p><a href="../V43_R4_GPU_20261031/actor_candidates/index.html#scene=waymo-train-212&actor=a0">打开修订后的车辆候选页</a></p>
<img src="architecture.png" alt="原始输入经只读提取、逐车证据检查，进入同步候选浏览的组件图">
<h2>补齐可检查的实际输入</h2><p>15 场保留全部 1,228 条车辆轨迹与原 45 条准备队列。当前完成 {len(media)}/15 场完整原始片段，共 {status['complete_video_frames']:,} 帧。原始 RGB 与三维框视频分别保存，可逐帧查看所选车辆、轨迹和 LiDAR；没有用单帧或抽样帧冒充完整视频。</p>
<p>浏览副本为 960×640：逐帧 JPEG 与 H.264 视频都经过缩放和有损编码。完整指原始时间序列的全部帧，不表示逐像素无损；这些浏览副本不用于正式图像指标或资格阈值计算。选车会先定位到该车可见处，完整片段和时间轴仍保留；“车辆出现处”可快速返回。</p>
<p>LiDAR 来自五个传感器、两次回波，保留 TOP 的逐像素运动补偿和 NLZ 过滤。每辆车的框内原始量测行以 NPZ 保存，浏览使用 6 厘米网格与 12,000 点显示上限；显示抽样不改变实测分母。框内点可能含路面、遮挡物或其他物体，不能当成已分割好的车体表面。投到相机上的颜色只作查看，灰色表示没有前向颜色对应，颜色不作为可见性证据。</p>
<p>逐帧模式从完整 NPZ 的各个原始帧独立抽样，使用 6 厘米网格、每帧最多 512 点，没有将全片去重后的点按帧反筛。真实零点、没有车辆标注、显示文件未提供分别显示；轨迹使用完整逐帧原始标注，缺测段断线，不补出虚构位置。</p>
<p>实际解码 497,156,385 行传感器回波，其中 44,046,945 行进入车辆三维框内量测集合；同一点可进入多个框，连帧和多回波不当作独立物理点。1,198 条轨迹有实测量，另 30 条全片框内零量测原样保留。45 条准备候选全片及建议窗口内都有量测，这只证明存在支持，不能证明完整表面或编辑资格。</p>
<p>三维框使用原始框、标定和提供的图像位姿投影；没有做逐行滚动快门及车辆曝光时刻运动补偿，因此用于定位与轨迹核对，不作像素精确几何真值。已修正视野外框在大畸变处折回画面的投影错误。</p>
<p>训练集 scene-212 使用远端 Linux 解码缓存，本机独立重解三帧的 32,997 行测点；两运行时的最大坐标分量差为 0.976 毫米，严格 0.02 毫米一致性检查未通过，原阈值与数据均保留。超过严格阈值的 584 行均来自 TOP，与约 7 公里世界坐标的 float32 消去误差相符；独立 float64 控制最大误差约 1.13 毫米。传感器计数、源行身份逐项一致，不声称其余坐标位级相等。其余 14 场的三帧独立控制均通过同一严格检查，见<a href="evidence/independent_review.json">完整独立审查</a>。212 此项不叫数值 parity 通过，也不据此判车辆几何失败。各帧刚体汇聚尚未补偿扫描期间车辆自身的运动。</p>
<p>三场已有模型读取了静态、动态两个分支的全部 5,499,413 个原始高斯，给 9 条既有场景准备候选制作三锚几何对照。对照先按当前合法姿态转入车辆局部坐标，再在同一米制坐标中显示 LiDAR 与高斯中心。逐车显示及最近距离的数值域限于有效 opacity≥0.01、中心位于三维框外扩 0.5 米内；这是诊断裁剪，框外椭球也可能投影入车身。它包括可能错留静态分支的车辆行；框内邻近行也可能属于背景或别的物体。中心最近距离不是椭球覆盖、深度一致率或可见表面完整度，不能代替 R4 的 f_geom。</p>
<p>27 个预定锚点中，26 个得到真实对照；训练集 scene-537 / a105 的 raw149 没有原始轨迹姿态，保留未测记录，没有插值造图。它的 raw173 与 raw197 对照仍可查看。</p>
<p>这三场共登记 309 条车辆轨迹，本轮中心裁剪对照只覆盖其中 9 条准备车。第四个已有场景 scene-008 的 60 条轨迹、完整视频与 LiDAR 仍保留，五个既有资产不等于无模型；其恢复和坐标转换尚未纳入本轮新高斯对照。</p>
<h2>二次检查怎样隔离无关坏例</h2><div class="note">45 条是准备顺序，不是 45 条合格车。当前没有标记任何正式 P=1 或 P=0：1,228 条及其逐操作资格仍为未知。已核实的观测缺口、准备缺口与未完成测量分别记录；这些状态不能冒充归属学习成功或失败。</div>
<table><thead><tr><th>检查</th><th>本轮实际完成</th><th>仍需完成的正式准入测量</th></tr></thead><tbody>
<tr><td>几何</td><td>真实框内 LiDAR、多帧局部对齐；已有三场完整双分支高斯对照；掩码库存单列</td><td>合法正像素车身掩码、有效深度附近的完整 G 累计贡献覆盖、原始 none 误差</td></tr>
<tr><td>视向</td><td>逐帧相机/对象方位代理，区分全片、建议 49 帧与当前模型窗口</td><td>操作将显露的真实表面、颜色和当前 G 的投影视向支持</td></tr>
<tr><td>背景</td><td>保持未知；不把框内车后区域自动当作已有道路</td><td>对应固定世界地点的历史真实显露射线与有效深度，按删除/移动分别登记</td></tr>
<tr><td>运动</td><td>时间、有限姿态/尺寸、帧缺口、坐标映射和原生窗口检查</td><td>逐车原生输入恢复、时间域和世界↔模型映射的完整准入</td></tr></tbody></table>
<p>45 条中，16 条只有 1 个、25 条只有 2 个前向粗方位扇区；43 条在 +30° 偏航的粗扇区诊断中出现源视向缺口。这是准备风险提示，不是表面完整度或旋转不可行的证明。当前 49 帧源窗口也不等于已经核实的 G 训练分区。</p>
<p>全体中 732 条没有前向投影，其中 563 条在其他相机有投影，不能隐藏它们或说没有观测；其余 169 条只表示导出的相机投影标注均无支持。另有 2 条轨迹存在帧缺口、7 条在前向出现时框内 LiDAR 标注计数始终为零，均保留在登记表中。</p>
<p>45 条中 6 条有既有掩码库存，其中 1 条（训练集 scene-008 / a5）掩码像素中位数为 0，不能把“49 个掩码文件”说成 49 帧有效车身。正像素帧数未存于现有汇总，仍保持未知。</p>
<p>45 条的固定首/中/末锚点检查发现 3 条缺少首锚原始标注姿态：训练集 scene-537 / a105 的 raw149、验证集 scene-004 / a14 的 raw149、scene-081 / a49 的 raw50。它们对应窗口内原始标注分别为 38/49、47/49、38/49 帧；没有在看过结果后另挑锚点，也不把缺测填成零误差。</p>
<p>badcase 表的 13 类按原因分开：归属、漏召回、弱可见来源和跨时序绑定进入决策证据检查；几何/尺度、颜色、新姿态缺面和未观测背景独立记录能力边界；双域支撑、关联效果、环境与合成补偿保留机制待查；展示接线归工程问题。本轮不从历史编辑残影或独立删车真值反推入选资格，也没有新增几何/颜色修复训练。</p>
<p><a href="evidence/source_only_screen_protocol.json">检查定义与 badcase 对应</a> · <a href="evidence/actor_eligibility.json">全部逐车、逐操作证据</a> · <a href="evidence/preparation45_source_diagnostics.csv">45 条准备候选二筛表</a></p>
<h2>全片分母与实际量测</h2><table><thead><tr><th>场景</th><th>原始帧</th><th>车辆轨迹</th><th>完整媒体帧</th><th>框内实测行</th></tr></thead><tbody>{''.join(scene_rows)}</tbody></table>
<p>轨迹数是登记分母，不是独立物理车辆或独立统计样本数。连帧和像素也不作为独立样本。本轮未改变旧 V43 有效/无效输出、候选合同或模型参数。</p>
<h2>核验与边界</h2><p>本轮训练更新与 GPU 查询均为 0。页面数据通过本地脚本直接载入，支持 file://，不依赖 fetch 服务。图像、媒体帧数、车辆映射、同帧定位及脚本交互按交付检查记录验收；浏览器交互受工具策略限制，未冒称完成真实浏览器测试。共享共同入口保持原有场景导航，不加入此报告、实验状态或研究计划。</p>
<p>来源：修订 R4 第 3 节；用户提供的 editor_badcase_analysis.xlsx 的 V43!B2:H14（只读提取见 <a href="evidence/badcase_workbook_extract.json">来源记录</a>）；原始 Waymo 数据和已有完整高斯导出。数据字段及坐标约定对照 <a href="https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto">Waymo 官方数据定义</a>，量测转换对照 <a href="https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/utils/range_image_utils.py">官方 range-image 转换</a>，原始框对照 <a href="https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/utils/box_utils.py">官方 box 工具</a>。附件内容用于核对设计与历史证据，不作为新增执行授权。</p>
<p class="muted">task: WS-V43-ACTOR-DIAGNOSTICS-20261101；human_verdict=null。源码、输入裁剪收据、显示与实测分母、已知投影近似均随交付保留。</p></main></html>'''
    (OUT/"report.html").write_text(body,encoding="utf-8")
    print(json.dumps({"report":str(OUT/"report.html"),"media_scenes":len(media),"frames":status["complete_video_frames"]},ensure_ascii=False))

if __name__=="__main__":main()
