"""本地审核页面：模型条件、写回、地面来源分开讲解。"""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[3]/'outputs/v77-hybrid-delete/third-scene'
if not ROOT.exists():
 ROOT=Path('outputs/v77-hybrid-delete/third-scene').resolve()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
r25=read(ROOT/'r25/state.json');r28=read(ROOT/'r28/state.json')
def vid(key,title,note):
 return f'<article><h3>{title}</h3><video data-sync controls loop muted playsinline preload="metadata" poster="{key}_poster.jpg" src="{key}.mp4"></video><p>{note}</p></article>'
rows=''.join(f'<tr><td>f{r["query_frame"]}</td><td>{r["arms"]["nearest"]["coverage"]:.1%}</td><td>{r["arms"]["nearest"]["mean_mae"]:.2f}</td><td>{r["arms"]["consensus"]["coverage"]:.1%}</td><td>{r["arms"]["consensus"]["mean_mae"]:.2f}</td></tr>' for r in r25['queries'])
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v77 · 第三例 DELETE 与路面证据</title><style>
body{margin:0;background:#111820;color:#e5edf5;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:30px}h1,h2,h3{line-height:1.4}h1{font-size:30px}h2{margin-top:38px;border-bottom:1px solid #395269;padding-bottom:10px}a{color:#8ac9ff}p{max-width:1100px}.lead,.note{background:#1c2b39;padding:18px 22px;border-left:4px solid #65c6a4;border-radius:6px}.warn{border-color:#e5b76c}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}article{background:#1a2632;padding:14px;border-radius:10px}article p{font-size:14px;color:#bccedf}video,img{display:block;width:100%;height:auto;background:#080d13;border-radius:6px}button,input{accent-color:#64c2ab}button{background:#315773;color:white;border:0;padding:10px 16px;border-radius:6px;cursor:pointer;margin-right:8px}.controls{position:sticky;top:0;background:#142331f2;padding:12px;z-index:2;border-radius:8px}.controls input{width:min(500px,60%)}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border:1px solid #35495b;padding:10px;text-align:left}details{margin:20px 0}summary{cursor:pointer;color:#97d2ff}small{color:#a8bacb}svg{width:100%;height:auto}.wide{max-width:1300px}.tag{font:13px monospace;color:#9fcde0}footer{margin-top:50px;color:#aabccb;font-size:14px}@media(max-width:1000px){.grid{grid-template-columns:1fr}main{padding:18px}h1{font-size:25px}}
</style><main><p><a href="../index.html">返回主审核页</a> · <a href="../fence-writeback/index.html">0255：r18 与围栏修复</a> · <a href="../object-scale/index.html">r23/r24 已保存反例</a></p>
<span class="tag">WS-V77-HYBRID-BG-20260927 · r25—r28 · 2026-09-28</span><h1>第三例：灰色车形块消失了，写回接缝仍需修</h1>
<div class="lead"><b>保留用户判断：scene_0255，r18 ＞ r15，继续修。</b><br>本页加入 official_000 / actor12。r28 使用同一冻结 DriveEditor，仅将生成器的洞改成小范围矩形；最终仍只写回旧的精确实例 mask。10 帧里旧灰色车形块明显减少，路面更连贯，但亮边、局部虚线及地面接缝仍可见。此为助手观察，待用户审核，尚未送入 Ω。</div>
<p>目标是右侧车道黑色 MPV <b>actor12</b>，不是其左后方的卡车。原视频黄框标出目标。后方卡车 <b>actor34</b> 是真实保留对象：f0 相机深度约 44.85m，目标约 34.09m。去掉目标后应露出它，被补出的可见部分仍需核对身份与结构。</p>
<h2>本轮到底改了哪一步</h2>
<svg viewBox="0 0 1260 190" role="img" aria-label="原视频经矩形条件进入冻结DriveEditor，再按精确mask写回；地面证据暂未接入"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#8ac9ff"/></marker></defs><g fill="#1f3448" stroke="#5b88a8"><rect x="5" y="20" width="180" height="65" rx="8"/><rect x="230" y="20" width="240" height="65" rx="8"/><rect x="515" y="20" width="215" height="65" rx="8"/><rect x="775" y="20" width="230" height="65" rx="8"/><rect x="1050" y="20" width="205" height="65" rx="8"/></g><g stroke="#8ac9ff" stroke-width="2" marker-end="url(#a)"><path d="M185,52H227"/><path d="M470,52H512"/><path d="M730,52H772"/><path d="M1005,52H1047"/></g><g fill="#e5edf5" font-size="18" text-anchor="middle"><text x="95" y="59">原视频 + SAM2</text><text x="350" y="47">外包矩形作为模型洞</text><text x="350" y="71" font-size="14">边缘8px / 下方24px</text><text x="622" y="48">冻结 DriveEditor</text><text x="622" y="71" font-size="14">空对象条件 / seed42</text><text x="890" y="47">按旧精确 mask 写回</text><text x="890" y="71" font-size="14">mask 外保持原 RGB</text><text x="1152" y="59">r28 候选 DELETE</text></g><rect x="120" y="119" width="1020" height="45" rx="6" fill="#352d27"/><text x="630" y="148" fill="#e9bf88" font-size="17" text-anchor="middle">实测路面 → 可见区重投影 → 目标洞覆盖为零；本轮没有把这条证据支线接入生成</text></svg>
<p>旧方案把车的精确轮廓直接作为模型输入洞。新方案遮掉轮廓附近少量上下文，让生成器看到连续矩形洞；生成后仍使用同一精确 mask，因此邻车、施工护栏和其他区域在 mask 外的像素完全保留。模型洞面积变大约 2.2–2.6 倍，不等于最终改动面积变大。原始整帧输出另列，避免把写回接缝误判为模型生成内容。</p>
<h2>同一 10 帧：原视频 / 旧方案 / r28</h2><div class="controls"><button id="play">同步播放 / 暂停</button><button id="restart">回到首帧</button><input id="seek" type="range" min="0" max="9" step="1" value="0"><span id="frame">f0</span></div><div class="grid">'''
html+=vid('target','原视频 · 黄框目标 actor12','f0–f9，CAM0，10Hz，1 秒。仅原视频叠加目标标记；不是重新渲染的 factual。')
html+=vid('old','旧方案 · 精确轮廓直接生成','DELETE-REPAIR/r1 已保存结果。删除处呈灰色车形块；保留作对照。')
html+=vid('new','r28 · 矩形条件 + 精确写回','灰色车形块消失，路面更连续；请注意原车底部/右侧的亮色边界，f8–f9 附近还有局部线状接缝。')
html+='''</div><div class="note warn"><b>当前结论：保留这一改善，继续修写回；不是最终通过。</b>尚未验证 30 帧连续性、跨相机或 Ω 重建。两组前 10 帧使用相同窗口、原 RGB、权重、seed42、25 步和输出 mask，无前一窗口条件。矩形化同时增加遮挡上下文，不能只归因于“矩形形状”本身。</div>
<h2>区分生成与写回</h2><div class="grid">'''
html+=vid('native','r28 原生生成整帧','用于检查模型内部内容。整帧解码会改变 mask 外观感，不直接当最终输出。')
html+=vid('condition','模型实际遮挡范围','灰区显示归一化输入张量的零值。没有粘贴想象中的路面，也没有新训练/新模型。')
html+=vid('mask','蓝色模型洞 / 黄色写回边界','蓝色是生成条件的矩形；黄色是旧精确 mask。两者分开，才能定位边界接缝。')
html+='''</div><details open><summary>逐帧放大：原图 → 旧方案 → r28 写回 → r28 原生（全部 10 帧）</summary><p>每行在相同位置裁剪、相同尺度展示；没有按生成结果重新选区域。</p><a href="all10_part1.jpg"><img src="all10_part1.jpg" alt="第0至4帧四列对照"></a><a href="all10_part2.jpg"><img src="all10_part2.jpg" alt="第5至9帧四列对照"></a></details>
<details><summary>后方卡车为什么不能被“任意车辆检测”一票否决</summary><img class="wide" src="r28/neighbors_f000.jpg" alt="原图目标12与后方卡车34的投影框"><p>黄色是删除目标12，青色是保留卡车34。三个抽查时刻 f0/f5/f9 均有卡车34的投影框进入目标 mask，约747/714/560像素。这是遮挡后的真实保留对象包络，不等于已看到的邻车像素被误删，也不是隐藏纹理 GT。当前生成卡车的细节保真仍未认证。</p><a href="r28/neighbor_audit.json">查看深度与投影交叠记录</a></details>
<h2>按 A→B→C 拆开的真实路面证据</h2>
<table><tr><th>步骤</th><th>检查</th><th>实际结果</th><th>决定</th></tr><tr><td>A→B / r25</td><td>首帧 LiDAR 拟合地面，其他时刻原 RGB 搬到已可见路面</td><td>220 个留出点中位误差4.67mm；已可见纹理大体对齐，但只覆盖局部</td><td>允许继续检查洞，不能把可见区成绩当 DELETE 成绩</td></tr><tr><td>B→C / r26</td><td>保持同一地面范围，改为实际 actor12 删除洞</td><td>30帧170,227像素帧中，实测范围支持0</td><td>范围位于中道，目标在右侧；不扩大平面冒充证据</td></tr><tr><td>B→B1→C / r27</td><td>采用其他时刻实际测到的局部地面小三角形</td><td>目标洞仍0覆盖；已可见对照后两时刻也无支持</td><td>首帧平面不能当整段共同地面；暂不写回</td></tr></table>
<p>额外核对发现：f10、f20 可见道路 LiDAR 相对 f0 平面的中位偏离已约8.7cm、17.0cm，高于固定5cm范围。这里尚未区分道路变化、坐标/位姿处理和局部平面近似，不能直接宣布“背景没被拍过”或“相机错误”。下一步几何修复应从这个具体不一致开始，不能只放宽阈值。</p>
<details><summary>r25 实测路面、覆盖和真实原图对照</summary><img class="wide" src="r25/plane_points.jpg" alt="黄色可见道路验证区域和实测LiDAR点"><table><tr><th>查询帧</th><th>最近来源覆盖</th><th>该覆盖内 MAE</th><th>双时刻一致覆盖</th><th>该覆盖内 MAE</th></tr>'''+rows+'''</table><p>MAE 是0–255 RGB每通道绝对差。不同方法的覆盖不同，数值不能当同像素公平排名。紫色表示缺少来源，不是生成结果。</p>'''
for f in [0,10,20]:html+=f'<a href="r25/f{f:03}_comparison.jpg"><img src="r25/f{f:03}_comparison.jpg" alt="r25 f{f} 原图与真实来源重投影"></a>'
html+='''</details><details><summary>r27 的零覆盖反例：目标洞仍未得到实际路面支撑</summary><img src="r27/source_005_ground.jpg" alt="f5实际地面局部支持区域"><img src="r27/contact.jpg" alt="三列原图及目标洞没有支持的诊断"><p>紫色是待补洞，不是补景画面。没有把紫色画面或无来源的平面涂色交给 DriveEditor/Ω。</p></details>
<h2>三个 scene 的保留状态</h2><table><tr><th>scene / 目标</th><th>当前保留</th><th>下一处问题</th></tr><tr><td>scene_0255 / actor25</td><td>用户明确 r18＞r15；r21 局部围栏修复另存</td><td>保留真实actor52的外观、围栏写回和跨窗连续性</td></tr><tr><td>official_000 / actor12</td><td>本页r28作为改善候选，旧失败完整保留</td><td>精确写回接缝；再检查更长时序及卡车34保持</td></tr><tr><td>scene_0230 / actor22</td><td>旧结果不改写，本轮无新生成</td><td>无依据再生车辆；证据补景和语义仍未解决</td></tr></table>
<p>第三例已在旧修复中参与过开发，因此这三例用于防止只围绕0255做手工适配，不称独立泛化测试。GT相机、3D框与LiDAR属于本轮POC额外输入，尚不是仅RGB的全自动系统。</p>
<h2>可复核记录</h2><p><a href="r28/registration.json">r28配置与单项改动</a> · <a href="r28/state.json">推理资源</a> · <a href="validation.json">视频解码与像素合同</a> · <a href="r25/registration.json">r25</a> · <a href="r26/registration.json">r26</a> · <a href="r27/registration.json">r27</a></p><footer>本页7个视频共70帧已实际解码，原始PNG与全部失败保存在远端同task。已检查逐帧图片；浏览器交互未验证。GPU推理仅r28的10帧，无新Ω、GLB、训练、权重下载或MOVE。所有新候选 human_verdict=null，background_input_dir=null。</footer>
<script>
const videos=[...document.querySelectorAll('video[data-sync]')], seek=document.querySelector('#seek'), label=document.querySelector('#frame');
function jump(i){videos.forEach(v=>{v.pause();v.currentTime=Math.min(i/10,Math.max(0,(v.duration||1)-.01));});seek.value=i;label.textContent='f'+i;}
document.querySelector('#restart').onclick=()=>jump(0);seek.oninput=()=>jump(Number(seek.value));
document.querySelector('#play').onclick=()=>{const play=videos[0].paused,t=videos[0].currentTime;videos.forEach(v=>{v.currentTime=t;if(play)v.play().catch(()=>{});else v.pause();});};
videos[0].addEventListener('timeupdate',()=>{if(!videos[0].paused){const i=Math.min(9,Math.floor(videos[0].currentTime*10));seek.value=i;label.textContent='f'+i;videos.slice(1).forEach(v=>{if(Math.abs(v.currentTime-videos[0].currentTime)>.12)v.currentTime=videos[0].currentTime;});}});
</script></main></html>'''
extra='<h2>r29 接缝控制：退化，已停用</h2><div class="grid">'+vid('target','同一原视频','目标 actor12，后方保留 actor34。')+vid('new','保留 r28','亮边仍存在，但路面较连贯。')+vid('seam','r29 Poisson 写回 · 拒绝','固定同一生成图和精确 mask，只做 NORMAL_CLONE。结果出现更明显灰黑色块，不能替代 r28。')+'</div><p>这次只检验颜色写回，不把 Poisson 当作几何修复。10帧反例全部保存；不再对这个配置换模式或调参。四个时刻的精确 mask 内部孔洞数均为0，所以尚不能把线状边界归为“内部漏洞”；下一步应单独检查车底/阴影范围和生成图在边界的颜色、结构对应。</p><details><summary>r29 全部帧与 mask 边界审计</summary><img src="r29/all10_part1.jpg" alt="Poisson前五帧退化"><img src="r29/all10_part2.jpg" alt="Poisson后五帧退化"><img src="r28/mask_boundary_audit.jpg" alt="原图、精确mask、原生生成的边界放大"><a href="r29/registration.json">r29配置</a> · <a href="r28/mask_boundary_audit.json">内部孔洞检查</a></details>'
html=html.replace('<h2>三个 scene 的保留状态</h2>',extra+'<h2>三个 scene 的保留状态</h2>').replace('r25—r28','r25—r29').replace('本页7个视频共70帧','本页8个视频文件共80帧').replace('实测','源数据')
html=html.replace('额外核对发现：','几何来自已处理数据目录中的 LiDAR/pose 文件，本轮没有追溯原始传感器采集链。额外核对发现：')
(ROOT/'index.html').write_text(html,encoding='utf-8');print(ROOT/'index.html')
