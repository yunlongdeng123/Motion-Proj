"""仅发布当前边界诊断代码与凭据；不接触其他用户改动。"""
from pathlib import Path
import json,tarfile,shutil,datetime,subprocess
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
NEW=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/unknown_detection_v3')
CODE=REPO/'scripts/worldsim_v82/v43_ownership_20261028/unknown_detection_v3/bounded_white_v1'
DOC=REPO/'docs/autoresearch/v43_ownership_20261028'
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();paths=[]
backup=NEW/'publication_history'/(stamp.replace(':','')+'_bounded')
def put(p,b):
    if p.exists():
        q=backup/p.relative_to(REPO);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.suffix in ['.py','.js','.cjs','.json','.md','.html','.svg']:b=b.replace(b'\r\n',b'\n')
    p.write_bytes(b);paths.append(str(p.relative_to(REPO)))
with tarfile.open(NEW/'bounded_publication.tar.gz') as t:
    for m in t.getmembers():
        p=(REPO/m.name).resolve();assert m.isfile() and p.is_relative_to(REPO) and not m.issym();put(p,t.extractfile(m).read())
put(REPO/'docs/RESEARCH_STATUS.md','''# 当前研究状态

用户最新验收收紧为：以 existing 为基线，蓝框外禁止新增 unknown；框内车底黑影与彩噪应白底显示。本轮三帧 raw98/113/114，训练更新0，第三场景保持停止。

已交付两种明确区分的图：局部全白范围预览（二维门控，非归属成功）与真实GPU候选剥离。真实新候选要求投影中心在用户蓝色多边形内，新增14874/8485/7936行，域外新中心0。每轮删除后重算 alpha*T；残留421/326/345个域外中心的支撑仍伸进查询域，没有将它们整颗标unknown。真实白底为 G_after+(1-alpha_after)*1；框外取同帧existing以保护跨界支撑。实际几何不是立体box约束，只是当前帧投影范围。

纯白范围预览满足显示边界要求；真实渲染仍有浅暗边/细彩纹，未通过完整去影验收。暗像素阈值计数为0不等于纯白或目视去影成功。未修改颜色、checkpoint、环境参数，无补景。没有获得语义P/R或跨场景泛化证据，不能将查询区域标签用于训练。

[算法与验收范围](autoresearch/v43_ownership_20261028/BOUNDED_WHITE_V1.md)。共享工作台保持同入口，当前页并列两种图，独立AI目视复核 human_verdict=null。下一步由用户验收局部范围；真实剥离若继续，需要处理跨边界混合支撑，不能扩大框外整行unknown来凑召回，也不能把显示全白当已解决身份归属。

CPU范围预览加约5.51秒GPU冻结渲染，峰值reserved约1.48GiB；CPU分量组合和完整结果均已取回。GPU资源只在提交推送、取回与活动进程检查后关闭。failure_ledger_refs=[V82-F01]；failure_ledger_delta=none。
'''.encode())
p=REPO/'docs/EXPERIMENTS.md';s=p.read_text();row='| WS-V43-TARGET-DETECTION-20261028 / bounded-native-r1+local-r2 | 用户蓝框硬上限；真实中心候选剥离与纯白范围预览分开验收，仍有跨界支撑，训练0 | [范围与结果](autoresearch/v43_ownership_20261028/BOUNDED_WHITE_V1.md) |'
if 'bounded-native-r1+local-r2' not in s:s=s.replace('|---|---|---|','|---|---|---|\n'+row,1)
put(p,s.encode())
p=DOC/'STATE.json';s=json.loads(p.read_text());s['checked_utc']=stamp;s['target_detection'].update(phase='bounded_white_user_review',latest_report='BOUNDED_WHITE_V1.md',background_precision_gate='hard user query boundary for new Gaussian centers',query_raws=[98,113,114],native_clean_removal_verified=False,training_updates=0,semantic_recall_available=False,third_scene_held=True,next='User reviews bounded scope; investigate cross-boundary Gaussian support without relabelling outside rows')
put(p,json.dumps(s,indent=2,ensure_ascii=False).encode())
subprocess.run(['git','add','--']+paths,cwd=REPO,check=True)
# 上一阶段只漏发布的已验Canvas检查脚本一并收尾，非浏览器绕过脚本。
p='scripts/worldsim_v82/v43_ownership_20261028/unknown_detection_v3/recall_upperbound/test_recall_canvas.cjs'
if (REPO/p).exists():subprocess.run(['git','add','--',p],cwd=REPO,check=True)
subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
print('bounded staged',len(paths))
