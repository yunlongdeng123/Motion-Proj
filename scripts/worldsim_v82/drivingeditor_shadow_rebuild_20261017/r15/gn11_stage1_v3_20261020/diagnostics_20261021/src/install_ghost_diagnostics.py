"""Install bounded offline R trace cases; retain all historical experiments."""
from pathlib import Path
import shutil,json,re

D=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-20/file-c-users-dengyunlong-documents-codex-2/outputs/GS_Reconstruction_Diagnostics_20261021_Evidence_r1')
HERE=Path(__file__).resolve().parent

def patch_app(s):
    # Keep the scientific channel names and the metrics' actual binding explicit.
    old="legend='原生渲染派生图像 · 完整480×320 · 显示编码非指标输入';"
    new="legend=(channel==='native_v1'?'原V1完整通路：本组原静态＋原动态＋环境，无道路；同checkpoint，非独立B。':channel==='native_static_delete'?'原静态＋环境：无动态、无道路。':channel==='deleted_five'&&arm==='v4_1_R'?'编辑候选：原静态＋道路＋环境，动态完全移除。':channel==='joint'&&arm==='v4_1_R'?'道路联合合成诊断：原静态＋原动态＋道路＋环境。':'')+' 原生渲染派生图像 · 480×320 · 显示编码非指标输入';"
    assert s.count(old)==1;s=s.replace(old,new)
    s=s.replace("${active?'已读取所选像素原生保存浮点", "${S.arm==='v4_1_R'?'以下固定为道路联合/删车对照指标，不随上方RGB通道切换。 ':''}${active?'已读取所选像素原生保存浮点")
    old="else $('rayState').textContent=`当前像素／模式 ${S.rayMode} 无离线全模型射线。可点击实时CUDA归因，或用CPU作独立参考；不以邻近像素替代。`;"
    new="else $('rayState').textContent=S.arm==='v4_1_R'?'V4.1已接入raw102五个固定像素、四种模式的原生CUDA离线归因。当前选择未覆盖；任意像素实时接口尚未开放，不用邻近像素替代。':`当前像素／模式 ${S.rayMode} 无离线全模型射线。可点击实时CUDA归因，或用CPU作独立参考；不以邻近像素替代。`;"
    assert old in s;s=s.replace(old,new)
    return s

def patch_gpu(s):
    old="$('cfState').textContent='V4：当前可看49帧原生输出、几何和深度；逐Gaussian trace与删除干预尚未导出。';$('viewState').textContent='V4原生结果见左侧二维通道；不调用旧V3/V1的GPU绑定。';return;"
    new="""$('viewState').textContent='本组原生结果见左侧二维通道；任意视角实时接口尚未开放。';
    $('cfState').textContent=s.arm==='v4_1_R'?'V4.1：raw102五个固定像素已有预计算贡献追踪及固定集合移除。任意集合实时接口尚未开放。':'V4：逐Gaussian trace与删除干预尚未导出。';
    const frozenKey=`cf/${s.arm}/${s.raw}/${s.x}/${s.y}`,frozenPath=window.GSDATA?.gpu_index?.counterfactuals?.[frozenKey];
    if(s.arm==='v4_1_R'&&s.rayMode==='deleted_five'&&frozenPath){try{const value=await window.gsdiag.load(frozenKey,frozenPath,false);if(token===generation)showCF(value,'已执行冻结诊断：本像素Tα最高的20个普通静态高斯；不是安全删除建议，不代表当前复选集合');}catch(error){if(token===generation)$('cfState').textContent='离线干预读取失败：'+error.message;}}return;"""
    assert s.count(old)==1;s=s.replace(old,new)
    return s

def patch_index(s):
    tag='<script src="data/gpu_index.js"></script>'
    assert s.count(tag)==1
    s=s.replace(tag,tag+'<script src="v41_ghost_index.js?v=ghost1"></script>')
    s=s.replace('</nav>','<a href="V41_GHOST_DIAGNOSIS.html">中心残影定位 ↗</a></nav>',1)
    s=s.replace('app.js?v=v41r30k1','app.js?v=v41ghost1').replace('gpu.js?v=v4results1','gpu.js?v=v41ghost1').replace('v41_experiments.js?v=v41r30k1','v41_experiments.js?v=v41ghost1')
    return s

def patch_registry(s):
    s=s.replace("g.defaultChannel='joint'","g.defaultChannel='native_v1'")
    s=s.replace('同checkpoint · 原V1完整重建','原V1完整通路 · 本组checkpoint（默认）')
    s=s.replace('"gaussian_trace_available":false','"gaussian_trace_available":true,"gaussian_trace_scope":"raw102 five fixed pixels; ordinary/deleted_five/joint/road_only; offline only"')
    return s

def patch_evidence(s):
    s=s.replace("$('geometrySummary').textContent='当前为CPU参考或旧数据，没有测量判别；保持未知。';", "$('geometrySummary').textContent=value.arm==='v4_1_R'?'当前为已核验原生CUDA贡献；逐高斯合法/冲突几何标签尚未完成，保持未知。':'当前为CPU参考或旧数据，没有测量判别；保持未知。';")
    s=s.replace("evalPanel.textContent='该旧干预尚未进行固定世界有效区评价，不用全图变化冒充改善。';", "evalPanel.textContent=value.arm==='v4_1_R'?'本次已记录目标车mask外观测损伤，但尚未完成固定世界有效路面验收；不把全图变化当隐藏路面改善。':'该旧干预尚未进行固定世界有效区评价，不用全图变化冒充改善。';")
    return s

def install():
    backup=HERE/'before_ghost';backup.mkdir(exist_ok=True)
    for name in ['app.js','gpu.js','evidence.js','index.html','data/v41_experiments.js']:
        dest=backup/name;dest.parent.mkdir(exist_ok=True,parents=True)
        if not dest.exists():shutil.copy2(D/name,dest)
    functions={'app.js':patch_app,'gpu.js':patch_gpu,'evidence.js':patch_evidence,'index.html':patch_index,'data/v41_experiments.js':patch_registry}
    for name,fn in functions.items():
        original=(backup/name).read_bytes().decode('utf-8');(D/name).write_bytes(fn(original).encode('utf-8'))
    for source in (HERE/'results').rglob('*'):
        if source.is_file():
            target=D/source.relative_to(HERE/'results');target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(source,target)
    # Fixed presets are meaningful only for raw102; no invented data elsewhere.
    extra="""
(function(){const p=document.getElementById('preset');for(const [v,t] of [['102,255,214','raw102 · 中心残影彩色上缘'],['102,255,290','raw102 · 中心下方拖影']]){if(!Array.from(p.options).some(o=>o.value===v)){const o=document.createElement('option');o.value=v;o.textContent=t;p.append(o);}}})();
"""
    (D/'v41_ghost_index.js').write_text((D/'v41_ghost_index.js').read_text(encoding='utf-8')+extra,encoding='utf-8')
    for name in ['app.js','gpu.js','evidence.js','index.html']:
        if (D/'source/viewer'/name).exists():shutil.copy2(D/name,D/'source/viewer'/name)
    print('installed original package; offline GPU cases + default native V1 image, historical channels retained')

if __name__=='__main__':install()
