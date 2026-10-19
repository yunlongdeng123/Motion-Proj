"""Evidence-driven report links and the current no-GPU export scope."""
import json

def banner(O,prefix=''):
    p=O/'cpu_export_media.json'
    if not p.exists():return ''
    r=json.loads(p.read_text(encoding='utf-8'))
    if r['status']!='READY_CPU_NATIVE_FOUR_GROUPS_AND_TWO_INDEPENDENT_EXPORTS':return ''
    return f'''<div class="state status"><h2>四组30K视频已可比较</h2>
<p><a href="{prefix}native30k_four_compare.html"><strong>四组同步总览与原生A对照：完整49帧、四通道</strong></a><br>
<a href="{prefix}independent_pair_compare.html">完整版与仅跨帧：独立49帧七通道，含删车与alpha</a></p>
<p>四组训练与全部24个保存节点已完成。用户切换无卡开机后，本轮只读下载并CPU编码；没有新增训练更新。完整版与仅跨帧的独立渲染已落盘；仅供体独立渲染在切机时中断，0张PNG、无完成凭据；仅归属独立渲染尚未开始。后两组的训练节点自带49帧完整，已在四组视频页展示；其独立删车/alpha仍需GPU。</p>
<p>已实际查看四组原生静态的全49帧：完整版仍有矩形道路纹理、暗条和接缝；仅跨帧持续有黄白亮块和蓝边；仅供体有明显黑块；仅归属保留黑色车状轮廓。均未达到干净静态道路要求。动态、完整重建、alpha及来源区域的全量质量对照尚待完成，不能据此归因到单个内部约束。<a href="{prefix}native_static_review.json">本轮静态实际观看范围与结论</a>。</p>
<p><a href="{prefix}cpu_export_validation.json">34个视频全1666帧解码与同步控制检查</a> · <a href="{prefix}cpu_export_sources.json">来源与真实完成范围</a>。文件/解码/JS fixture通过，未宣称浏览器UI验收；当前仍未作质量通过判断。</p></div>'''
