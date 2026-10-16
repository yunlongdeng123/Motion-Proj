import json,math,re,hashlib,shutil,subprocess,datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
BASE=Path(__file__).resolve().parents[2];OUT=BASE/'outputs/shadow_sequence_v2'
def load(p):return json.loads((OUT/p).read_text(encoding='utf-8'))
def main():
    final=load('final49_v1/summary.json');rows=load('final49_v1/per_frame.json');training=load('joint_v1/status.json')
    data=load('surface_guarded_data/summary.json');actor=load('joint_v1/actor_contamination.json');curve=load('joint_v1/validation_curve.json')
    m=final['metrics_mean_frames'];fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    x=np.arange(89,138)
    for v,label,col in [('original','Original checkpoint','#6e7884'),('deferred','Prior deferred_R4','#e18b28'),('final','Staged joint','#2375ba')]:
        axes[0,0].plot(x,[r[v]['surface_clean_MAE'] for r in rows],label=label,c=col)
        axes[0,1].plot(x,[r[v]['surface_shadow_MAE'] if r[v]['surface_shadow_MAE'] is not None else np.nan for r in rows],label=label,c=col)
    axes[0,0].set_title('Visible supported road: clean reference MAE');axes[0,1].set_title('Shadow pseudo-label MAE (linear RGB)')
    axes[1,0].plot(x,[r['fixed_supported_road_MAE_vs_first'] or 0 for r in rows],c='#2375ba',label='Supported road')
    axes[1,0].plot(x,[r['fixed_full_image_MAE_vs_first'] for r in rows],c='#b07d33',label='Whole image')
    axes[1,0].set_title('Fixed camera static change, sRGB MAE')
    axes[1,1].plot([r['global_step'] for r in curve],[r['mean_clean_surface_MAE'] for r in curve],label='Clean road',c='#2375ba')
    axes[1,1].plot([r['global_step'] for r in curve],[r['mean_shadow_surface_MAE'] for r in curve],label='Shadow',c='#b07d33');axes[1,1].set_title('7-frame optimization monitoring (not test)')
    for ax in axes.flat:ax.grid(alpha=.25);ax.legend(fontsize=8);ax.set_xlabel('Raw frame' if ax is not axes[1,1] else 'Gradient update')
    fig.savefig(OUT/'measured_curves.png',dpi=160);plt.close(fig)
    table=''
    for key,label in [('surface_clean_MAE','可见支持道路 / 无车影参考'),('surface_shadow_MAE','车影伪标注'),('surface_observed_clean_MAE','真实干净观测'),('old_local_visible_clean_MAE','旧局部窗口 / 可见且无遮挡道路')]:
        table+='<tr><td>'+label+'</td>'+''.join('<td>%.6f</td>'%m[v][key] for v in ['original','deferred','final'])+'</tr>'
    visual=load('evidence/visual_review.json') if (OUT/'evidence/visual_review.json').exists() else {'status':'PENDING_PRIMARY_VISUAL_REVIEW'}
    gates={
        'all49_evaluated':len(rows)==49,
        'delete_all_shadow_identity':final['max_delete_identity']<2e-6,
        'road_clean_MAE_improves_parent':m['final']['surface_clean_MAE']<m['original']['surface_clean_MAE'],
        'supported_road_time_MAE_at_most_1e_5':final['mean_fixed_supported_road_time_MAE']<=1e-5,
        'outside_scenery_max_at_most_1e_5':final['max_scenery_change_outside_road']<=1e-5,
        'complete_visual_review':visual.get('status')=='REVIEWED'}
    (OUT/'quality_gates.json').write_text(json.dumps(gates,ensure_ascii=False,indent=2),encoding='utf-8')
    def video(name,title):return f'<article><h3>{title}</h3><video controls preload="metadata" src="videos/final49_v1/{name}.mp4"></video><p><a href="videos/final49_v1/{name}.mp4">下载视频</a></p></article>'
    videos=''.join([video('comparison','原 checkpoint → 旧方案 → 联合微调 → 删除车辆'),video('delete_target','连续删除指定车辆'),video('translate_smooth','连续平移 ±1 米'),video('yaw_smooth_15','连续旋转 ±15°'),video('yaw_smooth_30','连续旋转 ±30° / 资产缺陷检查'),video('fixed_static','固定相机、49 个时刻的静态背景'),video('fixed_translate','固定相机下运动与连续平移'),video('shadow_effect','阴影开关产生的实际道路差异'),video('confidence','绿色：本次拥有恢复依据的道路范围')])
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>49 帧道路去影与联合微调实测</title>
    <style>body{{font:16px/1.65 'Microsoft YaHei',sans-serif;background:#f5f8fc;color:#243244;max-width:1180px;margin:30px auto;padding:0 22px}}h1{{font-size:29px}}h2{{margin-top:38px}}.card,article{{background:white;padding:20px;border-radius:14px;margin:18px 0}}.notice{{border-left:5px solid #d88722}}img{{max-width:100%;height:auto}}video{{width:100%;max-height:640px;background:#111}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(410px,1fr));gap:18px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;border-bottom:1px solid #dae2ea;text-align:left}}code{{font-size:13px;word-break:break-all}}summary{{cursor:pointer;color:#1768a0}}a{{color:#12689c}}</style>
    <h1>完整 49 帧诊断与分阶段联合微调</h1><p>scene0008，raw89–137；同一训练序列重建检查。仅报告真实执行结果。</p>
    <div class="card notice"><b>训练已执行，完整视频与逐帧指标已生成；质量结论以视觉验收和下方门槛为准。</b><p>当前视觉验收状态：{visual['status']}。新道路覆盖 {data['supported_surface_area_m2']:.1f} m² 的候选支持区域，内部可替换区域约 {data['interior_texels']*data['grid_spacing_m']**2:.1f} m²。未观测位置、保守保护区域和原车体资产缺陷不能被计为已解决。</p></div>
    <img src="architecture.svg" alt="输入、道路恢复、车身几何阴影与场景合成组件图">
    <h2>用真实观测重新分配道路与阴影</h2><p>49 帧 RGB 与 LiDAR 给出道路的固定坐标表面。至少三个相符观测、跨至少六个帧索引，并通过 LiDAR 距离筛选后才建立道路参考。上分位稳定观测用于抑制暂态暗影；这仍是可信伪参考，不是独立标注的无影真值。</p>
    <p>道路分支新增 {data['supported_texels']:,} 个固定平面高斯，8 cm 网格；优化颜色、透明度。原 checkpoint 初始化并提供其余场景与车辆。道路范围外回到原场景；未选车辆和行人的三维标注投影形成保守保护区域。训练渲染保留深度排序，仅对可见道路贡献施加阴影。</p>
    <p>投影遮挡体来自现有车身高斯几何的纵向分段包围盒，最多五段/Actor。光照、RGB 阴影强度、柔度和受限的代理形状参数由 RGB 拟合。编辑时读取新的车辆位姿重新投影；不输入目标 RGB，也没有按时刻自由预测阴影的网络。它仍是几何近似，未实现精细高斯密度光线积分。</p>
    <h2>实际训练与对照</h2><table><tr><th>阶段</th><th>梯度更新</th><th>优化内容</th></tr><tr><td>恢复背景</td><td>1,500</td><td>道路颜色、透明度；固定阴影</td></tr><tr><td>拟合阴影</td><td>2,000</td><td>固定道路；共享光照、柔度、阴影强度、分段遮挡体</td></tr><tr><td>小学习率联合</td><td>4,500</td><td>道路、阴影、受限代理形状及确认候选的 Actor 透明度</td></tr></table>
    <p>实际更新共 {training.get('training_updates',0):,}；训练采样 46 帧，raw101/113/125 不参与本轮梯度采样。但参考构建使用 49 帧，原 checkpoint 也见过全部 49 帧，因此不声称留出测试或泛化。阴影标签由亮度比、颜色一致性、道路/车身范围和邻车条件筛选；没有独立密集车影 GT。</p>
    <p>所有阶段保存了真实 addon 和 Adam 状态，重新加载后 raw101 图像逐值一致；源模型参数版本和原 checkpoint 文件身份保持不变。最初未经全场景车辆保护的扩大替换试验损坏银车外观，已在训练前否决，并保留失败记录。</p>
    <table><tr><th>线性 RGB MAE，49 帧等权均值</th><th>原 checkpoint</th><th>旧 deferred_R4</th><th>本轮联合</th></tr>{table}</table>
    <p>原 checkpoint 的已烘焙阴影可以直接匹配原 RGB；车影误差低并不代表可编辑。道路与车影需分别检查。旧局部“全部参考点”指标包含出视野或被其他车辆遮挡的位置，本报告另给可见且无车身遮挡的公平子集。</p>
    <img src="measured_curves.png" alt="逐帧道路、阴影、固定相机稳定性与优化曲线">
    <h2>连续视频</h2><p>每个视频完整包含 49 帧；10 fps 回放，未做抽帧。旋转视频用道路法线作旋转轴；原资产已有的破面和视角缺失单独标记。</p><div class="grid">{videos}</div>
    <h2>自验证与尚未达标的部分</h2><table><tr><th>检查</th><th>结果</th></tr>{''.join('<tr><td>'+k+'</td><td>'+str(v)+'</td></tr>' for k,v in gates.items())}</table>
    <p>固定相机道路内部平均时间 MAE：{final['mean_fixed_supported_road_time_MAE']:.8f}；整幅画面：{final['mean_fixed_full_image_time_MAE']:.8f}。删除全部 Actor 后，与无阴影静态图的最大差：{final['max_delete_identity']:.8g}。本次道路范围外、排除 Actor 车身后的最大变化：{final['max_scenery_change_outside_road']:.8g}。</p>
    <p>Actor 检查发现 {actor['persistent_outside_observed_body_and_geometry_candidates']} 个严格候选（共 {actor['total_actor_points']:,} 点）。候选依据三维框外位置和跨帧二维投影，尚不是完整可见性归属证明。本轮只开放这些候选的透明度，不宣称完成车身材质、自阴影、反射分解，也不能保证车辆资产完全无道路成分。大角度旋转的车体缺陷需从车辆资产本身继续修复。</p>
    <p>恢复只在观测支持的道路和保守保护范围之外生效；阴影跨出该范围可能被截断。局部固定平面不能表达整场景道路起伏。没有配对编辑 GT，复制/替换 Actor、完整重光照和从未观察的道路尚未认证。工程阶段有不同优化预算，不能作为等算力论文消融。</p>
    <h2>交付与复现</h2><p><a href="final49_v1/per_frame.json">全部逐帧指标</a> · <a href="joint_v1/validation_curve.json">阶段检查曲线</a> · <a href="joint_v1/gradient_smoke.json">真实 GPU 梯度</a> · <a href="joint_v1/runtime_contract.json">渲染与监督约定</a> · <a href="quality_gates.json">质量门槛</a> · <a href="evidence/visual_review.json">视觉审查</a></p>
    <p>远端保存：<code>/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_sequence_20261016_v2/joint_v1/final.pt</code>。这是依赖原 checkpoint、道路数据和新渲染器的 addon，不能直接替换完整原模型。实现与执行脚本位于本交付 scripts/，原文件保持只读；没有更改 GPU 电源状态。</p>
    <details><summary>49 帧逐帧画面对照（全部帧）</summary>{''.join('<img src="full49_review_'+str(k)+'.jpg">' for k in range(1,5))}</details></html>'''
    (OUT/'report.html').write_text(html,encoding='utf-8')
    print('Report written:',OUT/'report.html')
if __name__=='__main__':main()
