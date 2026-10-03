"""CPU准备审核页：明确旧基线、真实输入先验与尚未产生的新输出。"""
from common import *
import html, shutil
import numpy as np, cv2
from PIL import Image
from actor_state import Observation, cuboid_front_depth
from prepare_inputs import box_axis

ARCHITECTURE='''<svg viewBox="0 0 1120 285" role="img" aria-label="RGB与BEV进入DriveEditor的条件接口">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#60819f"/></marker></defs>
<g fill="#142335" stroke="#557696"><rect x="10" y="18" width="185" height="54" rx="6"/><rect x="10" y="97" width="185" height="54" rx="6"/><rect x="10" y="176" width="185" height="54" rx="6"/>
<rect x="250" y="18" width="215" height="54" rx="6"/><rect x="250" y="97" width="215" height="54" rx="6"/><rect x="250" y="176" width="215" height="54" rx="6"/>
<rect x="520" y="54" width="220" height="54" rx="6"/><rect x="520" y="147" width="220" height="54" rx="6"/><rect x="785" y="90" width="180" height="85" rx="6"/><rect x="998" y="90" width="112" height="85" rx="6"/></g>
<g fill="#ecf2fa" font-size="14" text-anchor="middle" font-family="Arial, Microsoft YaHei">
<text x="102" y="41">目标 RGB + 独立 M</text><text x="102" y="60">原r21 SAM保持</text>
<text x="102" y="120">邻时刻 / 多相机 RGB</text><text x="102" y="139">先剔除 A，再裁参考</text>
<text x="102" y="199">LiDAR + pose + tracks</text><text x="102" y="218">不读隐藏 Y 像素</text>
<text x="357" y="41">官方遮洞 / 原9通道</text><text x="357" y="60">保留原条件路径</text>
<text x="357" y="120">冻结官方 RGB VAE</text><text x="357" y="139">参考外观 + camera/时间</text>
<text x="357" y="199">BEV / 2D包络 / 深度 / 轴向</text><text x="357" y="218">实测背景与未知分开</text>
<text x="630" y="77">参考 RGB 交叉注意力</text><text x="630" y="96">protected crop + context</text>
<text x="630" y="170">轻量 BEV backbone</text><text x="630" y="189">射线采样 + 2D几何控制</text>
<text x="875" y="119">DriveEditor UNet</text><text x="875" y="141">官方主干冻结</text><text x="875" y="161">新增支路零初始化</text>
<text x="1054" y="119">生成候选</text><text x="1054" y="141">硬 M 写回</text><text x="1054" y="161">保留洞外 RGB</text></g>
<g fill="none" stroke="#60819f" stroke-width="2" marker-end="url(#arrow)"><path d="M195 45H250"/><path d="M195 124H250"/><path d="M195 203H250"/><path d="M465 45H765V110H785"/><path d="M465 124H493V81H520"/><path d="M465 203H493V175H520"/><path d="M740 81H760V123H785"/><path d="M740 175H760V152H785"/><path d="M965 132H998"/></g>
<text x="560" y="264" text-anchor="middle" fill="#bfd0df" font-size="14">r47：CPU合同与输入准备；完整官方模型接入、训练和生成仍待 GPU</text></svg>'''


def main():
    assert read(O/'preflight.json')['CPU_ready']
    plan=read(O/'manifest.json');summary={c['case_id']:c for c in read(O/'input_summary.json')['cases']}
    user=read(O/'user_review/human_review.json');scores={c['case_id']:c for c in user['cases']}
    out=O/'review';out.mkdir(exist_ok=True);shutil.copytree(O/'user_review',out/'user_review',dirs_exist_ok=True)
    if (O/'input_visual_review.json').exists():shutil.copy2(O/'input_visual_review.json',out/'input_visual_review.json')
    cards=[];sourcefiles=[]
    for c in plan['cases']:
        cid=c['case_id'];folder=O/'inputs'/cid;dest=out/'assets'/cid;dest.mkdir(parents=True,exist_ok=True)
        for p in folder.glob('*.png'):shutil.copy2(p,dest/p.name)
        for name in ['result.json','instruction.json']:
            shutil.copy2(folder/name,dest/name)
        x=images(c,'rgb');h=images(c,'hole')>0
        for f in [0,5,9]:
            image=x[f].copy();contours,_=cv2.findContours(h[f].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image,contours,-1,(55,165,255),2)
            fr=c['frames'][f];ob=Observation(image,h[f],np.array(fr['camera_to_world']),np.array(fr['intrinsics_1024']),fr['timestamp'])
            if c['target_token']:
                mask=np.isfinite(cuboid_front_depth(ob,[a for a in fr['actors'] if a['instance_token']==c['target_token']]))
                if mask.any():
                    yy,xx=np.where(mask);cv2.rectangle(image,(int(xx.min()),int(yy.min())),(int(xx.max()),int(yy.max())),(255,210,60),2)
            Image.fromarray(image).save(dest/f'original_{f:02}.jpg',quality=92)
            geometry=np.asarray(Image.open(dest/f'geometry_{f:02}.png').convert('RGB')).copy()
            for actor in fr['actors']:
                if actor['instance_token'] not in summary[cid]['protected_instances']:continue
                axis,uv,valid=box_axis(actor,ob)
                if valid:
                    start,end=np.rint(uv).astype(int);cv2.arrowedLine(geometry,tuple(start),tuple(end),(255,225,75),2,tipLength=.12)
                    label='B'+str(summary[cid]['protected_instances'].index(actor['instance_token'])+1)
                    cv2.putText(geometry,label,tuple(start),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,235,75),2)
            Image.fromarray(geometry).save(dest/f'geometry_axes_{f:02}.png')
        row=summary[cid];refs=read(folder/'references.json')['references'];score=scores.get(cid)
        detail=f'<p>{html.escape(c["scene"])} · {html.escape(c["split"])} · 目标A：<code>{c["target_token"] or "synthetic occluder"}</code></p>'
        if score:
            detail+=f'<p>已有 r46 人工分：<strong>{score["human_score"]}</strong>；RGB足够：{score["RGB_prior_sufficient"] or "未填"}；OCC足够：{score["OCC_prior_sufficient"] or "未填"}。{html.escape(score["note"] or "")}</p>'
        detail+=f'<p>候选 RGB {row["reference_bank"]} 张，固定选择6个slot；实际选中相机：{html.escape(", ".join(row["selected_distinct_cameras"]))}。3D背景实测点 {row["background_3D_points"]}；额外参考待GPU遮罩验证 {row["source_mask_GPU_checks"]} 个。B仅按洞中投影重叠选作保护对象；包络不能认证实体轮廓。</p>'
        views=''
        for f in [0,5,9]:
            views+=f'<div class="frame"><h3>f{f:02}</h3><div class="four">'+''.join(
                f'<figure><figcaption>{label}</figcaption><img loading="lazy" src="assets/{cid}/{name}_{f:02}.{ext}"></figure>'
                for name,ext,label in [('original','jpg','原RGB · 黄A包络 / 蓝实际编辑M'),('target','png','实际目标输入 · M内先擦除'),('geometry_axes','png','2D控制 · 绿保留车包络 / 蓝LiDAR / 黄轴向'),('bev','png','BEV · 80m×80m · 蓝实测 / 绿保留车包络')])+ '</div></div>'
        references='<div class="six">'+''.join(f'<figure><figcaption>slot{r["reference_slot"]} · {html.escape(r["role"])}<br>{html.escape(r["camera"])} · {(r["timestamp"]-c["frames"][0]["timestamp"])/1e6:+.2f}s'+
            (' · 占位，不进入注意力' if r['padding'] else '')+(' · 源mask待GPU验证' if r['GPU_source_mask_validation_pending'] and not r['padding'] else '')+
            f'</figcaption><img loading="lazy" src="assets/{cid}/reference_{r["reference_slot"]:02}.png"></figure>' for r in refs)+'</div>'
        videos=''
        if c['kind']=='real':
            for name in ['original','input','adapter_off']:
                src=T/'r46/review/assets'/cid/(name+'.mp4');assert src.is_file();shutil.copy2(src,dest/(name+'.mp4'));sourcefiles.append(src)
            videos='<p><strong>以下是既有r46视频，供输入核对；不是r47新推理。</strong></p><div class="three">'+''.join(f'<figure><figcaption>{label}</figcaption><video muted playsinline controls preload="metadata" src="assets/{cid}/{name}.mp4"></video></figure>' for name,label in [('original','原视频与目标M'),('input','旧基线实际遮洞输入'),('adapter_off','官方原模型 + r21 SAM · 旧结果')])+'</div>'
        example=''
        if score and score['user_GPT_example']:
            example=f'<details><summary>用户工作簿中的GPT静态补景例（原样归档）</summary><p>{html.escape(score["user_GPT_success_note"] or "未填")}</p><img class="example" src="user_review/{score["user_GPT_example"]["path"]}"><p>外部用户实验，仅静态图片；不作本轮生成或视频时序结论。</p></details>'
        body=f'<h2 id="{cid}">{cid}</h2>'+detail+views+'<h3>真实RGB参考（保持纵横比；灰为剔除A或padding）</h3>'+references+videos+example+f'<p><a href="assets/{cid}/instruction.json">参数化指令</a> · <a href="assets/{cid}/result.json">逐帧输入统计</a></p>'
        cards.append('<article>'+body+'</article>' if c['kind']=='real' else '<details class="training"><summary>'+cid+' · '+c['scene']+' · '+c['split']+'</summary>'+body+'</details>')
    realcards=[card for card,c in zip(cards,plan['cases']) if c['kind']=='real']
    syntheticcards=[card for card,c in zip(cards,plan['cases']) if c['kind']=='synthetic']
    table=''.join(f'<tr><td>{c["case_id"]}</td><td>{c["human_score"] if c["human_score"] is not None else "未填"}</td><td>{c["RGB_prior_sufficient"] or "未填"}</td><td>{c["OCC_prior_sufficient"] or "未填"}</td><td>{html.escape(c["user_GPT_success_note"] or "未填")}</td></tr>' for c in user['cases'] if c['kind']=='真实DELETE')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>v77 r47 · RGB / BEV条件接口CPU准备</title>
<style>body{font:16px/1.65 system-ui,"Microsoft YaHei";background:#0d1520;color:#dde7f2;margin:0;padding:22px}main{max-width:1600px;margin:auto}a{color:#9fcfff}h1{font-size:27px}h2{font-size:22px}p{max-width:1300px}article,details.training{border:1px solid #33465b;margin:22px 0;padding:20px;border-radius:8px}.four,.three,.six{display:grid;gap:12px;align-items:start}.four{grid-template-columns:repeat(4,1fr)}.three{grid-template-columns:repeat(3,1fr)}.six{grid-template-columns:repeat(6,1fr)}img,video{width:100%;height:auto}figure{margin:0}figcaption{font-size:13px;min-height:40px}table{border-collapse:collapse}td,th{border:1px solid #486079;padding:7px 14px;text-align:left}.note{background:#172941;padding:15px;border-left:4px solid #91bbe7}.example{max-width:1000px}nav{display:flex;gap:20px;flex-wrap:wrap}svg{max-width:1120px;width:100%}@media(max-width:1000px){.four,.six{grid-template-columns:repeat(2,1fr)}.three{grid-template-columns:1fr}}</style><main>
<h1>v77 r47 · RGB + BEV 条件接口 · CPU准备</h1>
<p class="note"><strong>当前已完成输入与新接口的CPU检查，尚未运行r47训练/补景。</strong>沿用官方原始DriveEditor和r21完整SAM作基线；此前微调/Adapter不进入这轮权重。查询窗口与旧r46相同，额外先验是合法的离线多时刻/多相机观测，不宣称在线因果预测。</p>'''+ARCHITECTURE+'''
<p>新增389,856参数的参考RGB交叉注意力和轻量BEV CNN。参考图先擦除目标A，再由官方冻结RGB VAE编码；BEV由实测LiDAR与GT位姿/保留轨迹构建，经过多深度射线采样进入UNet四个尺度。2D图提供包络、深度、轴向、可见支持和未知。少量指令目前编译为固定语义数值字段，完整自然语言编码器未实现。</p>
<p>绿区是GT框支持包络，<strong>不是精确车身分割</strong>；蓝色LiDAR点是实测返回，未铺成稠密墙/道路；灰区未知，不能解释为空地。BEV中的非车辆包络为棕色，图像向右为世界+x、向上为世界+y，中心是f05相机位置，范围±40米。3D真值是POC辅助输入，当前不声称完全自动几何。</p>
<p>原始实现只提供deletion的遮后视频/编辑mask，禁用了目标物外观与位置条件；新增参考不能塞回旧的“生成目标车”字段。r47用独立条件支路进入生成过程，并在最后按原M硬写回。洞外PNG保持逐像素原样；MP4压缩展示不承担像素等价检查。</p>
<h2>用户更新评分（原文导入）</h2><table><tr><th>case</th><th>r46人工分</th><th>RGB足够</th><th>OCC足够</th><th>用户GPT结果说明</th></tr>'''+table+'''</table>
<p>八个真实例：7个1分，A022为2分；空白没有补0或推断结论。当前用户判断是mask修复有收益、微调及Adapter未建立明确真实任务收益。工作簿中的6个合成标签按原文保存，未无依据映射为当前r46的4个合成DEV。</p>
<p><a href="user_review/打分记录.xlsx">评分原件</a> · <a href="user_review/human_review.json">完整导入记录及单元格来源</a> · <a href="user_review/gpt补景pipeline.md">用户补景方案原件</a> · <a href="preflight.json">CPU接口检查</a></p>
<p><a href="input_visual_review.json">8个真实DEV的输入查看记录</a>：A034/A061有较清楚的保留车参考；A013/A048/A042仍有大片擦除；A007洞中对应的车在参考里很小。候选数增加不等于RGB证据充分，不填写新的人工评分，也不从单帧判断时序。</p>
<h2>下一步GPU小实验</h2><p>先验证已选额外相机参考的A剔除，不重选参考、不改查询SAM；失败参考slot停用。再做真实官方模型零初始化等价、前后向和显存探针；通过后从原权重冻结主干训练新支路320步。12训练/4合成DEV/8真实DEV保持不变，最终固定step320。训练时RGB与几何各独立以25%概率替换为未知码，避免只训练完整条件再拿从没见过的空条件对比。比较原基线、训练分支全未知、RGB-only、几何-only、RGB+几何；这是同一训练分支的输入消融，不是各臂独立训练。</p>
<p>真实收益依用户完整视频审核：是否删净、幻觉车是否减少、后车/邻车是否保住；单帧误差或CPU梯度通过均不能替代。若只有合成误差改善而真实任务无稳定增量，本轮不推广。当前12训练例的RGB参考仅来自原相机，真实DEV的多相机输入迁移还需验证；本轮不顺手扩大为完整world encoder/2DGS/surfel。</p>
<nav>'''+''.join(f'<a href="#{c["case_id"]}">{c["case_id"]}</a>' for c in plan['cases'] if c['kind']=='real')+'</nav>'+''.join(realcards)+'<h2>训练与合成DEV输入（16例，逐例展开）</h2>'+''.join(syntheticcards)+'</main>'
    (out/'index.html').write_text(page)
    shutil.copy2(O/'preflight.json',out/'preflight.json')
    dump(out/'delivery.json',{'cases':24,'real_cases':8,'existing_baseline_videos':24,'new_GPU_windows':0,
        'user_baseline_scores_imported':8,'all_new_output_scores':None,'architecture_parameters':389856})
    print('CPU_HTML_READY',out,flush=True)


if __name__=='__main__':main()
