"""CPU encode existing images with separate native and independent-render scopes."""
from pathlib import Path
import datetime,hashlib,json,re,shutil,subprocess,uuid
from build_native30k_compare import probe,sha

W=Path(__file__).resolve().parent
O=W.parent.parent/'outputs/gn11_stage1_v2_20261019'
ARMS={'full':('完整版 · 30K','归属修复＋跨帧真实观测＋供体先验'),
      'real_only':('仅跨帧 · 30K','归属修复＋真实观测；供体权重0'),
      'prior_only':('仅供体 · 30K','归属修复＋供体先验；真实道路目标权重0'),
      'ownership_only':('仅归属修复 · 30K','两种道路目标权重均0'),
      'baseline':('原生A · 第一阶段30K','原生机制；无本轮归属修复与道路目标')}
NATIVE_CHANNELS={'static':'静态分支','dynamic':'动态分支','composite':'完整重建','gt':'原始观测'}
FINAL_CHANNELS={'static':'静态分支','dynamic':'动态分支','full':'完整重建','delete_all':'删除纳入车辆（动态关闭）','static_alpha':'静态alpha','dynamic_alpha':'动态alpha','gt':'原始观测'}
MEDIA=O/'cpu_export_media.json'

def encode(key,pattern,records):
    sources=[Path(str(pattern).replace('%07d',f'{i:07d}')) for i in range(8089,8138)]
    digest=hashlib.sha256(json.dumps([sha(p) for p in sources],separators=(',',':')).encode()).hexdigest()
    target=O/'cpu_export_videos'/(key+'.mp4');old=records.get(key)
    if target.exists():
        if not old or old['source_sha256']!=digest or sha(target)!=old['sha256']:raise RuntimeError('preserve existing unmatched media '+key)
    else:
        temp=target.with_name(target.stem+'.encoding_'+uuid.uuid4().hex[:8]+'.mp4')
        subprocess.run(['ffmpeg','-v','error','-nostdin','-framerate','10','-start_number','8089','-i',str(pattern),'-frames:v','49','-an',
          '-c:v','libx264','-preset','slow','-crf','14','-pix_fmt','yuv420p','-g','1','-keyint_min','1','-sc_threshold','0','-bf','0','-threads','4','-movflags','+faststart',str(temp)],check=True)
        probe(temp);temp.rename(target)
    poster=target.with_name(target.stem+'_raw089.png');shutil.copyfile(sources[0],poster)
    item={'path':target.relative_to(O).as_posix(),'poster':poster.relative_to(O).as_posix(),'source_sha256':digest,'sha256':sha(target),
      'bytes':target.stat().st_size,'source_pattern':str(pattern.relative_to(W)).replace('\\','/'),**probe(target)}
    records[key]=item;MEDIA.write_text(json.dumps({'videos':records},indent=2),encoding='utf-8')
    return item

def page(name,title,notice,models,channels,comparisons,default,scope):
    data={'models':models,'channels':channels,'comparisons':comparisons,'default_comparison':default,'fps':10,'first':89,'last':137}
    js=(W/'final_full49_player.js').read_text(encoding='utf-8')
    a="[['full,real_only','完整版 ↔ 仅跨帧'],['full,prior_only','完整版 ↔ 仅供体'],['full,ownership_only','完整版 ↔ 仅归属'],['ownership_only,baseline','仅归属 ↔ 原生A'],['all4','四组总览（2×2）'],['all5','四组＋原生A']]"
    assert a in js;js=js.replace(a,'DATA.comparisons').replace("$('comparison').value='full,real_only'","$('comparison').value=DATA.default_comparison")
    style=re.search(r'<style>(.*?)</style>',(W/'native30k_compare_template.html').read_text(encoding='utf-8'),re.S)[1]
    # At desktop width each panel displays the complete 480x320 image without stretching.
    style+='video{width:480px;max-width:100%;height:auto}.panel{overflow:hidden} .grid{align-items:start}table{display:block;overflow:auto}'
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>@@TITLE@@</title><style>@@STYLE@@</style><main>
<p class="badge">实际30K · raw89–137 · 全49帧 · 完整480×320</p><h1>@@TITLE@@</h1>
<p class="notice">@@NOTICE@@</p><p><a href="native30k_four_compare.html">四组原生30K＋原生A对照</a> · <a href="independent_pair_compare.html">已完成独立渲染的两组七通道</a> · <a href="index.html">方法报告</a></p>
<div class="controls"><div class="row"><label>对照 <select id="comparison"></select></label></div><div class="row" id="channels"></div>
<div class="row"><button id="play" disabled>同步播放</button><button id="previous" disabled>上一帧</button><button id="next" disabled>下一帧</button>
<label>速度 <select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><label><input type="checkbox" id="loop" checked>循环</label></div>
<div class="row"><input id="seek" type="range" min="0" max="48" step="1" value="0" aria-label="共同帧位置"><output id="frameLabel"></output></div>
<div class="row" id="anchors"></div><div id="status" role="status"></div><div id="error" role="alert"></div></div>
<div class="grid" id="modelGrid"></div><p class="muted">动态分支为五身份a6/a37/a5/a26/a18的集合；没有第二阶段逐身份独立资产。所有画面完整显示，没有裁剪或局部放大。</p>
<details open><summary>对照设计与来源</summary><img class="arch" src="ablations/architecture.svg" alt="相同输入经过归属修复与真实道路、供体道路两个开关，四组30K输出对照的模块箭头图">
<p>四组共享归属修复，以跨帧真实观测、供体先验形成2×2开关。仅归属组与原生A对照归属修复组合。冻结query与颜色不变，UNKNOWN始终无道路目标；模块内部细项尚未单独消融。@@SCOPE@@</p>
<p>49帧均为训练视图，使用额外3D标注框归属约束；不是heldout。旧原A初始GPU张量不可逐位核对。未重算投影阴影，不能把删去五身份动态集合称为全画面无车。</p></details>
<details><summary>完整视频与验证凭据</summary><table><thead id="downloadHead"></thead><tbody id="downloads"></tbody></table>
<p><a href="cpu_export_media.json">视频与来源hash</a> · <a href="cpu_export_sources.json">本轮PNG逐文件清单及独立渲染状态</a> · <a href="cpu_export_validation.json">全部解码与播放器控制检查</a></p></details>
<footer>CPU只读导出，无新增训练更新。浏览器UI未验收；质量结论见实际全画面复核记录，human_verdict=null。<p><a href="ablations/index.html">实验记录</a> · <a href="native30k_compare.html">最初两组视频页</a></p></footer>
</main><script type="application/json" id="data">@@DATA@@</script><script>@@JS@@</script></html>'''
    for key,value in {'TITLE':title,'STYLE':style,'NOTICE':notice,'SCOPE':scope,'DATA':json.dumps(data,ensure_ascii=False,separators=(',',':')),'JS':js}.items():text=text.replace('@@'+key+'@@',value)
    (O/name).write_text(text,encoding='utf-8')

def main():
    download=json.loads((W/'cpu_exports_download.json').read_text(encoding='utf-8'));assert download['all_member_hashes_verified']
    (O/'cpu_export_videos').mkdir(exist_ok=True)
    old=json.loads(MEDIA.read_text(encoding='utf-8')) if MEDIA.exists() else {'videos':{}}
    records=old['videos'];pair=json.loads((O/'native30k_media.json').read_text(encoding='utf-8'))
    native={}
    for arm in ARMS:
        videos={};native[arm]={'label':ARMS[arm][0],'description':ARMS[arm][1],'videos':videos}
        for channel in NATIVE_CHANNELS:
            if arm in ['full','real_only']:
                item=pair['videos'][arm+'_'+channel].copy();assert sha(O/item['path'])==item['sha256']
                item.update(source_pattern=f'native30k_pair/{arm}/{channel}/%07d.png',reused_existing_media=True)
            elif arm=='baseline':
                key={'gt':'observed','composite':'full'}.get(channel,channel)
                p=O.parent/'baseline_engineering_20261018/branch_videos'/('gn11five_'+key+'.mp4')
                item={'path':'../baseline_engineering_20261018/branch_videos/'+p.name,'sha256':sha(p),'bytes':p.stat().st_size,**probe(p),
                  'source_pattern':f'../baseline_engineering_20261018/GN11_stage1_full49/%07d/{"full" if channel=="composite" else channel}.png','reused_existing_media':True}
            else:item=encode('native_'+arm+'_'+channel,W/'cpu_exports/native'/arm/channel/'%07d.png',records)
            videos[channel]=item;records['native_'+arm+'_'+channel]=item
    independent={}
    for arm in ['full','real_only']:
        videos={};independent[arm]={'label':ARMS[arm][0],'description':ARMS[arm][1],'videos':videos}
        for channel in FINAL_CHANNELS:videos[channel]=encode('independent_'+arm+'_'+channel,W/'cpu_exports/independent'/arm/'%07d'/(channel+'.png'),records)
    comparisons=[['full,real_only','完整版 ↔ 仅跨帧'],['full,prior_only','完整版 ↔ 仅供体'],['full,ownership_only','完整版 ↔ 仅归属'],['real_only,prior_only','仅跨帧 ↔ 仅供体'],['ownership_only,baseline','仅归属 ↔ 原生A'],['all4','四组总览（2×2）'],['all5','四组＋原生A']]
    page('native30k_four_compare.html','四组30K完整视频对比',
      '四组训练均已完成30K；本页统一使用各组30K节点已保存的原生49帧，提供原图、完整重建、静态、动态四通道。切换无卡后仅进行CPU下载与编码。完整版/仅跨帧的独立七通道见另页；仅供体/仅归属的独立渲染仍需GPU，本页没有将原生静态改名为独立删车结果。',native,NATIVE_CHANNELS,comparisons,'all4',
      '四组逐帧GT PNG hash一致；原生A基线复用既有完整视频。先比较同帧静态车形，再切动态与完整重建观察保留和回退。')
    page('independent_pair_compare.html','完整版与仅跨帧：独立49帧七通道',
      '这两组已在切无卡之前，从各自真实30K终点严格恢复模型、Adam、RNG、采样器和SGCM，完成零更新独立渲染。源checkpoint未变，静态固定相机跨时间RGB差为0。此页包含实际独立删车与alpha；仅供体和仅归属尚没有独立渲染结果。',independent,FINAL_CHANNELS,[['full,real_only','完整版 ↔ 仅跨帧']],'full,real_only',
      '删除通道实际关闭五身份动态集合，独立渲染静态＋环境；它不能消除已进入静态层的车形。')
    receipt={'status':'READY_CPU_NATIVE_FOUR_GROUPS_AND_TWO_INDEPENDENT_EXPORTS','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'videos':records,'pages':{'native30k_four_compare.html':native,'independent_pair_compare.html':independent},
      'new_videos':22,'reused_videos':12,'frames_each':49,'full_frame_wh':[480,320],'export_GPU_used':False,'training_updates':0,
      'independent_completed':['full','real_only'],'independent_pending':['prior_only','ownership_only'],'quality_accepted':False,'human_verdict':None,'browser_UI_tested':False}
    MEDIA.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'new_videos':22,'reused_videos':12}))

if __name__=='__main__':main()
