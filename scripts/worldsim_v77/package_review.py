"""生成可离线审核的全部对象页面和固定分组接触表，不代填人工评分。"""
import argparse,html,json,pathlib,shutil
from PIL import Image,ImageDraw

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
 run=pathlib.Path(a.run_dir);ev=run/'evaluation_v2';out=pathlib.Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 summary=json.loads((ev/'summary.json').read_text());reg=json.loads((run/'registration.json').read_text());body=[];all_rows=[]
 for scene in reg['scenes']:
  name=scene['name'];dst=out/name;dst.mkdir(exist_ok=True)
  shutil.copy2(ev/name/'scene_comparison.jpg',dst/'scene_comparison.jpg')
  data=json.loads((ev/name/'evaluation.json').read_text());all_rows.extend(data['actors'])
  body.append(f'<section><h2>{name}</h2><p>六个观察相机：原始 RGB / 预测相机 + Sim3 / GT 标定 + 背景单尺度。深灰为未覆盖或范围外；不补洞。</p><img loading="lazy" src="{name}/scene_comparison.jpg"></section>')
  for variant in ['native','calibrated_control']:
   rows=[r for r in data['actors'] if r['variant']==variant];tiles=[]
   for row in rows:
    actor=row['actor_id'];relative=f'{name}/{variant}/actor_{actor}';d=out/relative;d.mkdir(parents=True,exist_ok=True)
    src=ev/relative
    for f in ['edit_crop.jpg','canonical.jpg','metrics.json']:shutil.copy2(src/f,d/f)
    recall=row['visible_lidar_recall_0p2m'];recall='缺参考' if recall is None else f'{recall:.1%}'
    title=f"{name} / {variant} / actor {actor} / {row['category']} / points {row['point_count']} / LiDAR recall {recall}"
    crop=Image.open(d/'edit_crop.jpg');canon=Image.open(d/'canonical.jpg');canon=canon.resize((1250,277))
    tile=Image.new('RGB',(1250,45+crop.height+277),'#edf2f8');ImageDraw.Draw(tile).text((8,13),title,fill='#13273b');tile.paste(crop,(0,45));tile.paste(canon,(0,45+crop.height));tiles.append(tile)
    body.append(f'<article data-variant="{variant}"><h3>{html.escape(title)}</h3><p>横排：原图、factual、MOVE（世界x+3m、yaw+15°）、DELETE、INSERT（clone至x+3m）。下排：actor-local前、后、左、右、顶视图。几何未做补全。</p><img loading="lazy" src="{relative}/edit_crop.jpg"><img loading="lazy" src="{relative}/canonical.jpg"><a href="{relative}/metrics.json">逐命令与对象指标</a></article>')
   for start in range(0,len(tiles),4):
    group=tiles[start:start+4];sheet=Image.new('RGB',(1250,sum(t.height for t in group)+8*(len(group)-1)),'#d5deea');y=0
    for tile in group:sheet.paste(tile,(0,y));y+=tile.height+8
    sheet.save(out/f'{name}_{variant}_review_{start//4+1}.jpg',quality=91)
 metrics=''.join(f'<tr><td>{k}</td><td>{v["empty_actors"]}/24</td><td>{v["macro_visible_lidar_recall_0p2m"]:.1%}</td><td>{v["valid_object_commands"]}/{v["commands"]}</td><td>{v["max_background_change_m"]}</td></tr>' for k,v in summary['by_variant'].items())
 intro=f'<header><p>WORLD SIMULATION · V7.7 · P0</p><h1>冻结 VGGT-Ω：24 个对象的结构化编辑</h1><p>三场景、同一 processed 时刻、每场景六相机；零训练。开发集结果，不是高保真编辑已通过的声明。</p></header><section><h2>数值结果与阅读边界</h2><table><tr><th>读出</th><th>空对象</th><th>可见 LiDAR 20cm 召回（宏平均）</th><th>非空对象操作</th><th>背景点变化/m</th></tr>{metrics}</table><p>calibrated_control额外使用GT相机内外参与同帧背景LiDAR尺度；不归为纯RGB结果。LiDAR指标仅反映观察到的表面，不等于完整性或纯净度。空对象上的算子检查不计有效编辑。</p><p>MOVE / DELETE / INSERT数值通过，只说明解析操作按指令执行。未见面、背景暴露缺口及GT框内混入物仍需检查。人工判定保留为空。</p></section>'
 architecture='<section><h2>Architecture components</h2><div class="architecture"><span>多视角 RGB</span> → <span>冻结 VGGT-Ω</span> → <span>深度 / 相机</span> → <span>米制对齐</span> → <span>GT 框选择</span> → <span>解析编辑</span> → <span>渲染 / 对象审核</span></div></section>'
 css='body{margin:0;background:#eef2f6;color:#172a40;font:16px/1.65 system-ui,sans-serif}main{max-width:1300px;margin:auto;padding:36px 22px}header{padding:28px 0}header p:first-child{letter-spacing:2px;color:#187e81;font-size:13px}h1{font-size:34px}h2{font-size:23px}h3{font-size:17px}section,article{background:white;padding:24px;border-radius:12px;margin:22px 0;box-shadow:0 2px 6px #19314b0a}img{display:block;max-width:100%;height:auto;margin:14px 0}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:12px;border-bottom:1px solid #dfe6ee}th{background:#f3f6fa}a{color:#086e8e}.architecture{display:flex;flex-wrap:wrap;align-items:center;gap:10px}.architecture span{padding:10px;background:#e8f2f4;border:1px solid #b3d7dc;border-radius:6px}nav{position:sticky;top:0;background:#edf2f5ef;padding:12px;z-index:1}button{padding:9px 16px;margin-right:8px;cursor:pointer;border:1px solid #abc2d2;border-radius:6px;background:white}'
 nav='<nav><button onclick="filter(\'all\')">全部对象</button><button onclick="filter(\'native\')">预测相机 + Sim3</button><button onclick="filter(\'calibrated_control\')">GT 标定辅助</button></nav>'
 script='<script>function filter(v){document.querySelectorAll("article[data-variant]").forEach(x=>x.hidden=v!=="all"&&x.dataset.variant!==v)}</script>'
 (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>V7.7 P0 对象编辑审核</title><style>'+css+'</style><main>'+intro+architecture+nav+''.join(body)+'</main>'+script+'</html>')
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 (out/'all_actor_metrics.json').write_text(json.dumps(all_rows,ensure_ascii=False,indent=2)+'\n')
 print(str(out/'index.html'))

if __name__=='__main__':main()
