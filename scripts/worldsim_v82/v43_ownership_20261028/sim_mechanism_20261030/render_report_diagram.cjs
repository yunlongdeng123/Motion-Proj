// 由主代理在最终build_report之后执行；读取最终output SVG，重渲PNG并登记实际mtime。
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const {createCanvas,loadImage}=require('C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@napi-rs/canvas');
const defaults=path.resolve(__dirname,'../../outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030');
const args=process.argv.slice(2);
if(args.length&&!(args.length===2&&args[0]==='--output-dir'))throw new Error('Usage: node render_report_diagram.cjs [--output-dir PATH]');
const out=args.length?path.resolve(args[1]):defaults;
(async()=>{
 const svg=path.join(out,'architecture.svg'),png=path.join(out,'architecture.png');
 const source=fs.readFileSync(svg,'utf8');
 assert(!/GPU smoke\s*待跑|当前均等待实际凭据/.test(source),'最终SVG仍有旧骨架pending标签，请先修正SVG');
 const im=await loadImage(Buffer.from(source)),canvas=createCanvas(im.width,im.height),g=canvas.getContext('2d');
 g.drawImage(im,0,0);fs.writeFileSync(png,canvas.toBuffer('image/png'));
 const attrs=s=>Object.fromEntries([...s.matchAll(/([\w-]+)="([^"]*)"/g)].map(m=>[m[1],m[2]]));
 const rects=[...source.matchAll(/<rect ([^>]*)\/>/g)].map(m=>{const a=attrs(m[1]);return{x:+(a.x||0),y:+(a.y||0),w:+a.width,h:+a.height};});
 const overflow=[];
 for(const match of source.matchAll(/<text ([^>]*)>(.*?)<\/text>/g)){
  const a=attrs(match[1]),x=+a.x,y=+a.y,text=match[2].replaceAll('&amp;','&').replaceAll('&lt;','<').replaceAll('&gt;','>');
  g.font=`${a['font-weight']||'normal'} ${a['font-size']}px "Microsoft YaHei",Arial,sans-serif`;
  const width=g.measureText(text).width,left=x-((a['text-anchor']||'middle')==='middle'?width/2:0),right=left+width;
  const box=rects.filter(r=>x>=r.x&&x<=r.x+r.w&&y>=r.y&&y<=r.y+r.h).sort((a,b)=>a.w*a.h-b.w*b.h)[0];
  if(box&&(left<box.x+3||right>box.x+box.w-3))overflow.push({text,width,box,left,right});
 }
 const receipt={artifact_kind:'actual_report_diagram_render_only',width:im.width,height:im.height,
  svg_source:svg,svg_mtime_ns:fs.statSync(svg,{bigint:true}).mtimeNs.toString(),
  png_mtime_ns:fs.statSync(png,{bigint:true}).mtimeNs.toString(),
  nested_rectangles:rects.length,text_overflow:overflow,native_canvas_rendered:true,browserQA:false,experiment_validation:false};
 fs.writeFileSync(path.join(out,'architecture_validation.json'),JSON.stringify(receipt,null,2)+'\n');
 console.log(JSON.stringify(receipt));assert.equal(overflow.length,0,'最终组件图文字越界');
})().catch(error=>{console.error(error);process.exitCode=1;});
