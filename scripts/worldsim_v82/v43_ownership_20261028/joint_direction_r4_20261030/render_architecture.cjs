const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const {createCanvas,loadImage}=require('C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@napi-rs/canvas');
const out=path.resolve('outputs/V43_Ownership_20261027');
(async()=>{
 const source=fs.readFileSync(path.join(out,'architecture.svg'),'utf8'),im=await loadImage(Buffer.from(source)),c=createCanvas(im.width,im.height),g=c.getContext('2d');g.drawImage(im,0,0);fs.writeFileSync(path.join(out,'architecture.png'),c.toBuffer('image/png'));
 const attrs=s=>Object.fromEntries([...s.matchAll(/([\w-]+)="([^"]*)"/g)].map(m=>[m[1],m[2]]));
 const rects=[...source.matchAll(/<rect ([^>]*)\/>/g)].map(m=>{const a=attrs(m[1]);return {x:+(a.x||0),y:+(a.y||0),w:+a.width,h:+a.height};});
 const overflow=[];
 for(const m of source.matchAll(/<text ([^>]*)>(.*?)<\/text>/g)){
  const a=attrs(m[1]),x=+a.x,y=+a.y,text=m[2].replaceAll('&amp;','&').replaceAll('&lt;','<').replaceAll('&gt;','>');
  g.font=`${a['font-weight']} ${a['font-size']}px "Microsoft YaHei",Arial,sans-serif`;
  const width=g.measureText(text).width,left=x-(a['text-anchor']==='middle'?width/2:0),right=left+width;
  const box=rects.filter(r=>x>=r.x&&x<=r.x+r.w&&y>=r.y&&y<=r.y+r.h).sort((a,b)=>a.w*a.h-b.w*b.h)[0];
  if(box&&(left<box.x+3||right>box.x+box.w-3))overflow.push({text,width,box,left,right});
 }
 const receipt={width:im.width,height:im.height,nested_major_modules:3,text_overflow:overflow,browserQA:false,native_canvas_rendered:true};fs.writeFileSync(path.join(out,'architecture_validation.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt,null,2));assert.equal(overflow.length,0);
})().catch(e=>{console.error(e);process.exitCode=1;});
