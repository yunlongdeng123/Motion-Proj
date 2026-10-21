(function(root){
 'use strict';
 const math={
  ray(cam,x,y){const v=[(x+.5-cam.K[0][2])/cam.K[0][0],(y+.5-cam.K[1][2])/cam.K[1][1],1];return [0,1,2].map(i=>cam.W[0][i]*v[0]+cam.W[1][i]*v[1]+cam.W[2][i]*v[2]);},
  depthPoint(cam,x,y,z,scale){const d=math.ray(cam,x,y);return cam.center.map((v,i)=>v+d[i]*z*scale);},
  panFocus(v,dx,dy,width){const cy=Math.cos(v.yaw),sy=Math.sin(v.yaw),cp=Math.cos(v.pitch),sp=Math.sin(v.pitch),right=[cy,-sy,0],down=[cp*sy,cp*cy,-sp],k=v.distance/(width*.8);return v.focus.map((x,i)=>x-k*(right[i]*dx+down[i]*dy));},
  within(p,r){return p&&p[0]>=Math.min(r[0],r[2])&&p[0]<=Math.max(r[0],r[2])&&p[1]>=Math.min(r[1],r[3])&&p[1]<=Math.max(r[1],r[3]);}
 };root.GSInspectionMath=math;if(typeof module!=='undefined')module.exports=math;
})(typeof window==='undefined'?globalThis:window);
