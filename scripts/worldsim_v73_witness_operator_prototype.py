"""首面自定义反传与观测见证零空间更新原型；不接入正在运行的训练。"""
import json,time,resource
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
from scipy.sparse.linalg import LinearOperator, lsmr

class FirstDepth(torch.autograd.Function):
 @staticmethod
 def forward(ctx,vertices,faces,origins,directions):
  vv=vertices.detach().cpu().numpy();ff=faces.detach().cpu().numpy()
  s=o3d.t.geometry.RaycastingScene(nthreads=2)
  s.add_triangles(o3d.core.Tensor(vv.astype(np.float32)),o3d.core.Tensor(ff.astype(np.uint32)))
  ans=s.cast_rays(o3d.core.Tensor(torch.cat([origins,directions],-1).detach().cpu().numpy().astype(np.float32)),nthreads=2)
  ids=torch.as_tensor(ans['primitive_ids'].numpy().astype(np.int64),device=vertices.device);valid=ids<len(faces)
  vid=faces[ids[valid]];tri=vertices[vid];d=directions[valid];o=origins[valid]
  n=torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);den=(n*d).sum(-1)
  # BVH负责离散选面，双精度解算选中面的交点与重心坐标。
  mat=torch.stack([d,-(tri[:,1]-tri[:,0]),-(tri[:,2]-tri[:,0])],-1)
  tuv=torch.linalg.solve(mat,tri[:,0]-o)
  bary=torch.stack([1-tuv[:,1]-tuv[:,2],tuv[:,1],tuv[:,2]],-1)
  coeff=bary[:,:,None]*n[:,None,:]/den[:,None,None]
  out=vertices.new_full((len(origins),),float('inf'));out[valid]=tuv[:,0]
  ctx.save_for_backward(vid,coeff,valid);ctx.vshape=vertices.shape
  return out
 @staticmethod
 def backward(ctx,grad):
  vid,coeff,valid=ctx.saved_tensors;g=grad.new_zeros(ctx.vshape)
  g.index_add_(0,vid.reshape(-1),(coeff*grad[valid,None,None]).reshape(-1,3))
  return g,None,None,None

class WitnessApply(torch.autograd.Function):
 @staticmethod
 def forward(ctx,u,indices,coeff):
  ctx.save_for_backward(indices,coeff);ctx.shape=u.shape
  return (u[indices]*coeff).sum((1,2))
 @staticmethod
 def backward(ctx,g):
  ids,c=ctx.saved_tensors;du=g.new_zeros(ctx.shape)
  du.index_add_(0,ids.reshape(-1),(c*g[:,None,None]).reshape(-1,3))
  return du,None,None

def apply(u,ids,c):return WitnessApply.apply(u,ids,c)
def adjoint(z,ids,c,n):
 out=z.new_zeros(n,3);out.index_add_(0,ids.reshape(-1),(z[:,None,None]*c).reshape(-1,3));return out
def main():
 torch.set_num_threads(2);torch.manual_seed(7321);start=time.time()
 root=Path('/root/autodl-tmp/motion_proj/docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1')
 data=np.load(root/'cases/First-r6.npz');v=torch.tensor(data['vertices'],dtype=torch.float64);f=torch.tensor(data['faces'],dtype=torch.long)
 o=torch.tensor(data['origins'],dtype=torch.float64);d=torch.tensor(data['directions'],dtype=torch.float64);r=torch.tensor(data['ranges'],dtype=torch.float64)
 ids0=data['face_ids'];valid=(ids0<len(f))&np.isfinite(data['first'])
 bary0=np.c_[1-data['uv'].sum(-1),data['uv']]
 # 此处使用DEV保存几何做算子校验，不把局部试步作为可训练方法的泛化收益。
 take=np.where(valid&(bary0.min(-1)>.03))[0][:64]
 ids=f[torch.tensor(ids0[take])];tri=v[ids];normal=torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal=normal/normal.norm(dim=-1,keepdim=True)
 mat=torch.stack([d[take],-(tri[:,1]-tri[:,0]),-(tri[:,2]-tri[:,0])],-1);tuv=torch.linalg.solve(mat,tri[:,0]-o[take]);b=torch.stack([1-tuv[:,1]-tuv[:,2],tuv[:,1],tuv[:,2]],-1)
 c=b[:,:,None]*normal[:,None,:]
 g=torch.randn_like(v);z=torch.randn(len(take),dtype=v.dtype)
 adjerr=abs((apply(g,ids,c)@z-(g*adjoint(z,ids,c,len(v))).sum()).item())
 op=LinearOperator((v.numel(),len(take)),matvec=lambda z:adjoint(torch.from_numpy(z),ids,c,len(v)).numpy().reshape(-1),rmatvec=lambda u:apply(torch.from_numpy(u.reshape(-1,3)),ids,c).numpy(),dtype=np.float64)
 # LSMR直接作用于稀疏B及其伴随，避免法方程平方条件数与递归残差漂移。
 solved=lsmr(op,g.numpy().reshape(-1),atol=1e-13,btol=1e-13,conlim=1e14,maxiter=6000)
 lam=torch.from_numpy(solved[0]);it=int(solved[2]);res=float(solved[4]);p=g-adjoint(lam,ids,c,len(v));nullres=apply(p,ids,c).norm().item()
 q=torch.randn_like(v,requires_grad=True);loss=apply(q,ids,c).square().sum();autograd=torch.autograd.grad(loss,q)[0]
 expected=adjoint(2*apply(q.detach(),ids,c),ids,c,len(v));backerr=(autograd-expected).abs().max().item()
 # 固定首面内部的差分校验包含一般三角、共享顶点与实际射线。
 vv=v.clone().requires_grad_();t=FirstDepth.apply(vv,f,o[take],d[take]);w=torch.randn_like(t);gradient=torch.autograd.grad((t*w).sum(),vv)[0]
 direction=torch.randn_like(v);direction/=direction.norm();eps=1e-5
 fd=((FirstDepth.apply(v+eps*direction,f,o[take],d[take])-FirstDepth.apply(v-eps*direction,f,o[take],d[take]))*w).sum()/(2*eps)
 pred=(gradient*direction).sum();relative=abs((fd-pred).item())/max(1,abs(pred.item()))
 p=p/p.norm();g=g/g.norm();base=FirstDepth.apply(v,f,o[take],d[take]);scales=[]
 for step in [.001,.0005,.00025]:
  tp=FirstDepth.apply(v+step*p,f,o[take],d[take]);tg=FirstDepth.apply(v+step*g,f,o[take],d[take])
  scales.append({'step_norm_m':step,'projected_max_depth_change_m':(tp-base).abs().max().item(),'raw_max_depth_change_m':(tg-base).abs().max().item()})
 # 保存张量由autograd实际记录，不把Python/CPU内存等同CUDA allocated。
 saved=[]
 def pack(x):saved.append(x.numel()*x.element_size());return x
 with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):
  testv=v.clone().requires_grad_();tt=FirstDepth.apply(testv,f,o[take],d[take]);tt.sum().backward()
 result={'status':'done','scope':'CPU float64 operator verification on one saved DEV surface; not learned efficacy','witnesses':len(take),'vertices':len(v),'faces':len(f),'adjoint_absolute_error':adjerr,'witness_backward_max_error':backerr,'first_depth_directional_relative_error':relative,'projected_constraint_residual':nullres,'lsmr_iterations':it,'lsmr_normal_residual':res,'step_response':scales,'saved_autograd_bytes':sum(saved),'dense_ray_face_entries_avoided':len(take)*len(f),'wall_s':time.time()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'boundary':'BVH float32 active set, float64 selected intersection; geometry backward only; no derivatives for rays/face choice; fixed local witnesses; projection coefficients detached; silhouettes/ties/grazing not made differentiable; no training or external20'}
 result['relative_constraint_residual']=nullres/max(1,apply(g,ids,c).norm().item())
 assert relative<1e-5 and adjerr<1e-9 and backerr<1e-10 and result['relative_constraint_residual']<1e-6,result
 (root/'operator_check.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
