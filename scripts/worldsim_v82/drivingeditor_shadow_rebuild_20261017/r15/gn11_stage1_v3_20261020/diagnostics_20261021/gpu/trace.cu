// Read-only decoder for inspected DrivingEditor buffer ABI. Does not replace
// native preprocessing, tile duplication, sorting, or full image rendering.
// Blending expressions match the Graphdeco research-licensed forward kernel.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

template<class T> void take(char*& c,T*& p,size_t n) {
  auto x=(reinterpret_cast<uintptr_t>(c)+127)&~uintptr_t(127);
  p=reinterpret_cast<T*>(x);c=reinterpret_cast<char*>(p+n);
}
__global__ void run_trace(const int* pixels,int Q,int W,int H,int cap,
 const float* depths,const int* radii,const float2* uv,const float4* conic,
 const float* colors,const float* features,int S,const uint32_t* list,
 const uint2* ranges,const float* nativeT, int64_t* ids,float* records,
 int64_t* counts,float* sums) {
 int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=Q)return;
 int x=pixels[2*q],y=pixels[2*q+1],tile=(y/16)*((W+15)/16)+x/16;
 uint2 range=ranges[tile];float T=1,C[3]={0,0,0},aSum=0,zSum=0;
 int count=0,last=0,visited=0;
 for(uint32_t j=range.x;j<range.y;++j){
   visited++;int id=list[j];float2 u=uv[id];float dx=u.x-x,dy=u.y-y;float4 co=conic[id];
   float power=-0.5f*(co.x*dx*dx+co.z*dy*dy)-co.y*dx*dy;
   if(power>0.0f)continue;float alpha=min(0.99f,co.w*exp(power));
   if(alpha<1.0f/255.0f)continue;float test_T=T*(1-alpha);
   if(test_T<0.0001f)break;
   float w=alpha*T;
   if(count<cap){
     ids[q*cap+count]=id;float* r=records+(q*cap+count)*13;
     r[0]=depths[id];r[1]=alpha;r[2]=T;r[3]=w;
     for(int k=0;k<3;k++)r[4+k]=colors[id*3+k];
     r[7]=u.x;r[8]=u.y;r[9]=co.x;r[10]=co.y;r[11]=co.z;r[12]=radii[id];
   }
   for(int k=0;k<3;k++)C[k]+=colors[id*3+k]*w;
   zSum+=features[id*S+S-2]*w;aSum+=features[id*S+S-1]*w;
   T=test_T;count++;last=visited;
 }
 counts[q*4]=range.y-range.x;counts[q*4+1]=count;counts[q*4+2]=last;counts[q*4+3]=visited;
 float* v=sums+q*7;for(int k=0;k<3;k++)v[k]=C[k];v[3]=zSum;v[4]=aSum;v[5]=T;v[6]=nativeT[y*W+x];
}
std::vector<torch::Tensor> trace_cuda(torch::Tensor geom,torch::Tensor bin,
 torch::Tensor image,torch::Tensor precolor,torch::Tensor features,torch::Tensor pixels,torch::Tensor nativeRadii,
 int64_t P,int64_t W,int64_t H,int64_t cap){
 TORCH_CHECK(P>0&&W==480&&H==320&&cap>0&&cap<=131072,"fixed dimensions/capacity required");
 for(auto t:{geom,bin,image,features,pixels})TORCH_CHECK(t.is_cuda()&&t.is_contiguous(),"contiguous CUDA buffers required");
 TORCH_CHECK(geom.scalar_type()==torch::kUInt8&&bin.scalar_type()==torch::kUInt8&&image.scalar_type()==torch::kUInt8,"opaque byte buffers required");
 TORCH_CHECK(pixels.scalar_type()==torch::kInt32&&pixels.dim()==2&&pixels.size(1)==2,"pixels must be Qx2 int32");
 TORCH_CHECK(features.scalar_type()==torch::kFloat32&&features.dim()==2&&features.size(0)==P&&features.size(1)>=2,"native depth/alpha features required");
 char* g=(char*)geom.data_ptr();float *depth,*cov,*rgb;bool* clamped;int* rad;float2* uv;float4* co;
 take(g,depth,P);take(g,clamped,P*3);take(g,rad,P);take(g,uv,P);take(g,cov,P*6);take(g,co,P);take(g,rgb,P*3);
 TORCH_CHECK(g<=(char*)geom.data_ptr()+geom.numel(),"unknown geometry ABI");
 char* b=(char*)bin.data_ptr();uint32_t* list;take(b,list,0);
 char* im=(char*)image.data_ptr();float* T;uint32_t* contrib;uint2* ranges;
 take(im,T,W*H);take(im,contrib,W*H);take(im,ranges,W*H);
 TORCH_CHECK(im<=(char*)image.data_ptr()+image.numel(),"unknown image ABI");
 int Q=pixels.size(0);TORCH_CHECK(Q>0&&Q<=64,"1..64 pixels only");
 auto ids=torch::full({Q,cap},-1,geom.options().dtype(torch::kInt64));
 auto rec=torch::zeros({Q,cap,13},features.options());
 auto counts=torch::zeros({Q,4},ids.options());auto sums=torch::zeros({Q,7},features.options());
 const float* c=precolor.numel()?precolor.data_ptr<float>():rgb;
 TORCH_CHECK(nativeRadii.is_cuda()&&nativeRadii.is_contiguous()&&nativeRadii.scalar_type()==torch::kInt32&&nativeRadii.numel()==P,"native output radii required");
 run_trace<<<(Q+63)/64,64,0,at::cuda::getCurrentCUDAStream()>>>(pixels.data_ptr<int>(),Q,W,H,cap,depth,nativeRadii.data_ptr<int>(),uv,co,c,features.data_ptr<float>(),features.size(1),list,ranges,T,ids.data_ptr<int64_t>(),rec.data_ptr<float>(),counts.data_ptr<int64_t>(),sums.data_ptr<float>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {ids,rec,counts,sums};
}
