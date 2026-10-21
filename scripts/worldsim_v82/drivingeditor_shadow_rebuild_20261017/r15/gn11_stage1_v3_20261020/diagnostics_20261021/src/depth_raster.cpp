// 只读CPU光栅参考：与已检查的native tile/conic/alpha/early-stop约定一致。
#include <vector>
#include <algorithm>
#include <cmath>
extern "C" int project_depth(int n,const float* xyz,const float* scales,const float* quat,const float* opacity,const float* time,const float* sigma_t,const float* vel,const long long* road,float timestamp,float cycle,const float* W,const float* K,int neg,float* uv,float* conic,float* op,float* depth,int* lo,int* hi,unsigned char* isroad){
 int count=0;float omega=float(2*3.14159265358979323846)/cycle;
 double tx=neg?std::tan(-.5):480./(2*K[0]),ty=neg?std::tan(-.5):320./(2*K[4]);float fx=float(480./(2*tx)),fy=float(320./(2*ty));
 for(int i=0;i<n;i++){
  bool rd=road[i]>=0;float delta=timestamp-time[i],m=rd?1.f:std::exp(-.5f*(delta/sigma_t[i])*(delta/sigma_t[i]));if(m<=.05f)continue;
  float move=rd?0.f:std::sin(delta*omega)/omega,p[3];for(int k=0;k<3;k++)p[k]=xyz[3*i+k]+vel[3*i+k]*move;
  float c[3];for(int k=0;k<3;k++)c[k]=W[k*4]*p[0]+W[k*4+1]*p[1]+W[k*4+2]*p[2]+W[k*4+3];if(c[2]<=.2f||!std::isfinite(c[2]))continue;
  float norm=std::max(1e-12f,std::sqrt(quat[4*i]*quat[4*i]+quat[4*i+1]*quat[4*i+1]+quat[4*i+2]*quat[4*i+2]+quat[4*i+3]*quat[4*i+3]));
  float w=quat[4*i]/norm,x=quat[4*i+1]/norm,y=quat[4*i+2]/norm,z=quat[4*i+3]/norm;
  float R[9]={1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)};
  float xz=float(std::min(1.3*tx,std::max(-1.3*tx,double(c[0]/c[2])))),yz=float(std::min(1.3*ty,std::max(-1.3*ty,double(c[1]/c[2]))));
  float J[6]={fx/c[2],0,-fx*xz/c[2],0,fy/c[2],-fy*yz/c[2]},T[6],B[6];
  for(int row=0;row<2;row++)for(int col=0;col<3;col++)T[row*3+col]=J[row*3]*W[col]+J[row*3+1]*W[4+col]+J[row*3+2]*W[8+col];
  for(int row=0;row<2;row++)for(int col=0;col<3;col++)B[row*3+col]=(T[row*3]*R[col]+T[row*3+1]*R[3+col]+T[row*3+2]*R[6+col])*scales[3*i+col];
  float a=.3f+B[0]*B[0]+B[1]*B[1]+B[2]*B[2],b=B[0]*B[3]+B[1]*B[4]+B[2]*B[5],d=.3f+B[3]*B[3]+B[4]*B[4]+B[5]*B[5],det=a*d-b*b;if(!(det>0))continue;
  float u=K[0]*c[0]/c[2]+K[2]-.5f,v=K[4]*c[1]/c[2]+K[5]-.5f,mid=.5f*(a+d),radius=std::ceil(3*std::sqrt(std::max(0.f,mid+std::sqrt(std::max(.1f,mid*mid-det)))));
  if(!std::isfinite(radius)||!std::isfinite(u)||!std::isfinite(v))continue;
  auto bound=[](double value,int maximum){return int(std::min(double(maximum),std::max(0.,std::trunc(value))));};
  int lx=bound((u-radius)/16,30),ly=bound((v-radius)/16,20),hx=bound((u+radius+15)/16,30),hy=bound((v+radius+15)/16,20);if(hx<=lx||hy<=ly)continue;
  uv[count*2]=u;uv[count*2+1]=v;conic[count*3]=d/det;conic[count*3+1]=-b/det;conic[count*3+2]=a/det;op[count]=opacity[i]*m;depth[count]=c[2];lo[count*2]=lx;lo[count*2+1]=ly;hi[count*2]=hx;hi[count*2+1]=hy;isroad[count]=rd;count++;
 }return count;
}
extern "C" int raster_depth(int n,const float* uv,const float* conic,const float* opacity,const float* z,const int* lo,const int* hi,float* alpha,float* depth){
 // 全局中心z稳定顺序与逐tile顺序等价。只访问alpha可达到1/255的椭圆包围盒。
 std::vector<int> ids;for(int i=0;i<n;i++)if(opacity[i]>=1.f/255.f)ids.push_back(i);
 std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return z[a]<z[b];});
 std::vector<float> trans(480*320,1.f);std::vector<unsigned char> done(480*320,0);
 for(int i:ids){
  double a=conic[3*i],b=conic[3*i+1],c=conic[3*i+2],det=a*c-b*b;if(det<=0)continue;
  double r=std::max(0.,2*std::log(double(opacity[i])*255.)),rx=std::sqrt(r*c/det)+1,ry=std::sqrt(r*a/det)+1;
  int xmin=std::max(lo[2*i]*16,int(std::floor(uv[2*i]-rx))),xmax=std::min(hi[2*i]*16-1,int(std::ceil(uv[2*i]+rx)));
  int ymin=std::max(lo[2*i+1]*16,int(std::floor(uv[2*i+1]-ry))),ymax=std::min(hi[2*i+1]*16-1,int(std::ceil(uv[2*i+1]+ry)));
  for(int y=ymin;y<=ymax;y++)for(int x=xmin;x<=xmax;x++){
   int pixel=y*480+x;if(done[pixel])continue;float T=trans[pixel];float dx=uv[2*i]-x,dy=uv[2*i+1]-y;float power=-.5f*(conic[3*i]*dx*dx+conic[3*i+2]*dy*dy)-conic[3*i+1]*dx*dy;
    if(power>0)continue;float a=std::min(.99f,opacity[i]*std::exp(power));if(a<1.f/255.f)continue;
    float test=T*(1.f-a);if(test<.0001f){done[pixel]=1;continue;}float w=T*a;alpha[pixel]+=w;depth[pixel]+=w*z[i];trans[pixel]=test;
  }
 }return 0;
}
