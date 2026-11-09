#include <cuda_runtime.h>
// A fixed row/block reduction, avoiding atomic-add-dependent CSR SpMM order.
__global__ void mm3(int rows,const int* offsets,const int* ids,const float* v,const float* x,float* out){
    int row=blockIdx.x;int lane=threadIdx.x;if(row>=rows)return;
    __shared__ double scratch[128][3];double a=0,b=0,c=0;
    for(int p=offsets[row]+lane;p<offsets[row+1];p+=128){
        int id=3*ids[p];double w=(double)v[p];
        a+=w*(double)x[id];b+=w*(double)x[id+1];c+=w*(double)x[id+2];
    }
    scratch[lane][0]=a;scratch[lane][1]=b;scratch[lane][2]=c;__syncthreads();
    for(int step=64;step;step>>=1){
        if(lane<step){scratch[lane][0]+=scratch[lane+step][0];scratch[lane][1]+=scratch[lane+step][1];scratch[lane][2]+=scratch[lane+step][2];}
        __syncthreads();
    }
    if(lane==0){out[3*row]=(float)scratch[0][0];out[3*row+1]=(float)scratch[0][1];out[3*row+2]=(float)scratch[0][2];}
}
extern "C" int csr_mm3(int rows,const int* offsets,const int* ids,const float* v,const float* x,float* out,void* stream){
    mm3<<<rows,128,0,(cudaStream_t)stream>>>(rows,offsets,ids,v,x,out);return (int)cudaGetLastError();
}
