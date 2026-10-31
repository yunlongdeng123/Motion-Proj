"""Read Waymo packed MatrixFloat without materializing a Python float per pixel.
Strict wire parser; numeric arrays are identical to the protobuf reference.
"""
import zlib
import numpy as np

def varint(b,i):
    v=0;s=0
    while True:
        t=b[i];i+=1;v|=(t&127)<<s
        if t<128:return v,i
        s+=7
        if s>63:raise ValueError('invalid protobuf varint')

def fields(b):
    i=0
    while i<len(b):
        tag,i=varint(b,i);wire=tag&7
        if wire==2:
            n,i=varint(b,i);v=b[i:i+n];i+=n
        elif wire==0:v,i=varint(b,i)
        elif wire==5:v=b[i:i+4];i+=4
        else:raise ValueError(f'unexpected wire type {wire}')
        if i>len(b):raise ValueError('truncated protobuf')
        yield tag>>3,wire,v

def matrix(compressed,dataset_pb2=None):
    chunks=[];dims=[]
    for field,wire,value in fields(zlib.decompress(compressed)):
        if field==1 and wire in (2,5):chunks.append(np.frombuffer(value,dtype='<f4'))
        elif field==2 and wire==2:
            for f,w,v in fields(value):
                if f!=1:raise ValueError('unexpected MatrixShape field')
                if w==0:dims.append(v)
                elif w==2:
                    i=0
                    while i<len(v):d,i=varint(v,i);dims.append(d)
                else:raise ValueError('unexpected dims wire')
        else:raise ValueError('unexpected MatrixFloat field')
    a=(chunks[0] if len(chunks)==1 else np.concatenate(chunks)).reshape(dims)
    if not np.isfinite(a).all():raise ValueError('nonfinite range image')
    return a

if __name__=='__main__':
    import sys,time,json
    sys.path.insert(0,'/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
    import waymo_numpy_cpu as ref
    pb=ref.protobuf();raw='/root/autodl-tmp/datasets/waymo_v1_4_3/individual_files/training/segment-10094743350625019937_3420_000_3440_000_with_camera_labels.tfrecord'
    _,payload,_=next(ref.records(raw,0));fr=pb.Frame.FromString(payload)
    values=[fr.lasers[0].ri_return1.range_image_pose_compressed]
    values += [getattr(l,f'ri_return{k}').range_image_compressed for l in fr.lasers for k in (1,2)]
    t=time.time();count=0
    for v in values:
        if not v:continue
        a=ref.matrix(v,pb);b=matrix(v,pb);assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b);count+=1
    print(json.dumps(dict(checked_matrices=count,all_arrays_exact=True,elapsed_seconds=time.time()-t)),flush=True)
