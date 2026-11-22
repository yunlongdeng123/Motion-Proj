"""Availability/count index only: original199 records, no XYZ/cloud conversion or pool choice."""
import argparse,csv,hashlib,importlib.util,json,struct,time,zlib
from pathlib import Path
import numpy as np
from diagnose_nonplanar_inference_cpu import read,write,pin

TRAIN=[60,70,80,95,110,125,145,165]
HELDOUT=[134,179,198]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def counts(a):return {str(int(k)):int(n) for k,n in zip(*np.unique(a,return_counts=True))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    C,out=a.campaign_root,a.output;out.mkdir(parents=True,exist_ok=False);start=time.time();R=C/'road_underbody_local_20261123_r10'
    bank=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs');rp=bank/'receipt.json';receipt=read(rp)
    dr=Path('/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101');dp=dr/'waymo_numpy_cpu.py';mp=dr/'fast_range_matrix.py'
    assert pin(dp)['sha256']==receipt['decoder_sha256'] and pin(mp)['sha256']==receipt['matrix_decoder_sha256']
    dec=module('TOP_availability_decoder',dp);dec.matrix=module('TOP_availability_fast_matrix',mp).matrix;pb=dec.protobuf();crc=dec.crc_function()
    enum_path=R/'audit/official_ground_geometry_v1/point_evidence_contract.json';enum=read(enum_path)['official_label_enum'];road,lane=enum['TYPE_ROAD'],enum['TYPE_LANE_MARKER']
    prior=read(R/'audit/anomalous_raw_point_chain_v1/point_chain_summary.json');previous={x['raw']:x['original_record'] for x in prior['rows']}
    frame_map={x['raw']:x for x in receipt['frames']};assert sorted(frame_map)==list(range(199))
    rawpath=Path(receipt['raw']);assert rawpath.stat().st_size==receipt['raw_bytes']
    contract=dict(schema='R10.TOP_semantic_availability_contract.v1',complete=True,
        scope='All199 original TOP returns; availability, label/valid-range counts and exact source record SHA only',
        raw_input=dict(path=str(rawpath),bytes=receipt['raw_bytes'],expected_sha256=receipt['raw_sha256']),
        frames=list(range(199)),current_training_pool_unchanged=TRAIN,current_geometry_heldout_prohibited_for_fit=HELDOUT,
        requested_reference_time_window_inclusive=[86,198],selection='No reference pool selected; no ranking by XYZ/Z/height/colour/quality',
        operations='Parse frame protobuf; decode semantic integer matrices and range positive/NLZ masks only when semantic data present; no lidar(), XYZ, projection, RGB or model inference',
        official_ground_enum=dict(road=road,lane_marker=lane),official_enum_provenance=pin(enum_path),
        no_pointcloud_conversion=True,no_Z_or_residual_filter=True,no_pool_or_split_mutation=True,GPU_forward=0,backward=0,updates=0)
    write(out/'availability_contract.json',contract)
    rows=[];whole=hashlib.sha256();offsets=[]
    with rawpath.open('rb') as f:
        for raw in range(199):
            offset=f.tell();header=f.read(12);assert len(header)==12;length,check=struct.unpack('<QI',header)
            assert 0<length<96*1024**2 and crc(header[:8])==check
            payload=f.read(length);trailer=f.read(4);assert len(payload)==length and len(trailer)==4 and crc(payload)==struct.unpack('<I',trailer)[0]
            whole.update(header);whole.update(payload);whole.update(trailer)
            digest=hashlib.sha256(payload).hexdigest();frame=pb.Frame.FromString(payload)
            assert frame.context.name==receipt['context_id'] and frame.timestamp_micros==frame_map[raw]['timestamp_micros']
            record=dict(raw=raw,header_offset=offset,payload_length=length,payload_sha256=digest,header_CRC32C=True,payload_CRC32C=True)
            if raw in previous:
                assert previous[raw]['header_offset']==offset and previous[raw]['payload_length']==length and previous[raw]['payload_sha256']==digest
            top=next(x for x in frame.lasers if x.name==1);returns=[]
            for ret in [0,1]:
                item=getattr(top,f'ri_return{ret+1}');compressed=item.segmentation_label_compressed
                rr=dict(return_index=ret,semantic_available=bool(compressed),semantic_compressed_bytes=len(compressed))
                if compressed:
                    mat=pb.MatrixInt32();mat.ParseFromString(zlib.decompress(compressed));labels=np.asarray(mat.data,np.int32).reshape(tuple(mat.shape.dims));assert labels.ndim==3 and labels.shape[-1]==2
                    image=dec.matrix(item.range_image_compressed,pb);assert image.shape[:2]==labels.shape[:2]
                    valid=(image[...,0]>0)&(image[...,3]!=1);semantic=labels[...,1]
                    rr.update(range_image_HW=list(labels.shape[:2]),label_matrix_bytes=labels.nbytes,
                        label_matrix_sha256=hashlib.sha256(labels.astype('<i4',copy=False).tobytes()).hexdigest(),
                        all_range_pixel_semantic_class_counts=counts(semantic),
                        valid_range_nonNLZ_semantic_class_counts=counts(semantic[valid]),
                        all_range_road_pixels=int((semantic==road).sum()),all_range_lane_pixels=int((semantic==lane).sum()),
                        valid_range_nonNLZ_road_points=int((valid&(semantic==road)).sum()),
                        valid_range_nonNLZ_lane_points=int((valid&(semantic==lane)).sum()),
                        valid_range_nonNLZ_total=int(valid.sum()),no_world_coordinates_computed=True)
                returns.append(rr)
            row=dict(raw=raw,timestamp_micros=int(frame.timestamp_micros),record=record,TOP_returns=returns,
                any_TOP_segmentation_available=any(x['semantic_available'] for x in returns),
                valid_range_nonNLZ_official_ground_count=sum(x.get('valid_range_nonNLZ_road_points',0)+x.get('valid_range_nonNLZ_lane_points',0) for x in returns),
                in_evaluation_time_window=86<=raw<=198,current_geometry_heldout_do_not_fit=raw in HELDOUT,
                current_registered_training_frame=raw in TRAIN,availability_only_not_qualified_reference=True)
            rows.append(row);offsets.append(record)
            if raw%25==0 or raw==198:print(json.dumps(dict(stage='availability_progress',raw=raw,available_so_far=sum(x['any_TOP_segmentation_available'] for x in rows))),flush=True)
        assert f.read(1)==b''
    rawsha=whole.hexdigest();assert rawsha==receipt['raw_sha256']
    available=[x['raw'] for x in rows if x['any_TOP_segmentation_available']]
    window=[x['raw'] for x in rows if x['any_TOP_segmentation_available'] and x['in_evaluation_time_window']]
    extra=[r for r in window if r not in TRAIN and r not in HELDOUT]
    with (out/'source_record_index.jsonl').open('w',encoding='utf-8') as f:
        for r in offsets:f.write(json.dumps(r,separators=(',',':'))+'\n')
    with (out/'availability_199.csv').open('w',encoding='utf-8',newline='') as f:
        cols=['raw','timestamp_micros','TOP_return0_available','TOP_return1_available','valid_official_road_count','valid_official_lane_count','in_time_window86_198','current_training','current_heldout_do_not_fit','record_offset','record_payload_sha256']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for x in rows:
            w.writerow(dict(raw=x['raw'],timestamp_micros=x['timestamp_micros'],TOP_return0_available=x['TOP_returns'][0]['semantic_available'],TOP_return1_available=x['TOP_returns'][1]['semantic_available'],
                valid_official_road_count=sum(r.get('valid_range_nonNLZ_road_points',0) for r in x['TOP_returns']),valid_official_lane_count=sum(r.get('valid_range_nonNLZ_lane_points',0) for r in x['TOP_returns']),
                in_time_window86_198=x['in_evaluation_time_window'],current_training=x['current_registered_training_frame'],current_heldout_do_not_fit=x['current_geometry_heldout_do_not_fit'],record_offset=x['record']['header_offset'],record_payload_sha256=x['record']['payload_sha256']))
    result=dict(schema='R10.original199_TOP_segmentation_availability.v1',complete=True,decision='AVAILABILITY_INDEX_COMPLETE_NO_POOL_SELECTED',
        source_dataset_receipt=pin(rp),raw_source=dict(path=str(rawpath),bytes=rawpath.stat().st_size,sha256=rawsha,whole_file_SHA_recomputed=True),
        records=199,all_record_CRC_and_payload_SHA_verified=True,existing5_record_index_exact=True,
        available_raw_indices=available,available_count=len(available),time_window_inclusive=[86,198],
        available_in_time_window=window,available_in_time_window_count=len(window),
        extra_available_in_time_window_excluding_registered_train_and_heldout_UNSELECTED=extra,
        current_registered_training_pool_unchanged=TRAIN,current_geometry_heldout_prohibited_for_fit=HELDOUT,
        heldout_availability=[dict(raw=r,available=rows[r]['any_TOP_segmentation_available']) for r in HELDOUT],
        rows=rows,source_record_index=pin(out/'source_record_index.jsonl'),CSV=pin(out/'availability_199.csv'),contract=pin(out/'availability_contract.json'),
        no_pointcloud_conversion=True,no_Z_or_residual_filter=True,no_frame_pool_selected=True,
        availability_not_registration_or_geometry_quality_PASS=True,GPU_forward=0,backward=0,updates=0,seconds=time.time()-start)
    write(out/'TOP_segmentation_availability.json',result)
    print(json.dumps(dict(complete=True,receipt=pin(out/'TOP_segmentation_availability.json'),available=available,
        time_window_available=window,extra_available_UNSELECTED=extra,raw_SHA=rawsha,seconds=time.time()-start)),flush=True)


if __name__=='__main__':main()
