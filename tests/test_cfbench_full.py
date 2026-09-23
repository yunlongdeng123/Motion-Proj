"""针对原始参考导出和公共时间窗的CPU回归。"""
from pathlib import Path
import json
import sys

import av
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from add_cfbench_original_videos import make_original
from run_resim_cfbench_queue import trim_video
from build_cfbench_html import describe_case, task_intent_html


def test_all_frozen_cases_have_explicit_counterfactual_intents():
    cases=json.loads((ROOT/'docs/autoresearch/worldsim_v75/downstream_bench/cases.json').read_text())['cases']
    descriptions={c['case_id']:describe_case({'case':c,'event_output_frame':15,'fps':30,'score_frames':70}) for c in cases}
    assert len(descriptions)==24
    for description in descriptions.values():
        assert all(description[k] for k in ['title','target','factual','counterfactual','timing','unchanged','expected_visible_change'])
        assert '0.5秒' in description['timing'] and '2.3秒' in description['timing']
        assert '目标，不是执行结果' in task_intent_html(description,{})
    assert '自车减速' in descriptions['CFB-SPEED-EGO-01']['title']
    assert '0.5×' in descriptions['CFB-SPEED-EGO-01']['counterfactual']
    assert '加速' in descriptions['CFB-SPEED-ACTOR-02']['title']
    assert '1.5×' in descriptions['CFB-SPEED-ACTOR-02']['counterfactual']
    assert '-3.5 m' in descriptions['CFB-LATERAL-EGO-02']['title']
    assert '+3.5 m' in descriptions['CFB-LATERAL-ACTOR-02']['title']
    assert '挂车 #15' in descriptions['CFB-REMOVE-01']['title']
    assert '删除' in descriptions['CFB-REMOVE-01']['counterfactual']
    assert '供体原车继续保留' in descriptions['CFB-INSERT-02']['counterfactual']
    assert 'X +8 / Y -3.5 / Z +0 米' in descriptions['CFB-INSERT-02']['counterfactual']
    warning=task_intent_html(descriptions['CFB-SPEED-EGO-02'],{'target_endpoint_delta_m':.000053126})
    assert '0.000053米' in warning


def test_original_reference_window_and_idempotency(tmp_path):
    source=tmp_path/'data'/'179'/'images'
    source.mkdir(parents=True)
    case={'case_id':'fixture','dataset':{'root':str(tmp_path/'data'),'scene_id':'179'},'anchor':{'event_frame':9,'pre_frames':5,'rollout_frames':19}}
    for f in range(4,28):
        Image.new('RGB',(1600,900),(f*3,20,30)).save(source/f'{f:03d}_2.jpg')
    out=tmp_path/'output'
    record=make_original(case,2,out)
    assert record['source_frames']==list(range(4,28))
    assert record['camera_index']==2
    timestamp=(out/'original-nuscenes.mp4').stat().st_mtime_ns
    assert make_original(case,2,out)==record
    assert (out/'original-nuscenes.mp4').stat().st_mtime_ns==timestamp
    with pytest.raises(AssertionError):
        make_original(case,1,out)


def test_resim_common_window_is_indices_4_through_27(tmp_path):
    native=tmp_path/'native.mp4'
    with av.open(str(native),'w') as writer:
        stream=writer.add_stream('libx264',rate=10)
        stream.width,stream.height,stream.pix_fmt=896,512,'yuv420p'
        for i in range(49):
            rgb=np.full((512,896,3),i*4,np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
    output=tmp_path/'trimmed.mp4'
    trim_video(native,output)
    with av.open(str(output)) as reader:
        values=[np.mean(f.to_ndarray(format='rgb24')) for f in reader.decode(video=0)]
    assert len(values)==24
    assert abs(values[0]-16)<5 and abs(values[-1]-108)<5
