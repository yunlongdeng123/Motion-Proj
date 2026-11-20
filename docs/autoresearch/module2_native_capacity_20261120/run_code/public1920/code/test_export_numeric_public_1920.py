import json
import pytest
import export_numeric_public_1920 as e


def test_incomplete_numeric_cannot_be_selected_or_exported(tmp_path):
    completion=tmp_path/'result.json';completion.write_text(json.dumps({'completed':False,'held_score':0.}))
    with pytest.raises(RuntimeError,match='not completed'):e.inspect(tmp_path,tmp_path/'colors.pt',completion)


def test_invalid_new_lease_is_rejected_before_output_or_cuda(tmp_path):
    auth=tmp_path/'auth.json';auth.write_text(json.dumps({'GPU_authorized':True,'optimizer_updates':0,'max_native_forwards':40}))
    lease=tmp_path/'lease.json';lease.write_text(json.dumps({'gpu_authorized':True,'active':True,'owner':'different-owner'}))
    out=tmp_path/'out'
    with pytest.raises(RuntimeError,match='caller-owned'):e.execute({},out,lease,'numeric-public',auth)
    assert not out.exists()
