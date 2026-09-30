"""针对训练监督错接与缩放泄漏的回归检查，不认证视觉质量。"""
import numpy as np
import torch
from train_pilot import resized,sample
from iteration2.driveeditor_contract import equal

torch.set_num_threads(2)
def pair():
    rng=np.random.default_rng(16);y=rng.integers(0,256,(10,80,144,3),dtype='uint8');x=y.copy();h=np.zeros((10,80,144),bool);h[:,17:63,33:109]=True;x[h]=255
    return y,x,h

def test_hidden_actor_is_absent_from_all_fields():
    y,x,h=pair();a=sample(*resized(y,x,h,[64,128]),71);x[h]=0;b=sample(*resized(y,x,h,[64,128]),71)
    assert all(equal(a[k],b[k]) for k in a)

def test_hidden_real_background_changes_only_supervision():
    y,x,h=pair();a=sample(*resized(y,x,h,[64,128]),71);y[h]=255-y[h];b=sample(*resized(y,x,h,[64,128]),71)
    assert not equal(a['jpg'],b['jpg'])
    assert all(equal(a[k],b[k]) for k in a if k!='jpg')

def test_visible_context_is_retained():
    y,x,h=pair();a=sample(*resized(y,x,h,[64,128]),71);y[:,0:10,0:10]=0;x[:,0:10,0:10]=0;b=sample(*resized(y,x,h,[64,128]),71)
    assert not equal(a['cond_frames_without_noise'],b['cond_frames_without_noise'])

def test_unmasked_synthetic_influence_rejected():
    y,x,h=pair();x[:,2,3]=255
    try:resized(y,x,h,[64,128])
    except ValueError:return
    raise AssertionError('未拒绝mask外合成影响')
