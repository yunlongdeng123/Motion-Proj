import sys
from pathlib import Path
import pytest,torch
sys.path.insert(0,str(Path(__file__).parent))
from checkpoint_contract import complete_training_state
def data():
    encoder={f'first_stage_model.encoder.layer{i}':torch.ones(1) for i in range(106)}
    base={'model.decoder':torch.ones(2)}
    return dict(base,**encoder),base,encoder
def test_complete_pretrained_state():
    expected,base,encoder=data();assert set(complete_training_state(expected,base,encoder))==set(expected)
def test_missing_encoder_fails_closed():
    expected,base,encoder=data();encoder.pop(next(iter(encoder)))
    with pytest.raises(ValueError):complete_training_state(expected,base,encoder)
def test_unexpected_backbone_missing_fails_closed():
    expected,base,encoder=data();expected['model.other']=torch.ones(1)
    with pytest.raises(ValueError):complete_training_state(expected,base,encoder)
def test_wrong_shape_fails_closed():
    expected,base,encoder=data();base['model.decoder']=torch.ones(3)
    with pytest.raises(ValueError):complete_training_state(expected,base,encoder)
