"""只测试新增 checkpoint 路由，不加载模型或 GPU。"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_resim_sample import _checkpoint_path


class Args(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def test_public_resim_checkpoint_uses_non_ema(tmp_path):
    (tmp_path / "latest").write_text("30000")
    assert _checkpoint_path(Args(load=str(tmp_path), use_ema=False)) == tmp_path / "30000/mp_rank_00_model_states.pt"


def test_resim_ema_selection_is_explicit(tmp_path):
    (tmp_path / "latest").write_text("30000")
    assert _checkpoint_path(Args(load=str(tmp_path), use_ema=True)) == tmp_path / "30000-ema/mp_rank_00_model_states.pt"
