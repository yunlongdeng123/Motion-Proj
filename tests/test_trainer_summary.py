import json

from motion_proj.train.trainer import Trainer


def test_completion_summary_is_atomic_and_marks_run_complete(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.work_dir = str(tmp_path)
    trainer.experiment_type = "synthetic"
    trainer.step = 12
    trainer.stop_reason = "max_steps"
    trainer.config_fingerprint = "config-fp"
    trainer.cache_fingerprint = "cache-fp"

    summary = trainer._write_completion_summary("/tmp/step_12")

    assert summary["trained_steps"] == 12
    assert summary["checkpoint"] == "/tmp/step_12"
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert (tmp_path / "COMPLETE").read_text() == "ok\n"
