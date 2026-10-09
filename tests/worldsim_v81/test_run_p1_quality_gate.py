"""P1 controller 的轻量质量门契约；不启动训练或 GPU 推理。"""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def run_p1(monkeypatch):
    # fcntl 只在远端 Linux 初始化 controller 时使用，本地仅测试流程方法。
    monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace())
    path = Path(__file__).resolve().parents[2] / 'scripts/worldsim_v81/run_p1.py'
    spec = importlib.util.spec_from_file_location('run_p1_quality_gate_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_controller(run_p1, tmp_path):
    controller = object.__new__(run_p1.Controller)
    controller.run = tmp_path
    states = []
    controller.update = lambda **changes: states.append(changes)
    return controller, states


def write_gate(run, filename_step, payload):
    folder = run / 'quality_gates'
    folder.mkdir(exist_ok=True)
    (folder / f'step{filename_step:06d}.json').write_text(json.dumps(payload), encoding='utf-8')


def test_checkpoint_validation_order_and_modes(run_p1, tmp_path):
    controller, _ = fake_controller(run_p1, tmp_path)
    events = []
    controller.infer = lambda checkpoint, root, sequence, side, output, name, mode='literal-public': (
        events.append(('infer', sequence, side, output, mode)))
    controller.wait_quality_gate = lambda step: events.append(('gate', step))
    chosen = ['first', 'middle', 'last']
    checkpoint = tmp_path / 'p1-checkpoint-001000.pt'

    controller.validate_checkpoint(1000, checkpoint, tmp_path, chosen)
    assert len(events) == 8
    assert [(e[1], e[2]) for e in events[:6]] == [
        (sequence, side) for sequence in chosen for side in (.125, .33)]
    assert all(e[3] == tmp_path / 'validation/step001000' and e[4] == 'literal-public'
               for e in events[:6])
    assert events[6] == ('infer', 'first', .33,
                         tmp_path / 'validation_feedforward/step001000', 'paper-feedforward')
    assert events[7] == ('gate', 1000)

    events.clear()
    controller.validate_checkpoint(5000, checkpoint, tmp_path, chosen)
    assert len(events) == 7 and events[-1] == ('gate', 5000)
    assert all(e[4] == 'literal-public' for e in events[:6])

    events.clear()
    controller.validate_checkpoint(2000, checkpoint, tmp_path, chosen)
    assert events == [('infer', 'first', .33,
                       tmp_path / 'validation/step002000', 'literal-public')]


def test_infer_records_mode_and_skips_only_matching_completed_result(run_p1, tmp_path):
    controller, states = fake_controller(run_p1, tmp_path)
    commands = []
    controller.command = lambda name, argv: commands.append((name, argv))
    output = tmp_path / 'validation/step001000'
    checkpoint = tmp_path / 'p1-checkpoint-001000.pt'

    controller.infer(checkpoint, tmp_path, 'first', .33, output, 'validation_001000')
    assert commands[0][0].endswith('literal-public')
    assert commands[0][1][commands[0][1].index('--mode') + 1] == 'literal-public'
    result = output / 'first/side_0.33/run.json'
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({'status': 'complete', 'mode': 'literal-public'}))
    controller.infer(checkpoint, tmp_path, 'first', .33, output, 'validation_001000')
    assert len(commands) == 1
    assert states[-1]['status'].endswith('literal-public_skipped')

    controller.infer(checkpoint, tmp_path, 'first', .33, output, 'diagnostic',
                     mode='paper-feedforward')
    assert len(commands) == 2
    assert commands[-1][1][commands[-1][1].index('--mode') + 1] == 'paper-feedforward'
    assert states[-1]['inference_mode'] == 'paper-feedforward'


def test_missing_gate_waits_then_matching_assistant_continue_passes(run_p1, tmp_path, monkeypatch):
    controller, states = fake_controller(run_p1, tmp_path)
    write_gate(tmp_path, 5000, {'step': 5000, 'reviewer': 'assistant', 'decision': 'continue'})
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        write_gate(tmp_path, 1000, {'step': 1000, 'reviewer': 'assistant',
                                    'decision': 'continue'})

    monkeypatch.setattr(run_p1.time, 'sleep', sleep)
    controller.wait_quality_gate(1000)
    assert sleeps == [30]
    assert [state['status'] for state in states] == [
        'waiting_assistant_quality_gate', 'assistant_quality_gate_continue']


def test_hold_waits_for_correction(run_p1, tmp_path, monkeypatch):
    controller, states = fake_controller(run_p1, tmp_path)
    write_gate(tmp_path, 1000, {'step': 1000, 'reviewer': 'assistant', 'decision': 'hold'})

    def sleep(seconds):
        assert seconds == 30
        write_gate(tmp_path, 1000, {'step': 1000, 'reviewer': 'assistant',
                                    'decision': 'continue'})

    monkeypatch.setattr(run_p1.time, 'sleep', sleep)
    controller.wait_quality_gate(1000)
    assert [state['status'] for state in states] == [
        'engineering_hold', 'assistant_quality_gate_continue']
    assert states[0]['quality_gate_decision'] == 'hold'


@pytest.mark.parametrize('payload', [
    {'step': 5000, 'reviewer': 'assistant', 'decision': 'continue'},
    {'step': True, 'reviewer': 'assistant', 'decision': 'continue'},
    {'step': 1000, 'reviewer': 'human', 'decision': 'continue'},
    {'step': 1000, 'reviewer': 'assistant', 'decision': 'proceed'},
])
def test_invalid_gate_cannot_continue(run_p1, tmp_path, monkeypatch, payload):
    controller, states = fake_controller(run_p1, tmp_path)
    write_gate(tmp_path, 1000, payload)

    class StopWaiting(Exception):
        pass

    monkeypatch.setattr(run_p1.time, 'sleep', lambda seconds: (_ for _ in ()).throw(StopWaiting()))
    with pytest.raises(StopWaiting):
        controller.wait_quality_gate(1000)
    assert states[-1]['status'] == 'engineering_hold'
