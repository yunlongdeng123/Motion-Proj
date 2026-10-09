"""P1 下载→原始组件训练→固定验证→论文测试；产物全部在仓库外。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
PYTHON = Path('/root/autodl-tmp/envs/motionproj/bin/python')
DATA = Path('/root/autodl-tmp/data/worldsim_v81')
DEFAULT_RUN = Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-SEEN-TO-SCENE-P1-20261009/r1')
PROPAGATION_PROTOCOLS = ('paper-bidirectional-m4', 'reference-m4', 'literal-allframes')
INFERENCE_MODES = ('paper-feedforward', 'literal-public')


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


class Controller:
    def __init__(self, run: Path, propagation_protocol: str = 'literal-allframes',
                 inference_mode: str = 'literal-public'):
        if propagation_protocol not in PROPAGATION_PROTOCOLS:
            raise ValueError(f'未知传播协议: {propagation_protocol}')
        if inference_mode not in INFERENCE_MODES:
            raise ValueError(f'未知推理模式: {inference_mode}')
        if propagation_protocol == 'paper-bidirectional-m4' and inference_mode != 'paper-feedforward':
            raise ValueError('论文双向参考协议只能用paper-feedforward，不能调用公开未来父链')
        self.propagation_protocol = propagation_protocol
        self.inference_mode = inference_mode
        self.run = run.resolve()
        if self.run.is_relative_to(REPO):
            raise ValueError('run 必须在源码仓库外')
        self.run.mkdir(parents=True, exist_ok=True)
        self.lock = (self.run / 'controller.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state_path = self.run / 'controller_state.json'
        if self.state_path.is_file():
            previous = json.loads(self.state_path.read_text())
            # 旧状态里的 inference_mode 是最近一次推理，可能是 feedforward 诊断。
            previous_protocol = previous.get('propagation_protocol', 'literal-allframes')
            previous_mode = (previous.get('inference_mode', 'literal-public')
                             if 'propagation_protocol' in previous else 'literal-public')
            if (previous_protocol != propagation_protocol or previous_mode != inference_mode):
                raise ValueError('已有 run 的训练传播协议或推理模式不符；请使用独立 run 目录')
            child = previous.get('child_pid')
            if child:
                command_path = Path(f'/proc/{int(child)}/cmdline')
                if command_path.exists():
                    command = command_path.read_bytes().replace(b'\0', b' ').decode(errors='replace')
                    if 'motion_proj.worldsim_v81' in command or 'evaluate_p1.py' in command:
                        raise RuntimeError(f'先前子作业 PID {child} 仍在运行，禁止重复启动')
        self.state = {'task': 'WS-V81-SEEN-TO-SCENE-P1-20261009', 'run': 'r1',
                      'controller_pid': os.getpid(), 'status': 'starting',
                      'training_budget': 100000, 'human_in_loop': False,
                      'assistant_qa_required': True,
                      'propagation_protocol': self.propagation_protocol,
                      'inference_mode': self.inference_mode,
                      'active_inference_mode': None}
        self.env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                        PYTHONUNBUFFERED='1', PYTHONPATH=str(REPO))
        self.update()

    def update(self, **changes) -> None:
        self.state.update(changes)
        self.state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.state_path, self.state)

    def wait_archive(self, filename: str) -> None:
        path = self.run / 'downloads' / (filename + '.extract.json')
        while True:
            if path.is_file():
                obj = json.loads(path.read_text())
                if obj.get('state') == 'extraction_complete':
                    return
                if obj.get('state') in ('failed', 'extraction_failed'):
                    raise RuntimeError(f'{filename} 提取失败，见 {path}')
            self.update(status='waiting_for_verified_data', waiting_for=filename)
            time.sleep(30)

    def command(self, name: str, argv: list[str]) -> None:
        log = self.run / 'logs' / f'{name}.log'
        log.parent.mkdir(exist_ok=True)
        self.update(status=name, command=argv, log=str(log), child_pid=None)
        with log.open('a', encoding='utf-8') as handle:
            child = subprocess.Popen(argv, cwd=REPO, env=self.env, stdout=handle,
                                     stderr=subprocess.STDOUT)
            self.update(child_pid=child.pid)
            code = child.wait()
        self.update(child_pid=None, return_code=code)
        if code:
            raise RuntimeError(f'{name} exit={code}；保留日志与断点，不重复启动')

    def latest_checkpoint(self) -> Path | None:
        paths = sorted((self.run / 'train').glob('p1-checkpoint-*.pt'))
        return paths[-1] if paths else None

    def prepare_inventory(self) -> Path:
        from evaluate_p1 import YOUTUBE_IDS
        davis = DATA / 'davis_2017_480p/DAVIS'
        david_ids = []
        for split in ('train', 'val'):
            david_ids += (davis / f'ImageSets/2017/{split}.txt').read_text().split()
        if len(david_ids) != 90 or len(set(david_ids)) != 90:
            raise ValueError('DAVIS2017官方train+val必须为90个唯一序列')
        all_frames = json.loads((self.run / 'downloads/valid_all_frames.extract.json').read_text())
        roots = {'davis2017': davis / 'JPEGImages/480p',
                 'youtube_vos': Path(all_frames['rgb_root'])}
        cases = []
        for dataset, ids in (('davis2017', sorted(david_ids)),
                             ('youtube_vos', sorted(YOUTUBE_IDS))):
            for sequence in ids:
                folder = roots[dataset] / sequence
                frames = len(list(folder.glob('*.jpg')))
                if frames < 25:
                    raise ValueError(f'正式序列不足25原始帧: {folder}: {frames}')
                cases.append({'dataset': dataset, 'sequence_id': sequence,
                              'source_id': f'{dataset}/{sequence}',
                              'data_root': str(roots[dataset]), 'frames_available': frames})
        path = self.run / 'benchmark_inventory.json'
        if path.exists():
            existing = json.loads(path.read_text())['cases']
            expected_keys = {(x['dataset'], x['sequence_id'], x['source_id'], x['data_root']) for x in cases}
            actual_keys = {(x['dataset'], x['sequence_id'], x['source_id'], x['data_root']) for x in existing}
            if actual_keys != expected_keys or len(existing) != 150:
                raise ValueError('已冻结评测清单与官方全集/来源路径不一致')
        else:
            atomic_json(path, {'cases': cases, 'selection': 'official_davis90_and_paper_appendix60'})
        return path

    def train_to(self, step: int, *, save_every: int) -> Path:
        latest = self.latest_checkpoint()
        if latest and int(latest.stem.rsplit('-', 1)[1]) >= step:
            self.check_checkpoint_protocol(latest)
            return latest
        args = [str(PYTHON), '-m', 'motion_proj.worldsim_v81.train_p1',
                '--output-dir', str(self.run / 'train'), '--max-steps', str(step),
                '--save-every', str(save_every), '--keep-checkpoints', '2',
                '--amp', 'bf16', '--propagation-protocol', self.propagation_protocol]
        if latest:
            args += ['--resume', str(latest)]
        self.command(f'train_to_{step:06d}', args)
        latest = self.latest_checkpoint()
        if latest is None or int(latest.stem.rsplit('-', 1)[1]) != step:
            raise RuntimeError('训练未产出预期断点')
        self.update(training_step=step, checkpoint=str(latest))
        return latest

    def check_checkpoint_protocol(self, checkpoint: Path) -> None:
        import torch
        # mmap只读元数据，不为一次协议检查复制整份模型/优化器到内存。
        state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if state.get('propagation_protocol', 'literal-allframes') != self.propagation_protocol:
            raise ValueError('已有断点传播协议不符，禁止冒充新协议')

    def infer(self, checkpoint: Path, root: Path, sequence: str, side: float,
              output: Path, name: str, mode: str | None = None,
              *, full_video: bool = False) -> None:
        if mode is None:
            mode = self.inference_mode
        if mode not in INFERENCE_MODES:
            raise ValueError(f'未知推理模式: {mode}')
        self.check_checkpoint_protocol(checkpoint)
        result_root = output / sequence
        if full_video:
            result_root /= 'full_video'
        result = result_root / f'side_{side:g}' / 'run.json'
        if result.exists():
            completed = json.loads(result.read_text())
            if (completed.get('status') == 'complete' and completed.get('mode') == mode
                    and completed.get('checkpoint') == str(checkpoint)
                    and completed.get('propagation_protocol', 'literal-allframes') == self.propagation_protocol
                    and (not full_video or completed.get('generation_protocol') == 'full_video_sliding_window_25_stride16')):
                self.update(status=f'{name}_{mode}_skipped', active_inference_mode=mode)
                return
        self.update(active_inference_mode=mode)
        command = [str(PYTHON), '-m', 'motion_proj.worldsim_v81.infer_p1',
                           '--data-root', str(root), '--sequence-id', sequence,
                           '--checkpoint', str(checkpoint), '--output-dir', str(output),
                           '--side-ratio', str(side), '--seed', '2026', '--steps', '25',
                           '--mode', mode, '--amp', 'bf16']
        if full_video:
            command.append('--full-video')
        self.command(f'{name}_{mode}', command)

    def wait_quality_gate(self, step: int) -> None:
        path = self.run / 'quality_gates' / f'step{step:06d}.json'
        while True:
            if not path.is_file():
                self.update(status='waiting_assistant_quality_gate', quality_gate_step=step,
                            quality_gate_file=str(path), quality_gate_decision=None,
                            quality_gate_error=None)
            else:
                try:
                    gate = json.loads(path.read_text(encoding='utf-8'))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    gate = None
                    error = f'{type(exc).__name__}: {exc}'
                else:
                    error = None
                if (isinstance(gate, dict) and type(gate.get('step')) is int
                        and gate['step'] == step and gate.get('reviewer') == 'assistant'
                        and gate.get('decision') in ('continue', 'hold')):
                    if gate['decision'] == 'continue':
                        self.update(status='assistant_quality_gate_continue', quality_gate_step=step,
                                    quality_gate_file=str(path), quality_gate_decision='continue',
                                    quality_gate_error=None)
                        return
                    error = 'assistant decision: hold'
                    decision = 'hold'
                elif error is None:
                    error = 'quality gate 需要相同步数、reviewer=assistant、decision=continue|hold'
                    decision = None
                else:
                    decision = None
                self.update(status='engineering_hold', quality_gate_step=step,
                            quality_gate_file=str(path), quality_gate_decision=decision,
                            quality_gate_error=error)
            time.sleep(30)

    def validate_checkpoint(self, step: int, checkpoint: Path, valid_root: Path,
                            chosen: list[str]) -> None:
        output = self.run / f'validation/step{step:06d}'
        if step in (1000, 5000, 10000, 50000, 100000):
            for sequence in chosen:
                for side in (.125, .33):
                    self.infer(checkpoint, valid_root, sequence, side, output,
                               f'validation_{step:06d}_{sequence}_{side}')
        else:
            self.infer(checkpoint, valid_root, chosen[0], .33, output,
                       f'validation_{step:06d}')
        if step == 1000 and self.propagation_protocol != 'paper-bidirectional-m4':
            diagnostic_mode = ('literal-public' if self.inference_mode == 'paper-feedforward'
                               else 'paper-feedforward')
            diagnostic_dir = ('validation_literal' if diagnostic_mode == 'literal-public'
                              else 'validation_feedforward')
            self.infer(checkpoint, valid_root, chosen[0], .33,
                       self.run / diagnostic_dir / 'step001000',
                       f'validation_diagnostic_001000_{diagnostic_mode}', mode=diagnostic_mode)
        if step in (1000, 5000):
            self.wait_quality_gate(step)

    def execute(self) -> None:
        self.wait_archive('train.tar')
        train_root = DATA / 'youtube_vos_2019/train/JPEGImages'
        self.command('inspect_train_data', [str(PYTHON), '-m',
                     'motion_proj.worldsim_v81.train_p1', '--inspect-data-only',
                     '--output-dir', str(self.run / 'train'),
                     '--propagation-protocol', self.propagation_protocol])
        train_ids = sorted('youtube_vos/' + p.name for p in train_root.iterdir()
                           if p.is_dir() and len(list(p.glob('*.jpg'))) > 25)
        (self.run / 'train_source_ids.json').write_text(json.dumps(train_ids), encoding='utf-8')
        self.wait_archive('valid.tar')
        valid_root = DATA / 'youtube_vos_2019/valid/JPEGImages'
        # 附录60条虽在镜像valid目录，角色仍是最终测试，验证必须排除。
        sys.path.insert(0, str(REPO / 'scripts/worldsim_v81'))
        from evaluate_p1 import YOUTUBE_IDS
        valid_ids = sorted(p.name for p in valid_root.iterdir()
                           if p.is_dir() and len(list(p.glob('*.jpg'))) >= 25)
        valid_ids = [item for item in valid_ids if item not in YOUTUBE_IDS]
        if len(valid_ids) < 3:
            raise RuntimeError('少于3条合法25帧验证序列')
        # 预先按名称冻结验证例；不观察测试内容、不按生成质量挑选。
        chosen = [valid_ids[i] for i in sorted({0, len(valid_ids)//2, len(valid_ids)-1})]
        atomic_json(self.run / 'validation_selection.json',
                    {'sequence_ids': chosen, 'rule': 'sorted_first_middle_last_ge25',
                     'used_for': 'engineering_and_assistant_qa_only', 'test_tuning': False})
        checkpoint = self.train_to(2, save_every=1)
        self.infer(checkpoint, valid_root, chosen[0], .33, self.run / 'validation/step000002',
                   'validation_000002')
        # 与公开配置保持1000步保存/验证；每段恢复同一RNG，不重设训练seed。
        for step in range(1000, 100001, 1000):
            checkpoint = self.train_to(step, save_every=1000)
            self.validate_checkpoint(step, checkpoint, valid_root, chosen)
        self.wait_archive('valid_all_frames')
        self.wait_archive('DAVIS-2017-trainval-480p.zip')
        inventory_path = self.prepare_inventory()
        inventory = json.loads(inventory_path.read_text())
        cases = []
        for index, item in enumerate(inventory['cases']):
            sequence, root = item['sequence_id'], Path(item['data_root'])
            for side, total in ((.125, .25), (.33, .66)):
                output = self.run / 'benchmark' / item['dataset']
                self.infer(checkpoint, root, sequence, side, output,
                           f'benchmark_{index:03d}_{sequence}_{side}', full_video=True)
                frames = output / sequence / 'full_video' / f'side_{side:g}'
                cases.append({'dataset': item['dataset'], 'sequence_id': sequence,
                              'source_id': item['source_id'], 'mask_total_ratio': total,
                              'source_dir': str((root / sequence).resolve()),
                              **{k: str(frames / f'{k}.mp4') for k in ('gt', 'pred', 'comp')}})
                atomic_json(self.run / 'benchmark_progress.json',
                            {'completed': len(cases), 'expected': 300})
        manifest = self.run / 'benchmark_videos.json'
        atomic_json(manifest, {'cases': cases})
        self.command('formal_metrics', [str(PYTHON), str(REPO / 'scripts/worldsim_v81/evaluate_p1.py'),
                     '--manifest', str(manifest), '--train-source-ids',
                     str(self.run / 'train_source_ids.json'), '--device', 'cuda',
                     '--generation-protocol', 'full-video',
                     '--output', str(self.run / 'metrics.json')])
        self.update(status='metrics_complete_assistant_review_pending',
                    metrics=str(self.run / 'metrics.json'), command=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--propagation-protocol', choices=PROPAGATION_PROTOCOLS,
                        default='literal-allframes')
    parser.add_argument('--inference-mode', choices=INFERENCE_MODES,
                        default='literal-public')
    args = parser.parse_args()
    controller = Controller(args.run, args.propagation_protocol, args.inference_mode)
    try:
        controller.execute()
    except Exception as exc:
        controller.update(status='failed', error=f'{type(exc).__name__}: {exc}', child_pid=None)
        raise


if __name__ == '__main__':
    main()
