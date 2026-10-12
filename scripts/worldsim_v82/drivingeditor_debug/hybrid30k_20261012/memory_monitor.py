"""轻量显存日志；调用方传入 torch，后台线程只查询 nvidia-smi。

set_phase 用于持续阶段，同名调用只更新迭代；phase 用于嵌套阶段。
所有 torch 读取/reset 必须在构造本对象的训练线程，均不 synchronize。
父阶段峰值包含子阶段，进入阶段时已存活的分配也计入，不能相加。
"""
import contextlib
import csv
import datetime
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


GIB = 1024 ** 3
PEAK_KEYS = ('peak_allocated_bytes', 'peak_reserved_bytes',
             'process_peak_used_bytes', 'device_peak_used_bytes')
CURRENT_KEYS = ('current_allocated_bytes', 'current_reserved_bytes')


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _merge(target, values):
    for key in PEAK_KEYS:
        value = values.get(key)
        if value is not None:
            target[key] = max(target.get(key) or 0, value)
    for key in CURRENT_KEYS:
        if key in values:
            target[key] = values[key]


def _mib(value):
    value = value.strip()
    if value in ('N/A', '[N/A]', '[Not Supported]', 'Not Supported', ''):
        return None
    number = float(value)
    if number < 0 or not number < float('inf'):
        raise ValueError('nvidia-smi 显存数字无效')
    return int(number * 1024 ** 2)


class MemoryMonitor:
    def __init__(self, output, torch_module=None, device=0, nvidia_device='0',
                 interval=1.0, pid=None, expected_iterations=30000,
                 headroom_bytes=2 * GIB, subprocess_run=None, query_timeout=2.0):
        if interval <= 0 or query_timeout <= 0 or headroom_bytes < 0:
            raise ValueError('采样间隔/查询timeout必须为正，余量不能为负')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if any((self.output / name).exists() for name in
               ('memory_samples.jsonl', 'memory_phases.jsonl', 'memory_summary.json')):
            raise FileExistsError('显存日志已存在；拒绝覆盖或混入旧run')
        self.torch, self.device, self.nvidia_device = torch_module, device, str(nvidia_device)
        self.pid = os.getpid() if pid is None else int(pid)
        self.interval, self.query_timeout = float(interval), float(query_timeout)
        self.expected_iterations, self.headroom = int(expected_iterations), int(headroom_bytes)
        self._run = subprocess.run if subprocess_run is None else subprocess_run
        self._nvidia_smi = shutil.which('nvidia-smi')
        self._owner = threading.get_ident()
        self._started, self._started_utc = time.monotonic(), _utc()
        self._lock, self._stop = threading.RLock(), threading.Event()
        self._thread = None
        self._closed = False
        self._iteration = None
        self._status, self._failure = 'running', None
        self._extra, self._overall, self._stack, self._phase_totals = {}, {}, [], {}
        self._phase_id = self._revision = self._resets = self._reset_errors = 0
        self._sample_count = self._sample_errors = self._device_samples = self._pid_samples = 0
        self._boundary_samples = self._error_count = 0
        self._errors, self._nvidia = [], {}

    def _assert_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError('PyTorch阶段/摘要操作只允许训练线程调用')

    def _error(self, source, error):
        self._error_count += 1
        self._errors.append({'source': source, 'error': repr(error), 'utc': _utc()})
        self._errors = self._errors[-16:]

    def _append(self, name, record):
        try:
            with (self.output / name).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        except (OSError, TypeError, ValueError) as error:
            self._error(name, error)

    def _capture(self):
        """重置前先累积所有活跃祖先，整体峰值不会被reset丢失。"""
        if self.torch is None:
            return
        try:
            cuda = self.torch.cuda
            if not cuda.is_initialized():
                return
            values = {
                'peak_allocated_bytes': int(cuda.max_memory_allocated(self.device)),
                'peak_reserved_bytes': int(cuda.max_memory_reserved(self.device)),
                'current_allocated_bytes': int(cuda.memory_allocated(self.device)),
                'current_reserved_bytes': int(cuda.memory_reserved(self.device)),
            }
            _merge(self._overall, values)
            for frame in self._stack:
                _merge(frame, values)
        except Exception as error:
            self._error('torch_snapshot', error)

    def _reset(self):
        if self.torch is None:
            return
        try:
            if self.torch.cuda.is_initialized():
                self.torch.cuda.reset_peak_memory_stats(self.device)
                self._resets += 1
        except Exception as error:
            self._reset_errors += 1
            self._error('torch_reset', error)

    def _push(self, name, kind):
        self._capture()
        self._phase_id += 1
        frame = {'phase_id': self._phase_id, 'name': str(name), 'kind': kind,
                 'parent_phase_id': self._stack[-1]['phase_id'] if self._stack else None,
                 'start_iteration': self._iteration, 'start_elapsed_seconds': time.monotonic() - self._started,
                 'children_wall_seconds': 0.0, 'sample_count': 0}
        self._stack.append(frame)
        self._revision += 1
        self._reset()
        self._capture()  # 已存活分配是新阶段的基线。
        return frame

    def _finish(self, frame, status='complete', error=None):
        if self._stack[-1] is not frame:
            raise RuntimeError('显存phase必须按嵌套栈顺序退出')
        self._capture()
        self._stack.pop()
        self._revision += 1
        end = time.monotonic() - self._started
        wall = end - frame['start_elapsed_seconds']
        exclusive = max(0.0, wall - frame['children_wall_seconds'])
        if self._stack:
            self._stack[-1]['children_wall_seconds'] += wall
        row = dict(frame, status=status, end_iteration=self._iteration,
                   end_elapsed_seconds=end, wall_seconds=wall,
                   wall_seconds_exclusive=exclusive, error=None if error is None else repr(error),
                   peak_scope='inclusive_of_children_and_live_entry_allocations')
        self._append('memory_phases.jsonl', row)
        total = self._phase_totals.setdefault(frame['name'],
                    {'count': 0, 'failed_count': 0, 'wall_seconds': 0.0,
                     'wall_seconds_exclusive': 0.0, 'sample_count': 0})
        total['count'] += 1
        total['failed_count'] += status == 'failed'
        total['wall_seconds'] += wall
        total['wall_seconds_exclusive'] += exclusive
        total['sample_count'] += frame['sample_count']
        _merge(total, frame)

    def set_phase(self, name, iteration=None):
        self._assert_owner()
        with self._lock:
            if self._closed:
                raise RuntimeError('监控已关闭')
            if iteration is not None:
                self._iteration = int(iteration)
            if any(frame['kind'] == 'context' for frame in self._stack):
                raise RuntimeError('嵌套phase中不能切换持续phase')
            if self._stack and self._stack[-1]['name'] == str(name):
                return self
            if self._stack:
                self._finish(self._stack[-1])
            self._push(name, 'persistent')
        return self

    @contextlib.contextmanager
    def phase(self, name, iteration=None):
        self._assert_owner()
        with self._lock:
            if self._closed:
                raise RuntimeError('监控已关闭')
            if iteration is not None:
                self._iteration = int(iteration)
            frame = self._push(name, 'context')
        failure = None
        try:
            yield self
        except BaseException as error:
            failure = error
            raise
        finally:
            with self._lock:
                if not self._closed:
                    self._finish(frame, 'failed' if failure is not None else 'complete', failure)
                    self._reset()
                    self._capture()

    def _query(self, fields, kind):
        # CPython 3.9 fastpath要求含目录的executable和close_fds=False。
        # CUDA已初始化的多线程进程禁止退回fork及其pthread_atfork hooks。
        if not self._nvidia_smi or not os.path.isabs(self._nvidia_smi):
            raise RuntimeError('nvidia-smi绝对路径不可用；跳过采样，禁止fork fallback')
        if sys.platform.startswith('linux'):
            if (not callable(getattr(os, 'posix_spawn', None))
                    or not getattr(subprocess, '_USE_POSIX_SPAWN', False)
                    or not callable(getattr(subprocess.Popen, '_posix_spawn', None))):
                raise RuntimeError('Linux posix_spawn不可用；跳过采样，禁止fork fallback')
            # 所有标准fd须存在，保证新PIPE fd>2，满足CPython fastpath。
            for descriptor in (0, 1, 2):
                os.fstat(descriptor)
        result = self._run([self._nvidia_smi, '--id=' + self.nvidia_device,
                            '--query-' + kind + '=' + fields, '--format=csv,noheader,nounits'],
                           close_fds=False, check=True, capture_output=True,
                           text=True, timeout=self.query_timeout)
        return list(csv.reader(io.StringIO(result.stdout.strip()))) if result.stdout.strip() else []

    def sample_now(self):
        """仅nvidia-smi；允许采样线程调用，不访问torch。"""
        with self._lock:
            if self._closed:
                return None
            revision = self._revision
            start_stack = [frame['phase_id'] for frame in self._stack]
            row = {'utc_start': _utc(), 'elapsed_start_seconds': time.monotonic() - self._started,
                   'iteration_start': self._iteration, 'phase_ids_start': start_stack,
                   'phase_names_start': [frame['name'] for frame in self._stack],
                   'pid': self.pid, 'nvidia_device': self.nvidia_device, 'errors': [],
                   'process_used_bytes': None, 'device_used_bytes': None, 'device_total_bytes': None}
        try:
            rows = self._query('uuid,index,name,memory.used,memory.total', 'gpu')
            if len(rows) != 1 or len(rows[0]) != 5:
                raise ValueError('nvidia-smi必须返回指定设备的一行')
            uuid, index, name, used, total = (field.strip() for field in rows[0])
            row.update(device_uuid=uuid, device_index=int(index), device_name=name,
                       device_used_bytes=_mib(used), device_total_bytes=_mib(total))
        except Exception as error:
            row['errors'].append({'source': 'nvidia_device', 'error': repr(error)})
        try:
            rows = self._query('gpu_uuid,pid,used_memory', 'compute-apps')
            if any(len(item) != 3 for item in rows):
                raise ValueError('nvidia-smi进程行格式无效')
            matching = [item for item in rows if int(item[1].strip()) == self.pid
                        and (not row.get('device_uuid') or item[0].strip() == row['device_uuid'])]
            values = [_mib(item[2]) for item in matching]
            row['process_present'] = bool(matching)
            row['process_used_bytes'] = (None if any(value is None for value in values) else sum(values))
        except Exception as error:
            row['errors'].append({'source': 'nvidia_process', 'error': repr(error)})
        with self._lock:
            if self._closed:  # join超时后晚到的daemon结果不能改写终态。
                return None
            row.update(utc_end=_utc(), elapsed_end_seconds=time.monotonic() - self._started,
                       iteration_end=self._iteration,
                       phase_ids_end=[frame['phase_id'] for frame in self._stack],
                       phase_boundary_crossed=revision != self._revision)
            self._sample_count += 1
            self._sample_errors += bool(row['errors'])
            self._device_samples += row['device_used_bytes'] is not None
            self._pid_samples += bool(row.get('process_present')) and row['process_used_bytes'] is not None
            self._boundary_samples += row['phase_boundary_crossed']
            values = {'process_peak_used_bytes': row['process_used_bytes'],
                      'device_peak_used_bytes': row['device_used_bytes']}
            _merge(self._overall, values)
            for key in ('device_uuid', 'device_index', 'device_name', 'device_total_bytes'):
                if row.get(key) is not None:
                    self._nvidia[key] = row[key]
            self._nvidia['current_process_used_bytes'] = row['process_used_bytes']
            self._nvidia['current_device_used_bytes'] = row['device_used_bytes']
            if not row['phase_boundary_crossed']:
                for frame in self._stack:
                    _merge(frame, values)
                    frame['sample_count'] += 1
            for error in row['errors']:
                self._error(error['source'], error['error'])
            self._append('memory_samples.jsonl', row)
        return row

    def _worker(self):
        deadline = time.monotonic()
        while not self._stop.is_set():
            self.sample_now()
            deadline += self.interval
            now = time.monotonic()
            if deadline < now:
                deadline = now + self.interval
            if self._stop.wait(max(0.0, deadline - now)):
                break

    def start(self):
        self._assert_owner()
        with self._lock:
            if not self._closed and self._thread is None:
                self._thread = threading.Thread(target=self._worker, name='gpu-memory-monitor', daemon=True)
                self._thread.start()
        return self

    def summary(self):
        """返回已有观测的快照；本方法不查询GPU、不调用torch。"""
        with self._lock:
            elapsed = time.monotonic() - self._started
            phases = {name: dict(value) for name, value in self._phase_totals.items()}
            for index, frame in enumerate(self._stack):
                total = phases.setdefault(frame['name'], {'count': 0, 'failed_count': 0,
                      'wall_seconds': 0.0, 'wall_seconds_exclusive': 0.0, 'sample_count': 0})
                wall = elapsed - frame['start_elapsed_seconds']
                live_child = (elapsed - self._stack[index + 1]['start_elapsed_seconds']
                              if index + 1 < len(self._stack) else 0.0)
                total['active_count'] = total.get('active_count', 0) + 1
                total['wall_seconds'] += wall
                total['wall_seconds_exclusive'] += max(0.0, wall - frame['children_wall_seconds'] - live_child)
                total['sample_count'] += frame['sample_count']
                _merge(total, frame)
            bound = max((self._overall.get(key) or 0 for key in PEAK_KEYS), default=0)
            budget = 24 * GIB - self.headroom
            full = self._status == 'complete' and (self._iteration or 0) >= self.expected_iterations
            if bound > budget:
                verdict = 'observed_peak_exceeds_preliminary_headroom_budget'
            elif not full:
                verdict = 'undetermined_before_full_30k_completion'
            elif not self._device_samples or not self._pid_samples or self._overall.get('peak_reserved_bytes') is None:
                verdict = 'undetermined_missing_memory_evidence'
            else:
                verdict = 'single_run_observed_within_budget_requires_four_arms_and_target_gpu_validation'
            return {'schema_version': 1, 'started_utc': self._started_utc,
                    'monitor_status': 'closed' if self._closed else 'running', 'run_status': self._status,
                    'error': self._failure, 'pid': self.pid, 'torch_device': str(self.device),
                    'nvidia_device': self.nvidia_device, 'iteration': self._iteration,
                    'expected_iterations': self.expected_iterations, 'wall_seconds': elapsed,
                    'phase_stack': [frame['name'] for frame in self._stack],
                    'pytorch': {key: self._overall.get(key) for key in PEAK_KEYS[:2] + CURRENT_KEYS},
                    'nvidia_smi': dict(self._nvidia,
                         process_peak_used_bytes=self._overall.get('process_peak_used_bytes'),
                         device_peak_used_bytes=self._overall.get('device_peak_used_bytes')),
                    'sampling': {'interval_seconds': self.interval, 'samples': self._sample_count,
                         'samples_with_errors': self._sample_errors, 'valid_device_samples': self._device_samples,
                         'valid_pid_present_samples': self._pid_samples, 'phase_boundary_samples': self._boundary_samples,
                         'sampled_peaks_are_lower_bounds': True, 'query_timeout_seconds': self.query_timeout,
                         'nvidia_smi_executable': self._nvidia_smi,
                         'linux_launch_policy': 'posix_spawn_fastpath_required; no_fork_fallback'},
                    'phases': phases, 'phase_attribution': {
                         'pytorch_peak_scope': 'inclusive_of_children_and_live_entry_allocations',
                         'nvidia_boundary_samples': 'overall_only_when_any_phase_boundary_crossed_during_queries',
                         'torch_peak_resets': self._resets, 'torch_reset_errors': self._reset_errors,
                         'requires_exclusive_ownership_of_torch_peak_resets': True,
                         'wall_time_scope': 'host_elapsed; parent_inclusive; exclusive_subtracts_child_wall'},
                    'rtx3090_assessment': {'status': verdict, 'certified_fit': False,
                         'reference_capacity_bytes': 24 * GIB, 'headroom_bytes': self.headroom,
                         'preliminary_budget_bytes': budget, 'observed_lower_bound_bytes': bound or None,
                         'device_samples_include_other_processes': True,
                         'four_arms_30000_iterations_all_phases_required': True,
                         'target_3090_validation_required': True,
                         'caveat': 'Polling misses brief peaks; allocator peaks exclude non-PyTorch CUDA use. '
                                   '30k densification, evaluation/LPIPS and checkpoints must be covered. '
                                   'A sampled or allocator peak below 24 GiB does not certify fit or speed.'},
                    'error_count': self._error_count, 'recent_errors': list(self._errors), 'extra': dict(self._extra)}

    def flush(self):
        self._assert_owner()
        with self._lock:
            if not self._closed:
                self._capture()
            record = self.summary()
            path = self.output / 'memory_summary.json'
            temporary = path.with_suffix('.json.tmp')
            try:
                temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
                os.replace(temporary, path)
            except (OSError, TypeError, ValueError) as error:
                self._error('memory_summary.json', error)
                record = self.summary()
            return record

    def update(self, iteration, extra=None):
        self._assert_owner()
        with self._lock:
            if self._closed:
                return self.summary()
            self._iteration = int(iteration)
            if extra:
                self._extra.update(extra)
            self._capture()
            return self.flush() if self._iteration == 1 or self._iteration % 100 == 0 else self.summary()

    def close(self, status='complete', error=None):
        self._assert_owner()
        with self._lock:
            if self._closed:
                return self.summary()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.query_timeout + 1.0)
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            if alive:
                self._error('sampler_shutdown', RuntimeError('bounded daemon join timed out'))
        if not alive:
            self.sample_now()  # 最后一次nvidia观测；没有同步CUDA。
        with self._lock:
            self._status, self._failure = str(status), None if error is None else repr(error)
            self._capture()
            while self._stack:
                frame = self._stack[-1]
                phase_status = ('failed' if status == 'failed' else
                                'interrupted' if frame['kind'] == 'context' else 'complete')
                self._finish(frame, phase_status, error)
            self._closed = True
            return self.flush()

    def __enter__(self):
        return self.start()

    def __exit__(self, kind, error, traceback):
        self.close('failed' if error is not None else 'complete', error)
        return False
