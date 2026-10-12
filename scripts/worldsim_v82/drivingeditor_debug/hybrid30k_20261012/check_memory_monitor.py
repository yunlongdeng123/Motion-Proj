"""CPU-only：fake CUDA和fake subprocess验证显存归属、终态及失败日志。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from memory_monitor import GIB, MemoryMonitor


class FakeCuda:
    def __init__(self):
        self.owner = threading.get_ident()
        self.allocated = self.reserved = self.peak_allocated = self.peak_reserved = 0
        self.resets = 0

    def _check(self):
        if threading.get_ident() != self.owner:
            raise AssertionError('后台线程访问torch')

    def use(self, allocated, reserved):
        self.allocated, self.reserved = allocated, reserved
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)

    def is_initialized(self):
        self._check()
        return True

    def memory_allocated(self, device):
        self._check()
        return self.allocated

    def memory_reserved(self, device):
        self._check()
        return self.reserved

    def max_memory_allocated(self, device):
        self._check()
        return self.peak_allocated

    def max_memory_reserved(self, device):
        self._check()
        return self.peak_reserved

    def reset_peak_memory_stats(self, device):
        self._check()
        self.resets += 1
        self.peak_allocated, self.peak_reserved = self.allocated, self.reserved

    def synchronize(self):
        raise AssertionError('禁止monitor synchronize')


def fake_run(command, **kwargs):
    assert kwargs['timeout'] > 0 and kwargs['check'] and kwargs['capture_output']
    assert os.path.isabs(command[0]) and kwargs['close_fds'] is False
    assert '--id=0' in command and '--format=csv,noheader,nounits' in command
    if any(item.startswith('--query-gpu=') for item in command):
        output = 'GPU-test, 0, Fake GPU, 1024, 24576\n'
    else:
        output = 'GPU-test, 77, 256\nGPU-test, 77, 128\nGPU-test, 88, 640\n'
    return types.SimpleNamespace(stdout=output)


class MemoryChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.cuda = FakeCuda()

    def tearDown(self):
        self.temp.cleanup()

    def monitor(self, **kwargs):
        with mock.patch('memory_monitor.shutil.which', return_value=str(self.output / 'nvidia-smi')):
            return MemoryMonitor(self.output, torch_module=types.SimpleNamespace(cuda=self.cuda),
                                 pid=77, subprocess_run=kwargs.pop('subprocess_run', fake_run), **kwargs)

    def test_linux_spawn_unavailable_fails_without_launching(self):
        run = mock.Mock(side_effect=AssertionError('不允许执行fallback'))
        monitor = self.monitor(subprocess_run=run)
        with mock.patch('memory_monitor.sys.platform', 'linux'), \
                mock.patch('memory_monitor.subprocess._USE_POSIX_SPAWN', False, create=True):
            row = monitor.sample_now()
        self.assertEqual(len(row['errors']), 2)
        run.assert_not_called()
        monitor.close(status='failed')

    @unittest.skipUnless(sys.platform.startswith('linux') and hasattr(os, 'posix_spawn')
                         and getattr(subprocess, '_USE_POSIX_SPAWN', False),
                         '真实posix_spawn验证需Linux；仅执行/bin/echo，不访问GPU')
    def test_real_cpu_subprocess_uses_posix_spawn_and_never_fork(self):
        # echo模拟查询命令；拦截实际Popen分支，任何fork fallback直接失败。
        with mock.patch('memory_monitor.shutil.which', return_value='/bin/echo'):
            monitor = MemoryMonitor(self.output, subprocess_run=subprocess.run)
        original = subprocess.Popen._posix_spawn
        calls = []
        def observed_spawn(process, *args, **kwargs):
            calls.append(args[1])
            return original(process, *args, **kwargs)
        with mock.patch.object(subprocess.Popen, '_posix_spawn', observed_spawn), \
                mock.patch('subprocess._fork_exec', side_effect=AssertionError('fork fallback禁止'), create=True), \
                mock.patch('_posixsubprocess.fork_exec', side_effect=AssertionError('fork fallback禁止')):
            rows = monitor._query('uuid,index,name,memory.used,memory.total', 'gpu')
        self.assertTrue(rows)
        self.assertEqual(calls, ['/bin/echo'])
        monitor._run = fake_run
        monitor.close()

    def test_resets_nested_peaks_and_phase_wall(self):
        monitor = self.monitor()
        monitor.set_phase('initialization')
        self.cuda.use(1000, 1200)
        self.cuda.use(50, 100)
        monitor.set_phase('training', 1)
        resets = self.cuda.resets
        monitor.set_phase('training', 2)
        self.assertEqual(self.cuda.resets, resets)
        self.cuda.use(500, 700)
        with monitor.phase('photo_eval', 100):
            self.cuda.use(1500, 1800)
            with monitor.phase('checkpoint'):
                self.cuda.use(2000, 2500)
            self.cuda.use(70, 100)
        self.cuda.use(100, 150)
        record = monitor.update(100)
        self.assertEqual(record['pytorch']['peak_allocated_bytes'], 2000)
        self.assertEqual(record['pytorch']['peak_reserved_bytes'], 2500)
        self.assertEqual(record['phases']['initialization']['peak_allocated_bytes'], 1000)
        for name in ('training', 'photo_eval', 'checkpoint'):
            self.assertEqual(record['phases'][name]['peak_allocated_bytes'], 2000)
        self.assertGreaterEqual(record['phases']['training']['wall_seconds'],
                                record['phases']['training']['wall_seconds_exclusive'])
        self.assertTrue((self.output / 'memory_summary.json').is_file())
        monitor.close()
        phases = [json.loads(line) for line in (self.output / 'memory_phases.jsonl').read_text().splitlines()]
        self.assertEqual([row['name'] for row in phases], ['initialization', 'checkpoint', 'photo_eval', 'training'])

    def test_nvidia_pid_sum_and_missing_values(self):
        monitor = self.monitor()
        monitor.set_phase('training', 10)
        row = monitor.sample_now()
        self.assertEqual(row['process_used_bytes'], 384 * 1024 ** 2)
        self.assertEqual(row['device_used_bytes'], 1024 * 1024 ** 2)
        self.assertEqual(row['device_total_bytes'], 24 * GIB)
        self.assertFalse(row['phase_boundary_crossed'])
        monitor._run = lambda *args, **kwargs: types.SimpleNamespace(stdout='')
        row = monitor.sample_now()
        self.assertIsNone(row['device_used_bytes'])
        self.assertFalse(row['process_present'])
        self.assertEqual(row['process_used_bytes'], 0)
        self.assertEqual(monitor.summary()['nvidia_smi']['process_peak_used_bytes'], 384 * 1024 ** 2)
        monitor.close(status='failed', error=RuntimeError('failure'))

    def test_failed_context_flush_and_sampler_errors(self):
        def broken(command, **kwargs):
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        monitor = self.monitor(subprocess_run=broken)
        with self.assertRaisesRegex(RuntimeError, 'training failed'):
            with monitor:
                monitor.set_phase('training', 123)
                with monitor.phase('gradient'):
                    self.cuda.use(500, 600)
                    raise RuntimeError('training failed')
        record = json.loads((self.output / 'memory_summary.json').read_text())
        self.assertEqual(record['run_status'], 'failed')
        self.assertEqual(record['monitor_status'], 'closed')
        self.assertEqual(record['pytorch']['peak_reserved_bytes'], 600)
        self.assertGreater(record['sampling']['samples_with_errors'], 0)
        self.assertEqual(record['phases']['gradient']['failed_count'], 1)
        self.assertFalse(monitor._thread.is_alive())
        self.assertEqual(monitor.close()['pytorch'], record['pytorch'])

    def test_sampler_thread_never_touches_torch_and_assessment(self):
        threads = set()
        def run(command, **kwargs):
            threads.add(threading.get_ident())
            return fake_run(command, **kwargs)
        monitor = self.monitor(subprocess_run=run, interval=0.01)
        monitor.set_phase('training', 1)
        monitor.start()
        deadline = time.monotonic() + 1
        while monitor.summary()['sampling']['samples'] < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertGreaterEqual(monitor.summary()['sampling']['samples'], 2)
        monitor.update(1)
        self.assertNotEqual(threads, {threading.get_ident()})
        self.assertEqual(monitor.summary()['rtx3090_assessment']['headroom_bytes'], 2 * GIB)
        self.assertEqual(monitor.summary()['rtx3090_assessment']['status'], 'undetermined_before_full_30k_completion')
        monitor.update(30000)
        record = monitor.close()
        self.assertFalse(record['rtx3090_assessment']['certified_fit'])
        self.assertIn('requires_four_arms', record['rtx3090_assessment']['status'])

    def test_cross_phase_queries_are_overall_only(self):
        monitor = self.monitor()
        monitor.set_phase('training', 100)
        def changing(command, **kwargs):
            if any(item.startswith('--query-gpu=') for item in command):
                with monitor.phase('gradient'):
                    self.cuda.use(100, 200)
            return fake_run(command, **kwargs)
        monitor._run = changing
        row = monitor.sample_now()
        self.assertTrue(row['phase_boundary_crossed'])
        self.assertEqual(monitor.summary()['phases']['training']['sample_count'], 0)
        self.assertEqual(monitor.summary()['nvidia_smi']['process_peak_used_bytes'], 384 * 1024 ** 2)
        monitor._run = fake_run
        monitor.close()


if __name__ == '__main__':
    unittest.main()
