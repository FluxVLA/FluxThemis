"""Sample whole-device GPU load without importing CUDA or changing workers."""

import csv
import json
import math
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SAMPLE_FIELDS = ('timestamp', 'gpu_id', 'index', 'uuid', 'utilization_gpu_pct',
                 'memory_used_mib', 'memory_total_mib')


def numeric(value):
    """Unavailable driver counters are missing data, never zero load."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def query_gpus(gpu_ids):
    result = subprocess.run([
        'nvidia-smi', '-i', ','.join(gpu_ids),
        '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total',
        '--format=csv,noheader,nounits'
    ],
                            check=True,
                            capture_output=True,
                            text=True,
                            timeout=3)
    timestamp = time.time()
    samples = []
    for values in csv.reader(result.stdout.splitlines()):
        if len(values) != 5:
            raise ValueError(f'Unexpected nvidia-smi row: {values!r}')
        index, uuid, utilization, used, total = [v.strip() for v in values]
        matches = [
            gpu for gpu in gpu_ids if gpu == index or (
                gpu.startswith('GPU-') and uuid.startswith(gpu))
        ]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous/unrequested GPU: {index}, {uuid}')
        samples.append(
            dict(
                zip(SAMPLE_FIELDS,
                    (timestamp, matches[0], index, uuid, numeric(utilization),
                     numeric(used), numeric(total)))))
    if (len(samples) != len(gpu_ids) or {row['gpu_id']
                                         for row in samples} != set(gpu_ids)):
        raise ValueError('nvidia-smi did not return every selected GPU once')
    return samples


class GpuMonitor:
    """Bounded queries in a background thread; stream samples to disk."""

    def __init__(self, directory, gpu_ids, interval=1.0):
        if not math.isfinite(interval) or interval < 0.2:
            raise ValueError('GPU sample interval must be at least 0.2s')
        self.directory = Path(directory)
        self.gpu_ids = list(gpu_ids)
        self.interval = interval
        self.started_at = self.stopped_at = None
        self.query_errors = 0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self.started_at = time.time()
        # Create the artifact synchronously so a permissions error is visible.
        with (self.directory / 'gpu_samples.csv').open('w', newline='') as out:
            csv.DictWriter(out, fieldnames=SAMPLE_FIELDS).writeheader()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self):
        try:
            with (self.directory / 'gpu_samples.csv').open(
                    'a', newline='') as out:
                writer = csv.DictWriter(out, fieldnames=SAMPLE_FIELDS)
                while not self._stop.is_set():
                    tick = time.monotonic()
                    try:
                        samples = query_gpus(self.gpu_ids)
                        writer.writerows(samples)
                        out.flush()
                    except (OSError, ValueError,
                            subprocess.SubprocessError) as exc:
                        self._record_error(exc)
                    self._stop.wait(
                        max(0, self.interval - (time.monotonic() - tick)))
        except Exception as exc:
            self._record_error(exc)

    def _record_error(self, exc):
        self.query_errors += 1
        if self.query_errors == 1:
            prefix = '\n' if sys.stderr.isatty() else ''
            print(
                f'{prefix}GPU monitoring warning: {exc}; '
                'see gpu_monitor_errors.log',
                file=sys.stderr,
                flush=True)
        try:
            with (self.directory / 'gpu_monitor_errors.log').open('a') as out:
                out.write(f'{datetime.now(timezone.utc).isoformat()}: {exc}\n')
        except OSError:
            pass

    def stop(self):
        self._stop.set()
        if self._thread is not None and self._thread.ident is not None:
            # An in-flight query is bounded by its three-second timeout.
            self._thread.join()
        self.stopped_at = time.time()


def epoch(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def event_windows(journal, started, stopped):
    """Use reported timestamps, not delayed terminal log reads."""
    phases = {'total': (started, stopped)}
    tasks, active = {}, set()
    run_start = run_end = None
    status = 'unavailable'
    if journal is not None and journal.is_file():
        with journal.open() as stream:
            for line in stream:
                # A killed writer can leave an incomplete final JSON line.
                if not line.endswith('\n'):
                    continue
                event = json.loads(line)
                payload, kind = event['payload'], event['event_type']
                if kind == 'run_start':
                    # payload.started_at is setup completion, before run().
                    run_start = epoch(event['recorded_at'])
                elif kind == 'run_end':
                    run_end = epoch(payload['finished_at'])
                    status = payload['status']
                elif kind in ('episode_start', 'episode_end'):
                    if payload.get('resumed_from'):
                        continue
                    task_id = str(payload['task_id'])
                    key = (task_id, payload['episode_index'])
                    task = tasks.setdefault(
                        task_id, dict(start=None, end=None, episodes=0))
                    if kind == 'episode_start':
                        when = epoch(payload['started_at'])
                        task['start'] = min(task['start'] or when, when)
                        active.add(key)
                    else:
                        when = epoch(payload['finished_at'])
                        task['end'] = max(task['end'] or when, when)
                        task['episodes'] += 1
                        active.discard(key)
    if run_start is not None:
        boundary = min(stopped, max(started, run_start))
        end = min(stopped, max(boundary, run_end or stopped))
        phases.update(startup=(started, boundary), rollout=(boundary, end))
        if run_end is not None:
            phases['teardown'] = (end, stopped)
    for task_id, task in tasks.items():
        task['complete'] = (
            task['start'] is not None
            and not any(k[0] == task_id for k in active))
        task['start'] = max(started, task['start'] or started)
        task['end'] = min(stopped,
                          task['end'] if task['complete'] else stopped)
    return phases, tasks, status


class SampleStats:
    """Bound memory using a histogram of integer utilization percentages."""

    def __init__(self):
        self.samples = 0
        self.utilization = Counter()
        self.memory_samples = 0
        self.memory_peak = self.memory_total = None
        self.first = self.last = None
        self.max_gap = 0
        self.index = self.uuid = None

    def add(self, sample):
        when = float(sample['timestamp'])
        self.samples += 1
        self.index, self.uuid = sample['index'], sample['uuid']
        if self.first is None:
            self.first = when
        if self.last is not None:
            self.max_gap = max(self.max_gap, when - self.last)
        self.last = when
        utilization = numeric(sample['utilization_gpu_pct'])
        if utilization is not None and utilization <= 100:
            self.utilization[round(utilization)] += 1
        used = numeric(sample['memory_used_mib'])
        if used is not None:
            self.memory_samples += 1
            self.memory_peak = max(self.memory_peak or 0, used)
        total = numeric(sample['memory_total_mib'])
        if total is not None:
            self.memory_total = total

    def report(self, start, end):
        n = sum(self.utilization.values())
        p95, cumulative = None, 0
        for value, count in sorted(self.utilization.items()):
            cumulative += count
            if cumulative >= math.ceil(0.95 * n):
                p95 = value
                break
        return dict(
            index=self.index,
            uuid=self.uuid,
            samples=self.samples,
            utilization_samples=n,
            memory_samples=self.memory_samples,
            utilization_mean_pct=(sum(k * v
                                      for k, v in self.utilization.items()) /
                                  n if n else None),
            utilization_p95_pct=p95,
            utilization_max_pct=max(self.utilization) if n else None,
            memory_peak_mib=self.memory_peak,
            memory_total_mib=self.memory_total,
            memory_peak_pct=(100 * self.memory_peak /
                             self.memory_total if self.memory_peak is not None
                             and self.memory_total else None),
            max_sampling_gap_s=max(
                self.max_gap,
                (self.first -
                 start) if self.first is not None else end - start,
                (end - self.last) if self.last is not None else end - start))


def summarize(monitor, journal, evaluation_completed):
    phases, tasks, status = event_windows(journal, monitor.started_at,
                                          monitor.stopped_at)
    windows = [('phase', key, start, end)
               for key, (start, end) in phases.items()]
    windows += [('task', key, task['start'], task['end'])
                for key, task in tasks.items()]
    counters = {
        (scope, key): {
            gpu: SampleStats()
            for gpu in monitor.gpu_ids
        }
        for scope, key, _, _ in windows
    }
    with (monitor.directory / 'gpu_samples.csv').open() as stream:
        for sample in csv.DictReader(stream):
            when = float(sample['timestamp'])
            for scope, key, start, end in windows:
                if start <= when < end:
                    counters[scope, key][sample['gpu_id']].add(sample)
    report = dict(
        schema_version=1,
        evaluation_completed=evaluation_completed,
        rollout_status=status,
        query_errors=monitor.query_errors,
        sample_interval_s=monitor.interval,
        samples_file='gpu_samples.csv',
        measurement='Whole GPU, including inference, rendering and other '
        'processes. Arithmetic mean of periodic nvidia-smi samples; '
        'missing metrics are null. Task windows span first episode_start to '
        'last episode_end and may overlap. They are not per-worker counters.',
        phases={},
        tasks={})
    for scope, key, start, end in windows:
        item = dict(
            start_timestamp=start,
            end_timestamp=end,
            duration_s=end - start,
            gpus={
                gpu: stats.report(start, end)
                for gpu, stats in counters[scope, key].items()
            })
        if scope == 'task':
            item.update(
                completed_episodes=tasks[key]['episodes'],
                window_complete=tasks[key]['complete'])
        report['phases' if scope == 'phase' else 'tasks'][key] = item
    has_samples = any(s['samples']
                      for s in report['phases']['total']['gpus'].values())
    missing = any(
        s['utilization_samples'] != s['samples'] or s['memory_samples'] !=
        s['samples'] or not s['samples'] or s['memory_total_mib'] is None
        for s in report['phases']['total']['gpus'].values())
    report['monitor_status'] = ('unavailable' if not has_samples else 'partial'
                                if missing or monitor.query_errors else 'ok')
    return report
