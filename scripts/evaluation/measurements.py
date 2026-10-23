"""Read-only simulator resource and inference accounting."""

import csv
import json
import math
import os
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

from .gpu_monitor import query_gpus
from .runtime import ProcessTree


def stats(values):
    values = sorted(
        v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    return dict(
        samples=len(values),
        mean=statistics.mean(values) if values else None,
        peak=max(values) if values else None,
        p95=values[math.ceil(.95 * len(values)) - 1] if values else None)


def host_snapshot():
    lines = Path('/proc/stat').read_text().splitlines()
    ticks = [int(x) for x in lines[0].split()[1:9]]
    mem = {
        line.split(':')[0]: int(line.split()[1])
        for line in Path('/proc/meminfo').read_text().splitlines()
    }
    return dict(
        timestamp=time.time(),
        cpu_total_ticks=sum(ticks),
        cpu_idle_ticks=ticks[3] + ticks[4],
        logical_cpus=sum(line.startswith('cpu') for line in lines) - 1,
        memory_total_mib=mem['MemTotal'] / 1024,
        memory_used_mib=(mem['MemTotal'] - mem['MemAvailable']) / 1024)


def process_snapshot(pid):
    root = Path('/proc') / str(pid)
    # comm may contain spaces and parentheses; fields start after the last ')'.
    fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    status = dict(
        line.split(':', 1)
        for line in (root / 'status').read_text().splitlines())
    return dict(
        pid=int(pid),
        container_pid=int(status.get('NSpid', str(pid)).split()[-1]),
        start_ticks=int(fields[19]),
        cpu_seconds=(int(fields[11]) + int(fields[12])) /
        os.sysconf('SC_CLK_TCK'),
        rss_mib=int(fields[21]) * os.sysconf('SC_PAGE_SIZE') / 1024**2,
        threads=int(fields[17]))


def compute_memory():
    result = subprocess.run([
        'nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
        '--format=csv,noheader,nounits'
    ],
                            check=True,
                            capture_output=True,
                            text=True,
                            timeout=3)
    rows = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 3:
            continue
        pid, gpu, used = (v.strip() for v in row)
        try:
            rows.append(
                dict(pid=int(pid), gpu_uuid=gpu, memory_mib=float(used)))
        except ValueError:
            continue
    return rows


class ProcessResourceMonitor:
    """Sample the simulator process tree, including torchrun child sessions."""

    sample_file = 'process_samples.jsonl'

    def __init__(self, output, gpu_ids, interval=3):
        self.output = Path(output)
        self.gpu_ids, self.interval = gpu_ids, interval
        self.baseline_rows, self.rows = [], []
        self._stop = threading.Event()
        self._thread = None
        self._tree = None
        self._process_time = time.monotonic()
        self.previous_host = None
        self.previous_processes = {}

    def _host(self):
        row = host_snapshot()
        old = self.previous_host
        if old and row['cpu_total_ticks'] > old['cpu_total_ticks']:
            row['cpu_busy_pct'] = 100 * (
                1 - (row['cpu_idle_ticks'] - old['cpu_idle_ticks']) /
                (row['cpu_total_ticks'] - old['cpu_total_ticks']))
        self.previous_host = row
        return row

    def baseline(self):
        for index in range(3):
            row = dict(host=self._host())
            try:
                row['gpus'] = query_gpus(self.gpu_ids)
            except Exception as exc:
                row['gpu_error'] = str(exc)
            self.baseline_rows.append(row)
            if index < 2:
                time.sleep(1)
        (self.output / 'resource_baseline.json'
         ).write_text(json.dumps(self.baseline_rows, indent=2) + '\n')

    def start(self, name):
        self._tree = ProcessTree(name)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _process_ids(self):
        return list(self._tree.members()) if self._tree else []

    def _processes(self):
        now = time.monotonic()
        old_time = self._process_time
        rows = []
        old_rows, self.previous_processes = self.previous_processes, {}
        for pid in self._process_ids():
            try:
                row = process_snapshot(pid)
            except (OSError, ValueError, IndexError):
                continue
            key = (row['pid'], row['start_ticks'])
            old = old_rows.get(key)
            if old and now > old_time:
                row['cpu_core_equivalents'] = max(
                    0, (row['cpu_seconds'] - old['cpu_seconds']) /
                    (now - old_time))
            self.previous_processes[key] = row
            row['pss_mib'] = None
            try:
                for line in Path(
                        f'/proc/{pid}/smaps_rollup').read_text().splitlines():
                    if line.startswith('Pss:'):
                        row['pss_mib'] = int(line.split()[1]) / 1024
                        break
            except (OSError, ValueError):
                pass
            rows.append(row)
        self._process_time = now
        return rows

    def _readers(self):
        return [('processes', self._processes),
                ('gpu_processes', compute_memory)]

    def _run(self):
        with (self.output / self.sample_file).open('w') as stream:
            while not self._stop.is_set():
                started = time.monotonic()
                row = dict(timestamp=time.time(), host=self._host())
                errors = {}
                for key, read in self._readers():
                    try:
                        row[key] = read()
                    except Exception as exc:
                        errors[key] = str(exc)
                pids = {p['pid'] for p in row.get('processes', [])}
                row['gpu_processes'] = [
                    p for p in row.get('gpu_processes', []) if p['pid'] in pids
                ]
                if errors:
                    row['errors'] = errors
                self.rows.append(row)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                self._stop.wait(
                    max(0, self.interval - (time.monotonic() - started)))

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=16)

    @staticmethod
    def total(row, key):
        processes = row.get('processes', [])
        values = [p.get(key) for p in processes]
        return sum(values) if values and all(v is not None
                                             for v in values) else None

    def summary(self, gpu_report, simulator_log, rollout_window=None):
        baseline = dict(
            host_memory_mib=statistics.median(r['host']['memory_used_mib']
                                              for r in self.baseline_rows),
            host_cpu_busy_pct=stats(
                [r['host'].get('cpu_busy_pct') for r in self.baseline_rows]),
            logical_cpus=self.baseline_rows[0]['host']['logical_cpus'])
        total_gpu = gpu_report.get('phases', {}).get('total',
                                                     {}).get('gpus', {})
        gpu = {}
        for device in self.gpu_ids:
            values = [
                g['memory_used_mib'] for r in self.baseline_rows
                for g in r.get('gpus', [])
                if g['gpu_id'] == device and g['memory_used_mib'] is not None
            ]
            base = statistics.median(values) if values else None
            peak = total_gpu.get(device, {}).get('memory_peak_mib')
            gpu[device] = dict(
                baseline_mib=base,
                whole_gpu_peak_mib=peak,
                incremental_peak_mib=peak -
                base if peak is not None and base is not None else None)
        names = {
            int(pid): worker
            for worker, pid in re.findall(
                r'simulator process id=(\S+) pid=(\d+)', simulator_log)
        }
        process_groups = {}
        for row in self.rows:
            for p in row.get('processes', []):
                key = (p['pid'], p['start_ticks'])
                group = process_groups.setdefault(
                    key,
                    dict(
                        pid=p['pid'],
                        container_pid=p['container_pid'],
                        worker_id=names.get(p['container_pid']),
                        cpu=[],
                        rss=[],
                        gpu=[]))
                group['cpu'].append(p.get('cpu_core_equivalents'))
                group['rss'].append(p['rss_mib'])
                group['gpu'].extend(g['memory_mib']
                                    for g in row.get('gpu_processes', [])
                                    if g['pid'] == p['pid'])
        processes = [
            dict(
                pid=g['pid'],
                container_pid=g['container_pid'],
                worker_id=g['worker_id'],
                cpu_core_equivalents=stats(g['cpu']),
                rss_mib=stats(g['rss']),
                gpu_memory_mib=stats(g['gpu']))
            for g in process_groups.values()
        ]
        windows = dict(total=self.rows)
        if rollout_window:
            start, end = rollout_window
            windows['episode_execution'] = [
                r for r in self.rows if start <= r['timestamp'] <= end
            ]
        phases = {}
        for name, rows in windows.items():
            phases[name] = dict(
                host_cpu_busy_pct=stats(
                    [r['host'].get('cpu_busy_pct') for r in rows]),
                host_memory_mib=stats(
                    [r['host']['memory_used_mib'] for r in rows]),
                **{
                    'process_tree_' + key:
                    stats([self.total(r, key) for r in rows])
                    for key in ('cpu_core_equivalents', 'pss_mib', 'rss_mib')
                })
        peak = phases['total']['host_memory_mib']['peak']
        return dict(
            runtime='native',
            baseline=baseline,
            gpus=gpu,
            phases=phases,
            processes=processes,
            host_memory_incremental_peak_mib=(peak -
                                              baseline['host_memory_mib']
                                              if peak is not None else None),
            samples=len(self.rows),
            samples_with_errors=sum(bool(r.get('errors')) for r in self.rows),
            measurement=(
                'CPU from process tick deltas; 100% is one logical core. '
                'PSS apportions shared memory; missing samples stay '
                'unavailable. RSS sums double-count shared pages. '
                'GPU PID namespaces may differ; unmatched GPU memory '
                'is unavailable. Whole GPU/host deltas include other '
                'processes; sampling may miss short peaks.'))


def read_jsonl(path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines(keepends=True):
        if line.endswith('\n'):
            rows.append(json.loads(line))
    return rows


def model_measurements(plan, remote_result_dir, window):
    """Match actual remote output directories, since IDs can be reused."""
    found = []
    sources = []
    if remote_result_dir and plan.get('model_result_dir'):
        paths = [Path(plan['model_result_dir']) / 'model_metrics.jsonl']
        for path in paths:
            matching = [
                r for r in read_jsonl(path)
                if r.get('deployment_id') == plan['deployment']
                and r.get('output')
                and Path(remote_result_dir).is_relative_to(Path(r['output']))
            ]
            if matching:
                found.extend(matching)
                sources.append(str(path))
    if window:
        found = [
            r for r in found if isinstance(
                r.get('received_at', r.get('timestamp')), (int, float)) and
            window[0] <= r.get('received_at', r.get('timestamp')) <= window[1]
        ]
    devices = sorted({g['index'] for r in found for g in r.get('gpus', [])})
    return dict(
        sources=sources,
        samples=len(found),
        gpus={
            device:
            dict(
                utilization_pct=stats([
                    g.get('utilization_pct') for r in found
                    for g in r.get('gpus', []) if g['index'] == device
                ]),
                memory_mib=stats([
                    g.get('memory_mib') for r in found
                    for g in r.get('gpus', []) if g['index'] == device
                ]))
            for device in devices
        },
        pending_requests=stats(
            [r.get('status', {}).get('pending_requests') for r in found]),
        samples_with_errors=sum(bool(r.get('errors')) for r in found),
        measurement=(
            ('Shared-filesystem samples use the model host timestamp; '
             'host clocks must be synchronized. '
             if plan.get('model_result_dir') else
             'Remote whole-GPU samples use local SSH receipt time. ') +
            'Queue length is sampled; short queues may be missed. '
            'It does not measure per-request queue latency.'))


def performance_summary(episodes,
                        starts,
                        complete,
                        model_workers,
                        local_start=None,
                        total_episodes=None):
    """Wall-clock throughput; summed times describe worker occupancy."""
    from datetime import datetime

    def timestamp(value):
        return datetime.fromisoformat(value).timestamp() if value else None

    begin = [timestamp(starts.get(key)) for key in episodes]
    end = [timestamp(row.get('finished_at')) for row in episodes.values()]
    valid_window = bool(episodes) and all(v is not None for v in begin + end)
    window = (min(begin), max(end)) if valid_window else None
    wall = window[1] - window[0] if window else None
    durations = [r.get('duration_s') for r in episodes.values()]
    durations_complete = all(
        isinstance(v, (int, float)) and v > 0 for v in durations)
    worker_seconds = sum(
        durations) if durations and durations_complete else None
    profiles = [
        p.get('profile', {}) for r in episodes.values()
        for p in r.get('prediction_metadata', [])
    ]
    calls = sum(r.get('model_calls', 0) for r in episodes.values())
    all_profiles = calls > 0 and calls == len(profiles) and all(
        isinstance(p.get(k), (int,
                              float)) and math.isfinite(p[k]) and p[k] >= 0
        for p in profiles
        for k in ('roundtrip_s', 'server_infer_s', 'network_and_queue_s'))
    latency = {
        k: stats([p.get(k) for p in profiles])
        for k in ('roundtrip_s', 'server_infer_s', 'network_and_queue_s')
    }
    predict_seconds = sum(p['roundtrip_s']
                          for p in profiles) if all_profiles else None
    infer_seconds = sum(p['server_infer_s']
                        for p in profiles) if all_profiles else None
    rate = len(
        episodes) * 3600 / wall if complete and wall and wall > 0 else None
    startup = window[0] - local_start if window and local_start else None
    return dict(
        episode_window=window,
        episode_execution_wall_s=wall,
        completed_episodes=len(episodes),
        episodes_per_hour=rate,
        startup_to_first_episode_s=startup,
        summed_episode_execution_s=worker_seconds,
        summed_predict_wait_s=predict_seconds,
        summed_model_infer_s=infer_seconds,
        simulator_predict_wait_fraction=(
            predict_seconds / worker_seconds
            if worker_seconds and predict_seconds is not None else None),
        model_worker_busy_fraction=(infer_seconds /
                                    (wall * model_workers) if wall and wall > 0
                                    and infer_seconds is not None else None),
        inference_calls=calls,
        profile_coverage_complete=all_profiles,
        latency_s=latency,
        target_episodes=total_episodes,
        remaining_hours=(max(total_episodes - len(episodes), 0) / rate
                         if rate and total_episodes is not None else None),
        projection_basis=('Extrapolation assumes equal throughput '
                          'for all tasks. Excludes startup and unknown task '
                          'switch/retry costs. Not a full-benchmark ETA.'),
        measurement=('Throughput uses first episode_start to last episode_end '
                     'wall time, including inter-episode gaps. Wait fraction '
                     'uses summed episode duration. Model busy fraction uses '
                     'inference time / (wall time * model workers).'))


def write_analysis(plan, result, starts, resource_monitor, gpu_report,
                   local_start):
    # Load actual episode bodies rather than result.metrics averages.
    entries = read_jsonl(Path(plan['output']) / 'events.jsonl')
    episodes = {
        e['payload']['episode_id']: e['payload']
        for e in entries if e['event_type'] == 'episode_end'
    }
    current = {
        key: row
        for key, row in episodes.items() if not row.get('resumed_from')
    }
    perf = performance_summary(
        current, starts, result['complete'], plan['model_workers'],
        local_start,
        (result.get('total') - plan.get('resume', {}).get('completed', 0)
         if result.get('total') is not None else None))
    history = plan.get('resume', {})
    cumulative_wall = (
        history.get('elapsed_execution_s', 0.) +
        (perf['episode_execution_wall_s'] or 0.))
    perf.update(
        restored_episodes=len(episodes) - len(current),
        total_completed_episodes=len(episodes),
        cumulative_execution_s=cumulative_wall,
        cumulative_episodes_per_hour=(len(episodes) * 3600 /
                                      cumulative_wall if cumulative_wall > 0
                                      and result['complete'] else None))
    remote = model_measurements(plan, result.get('remote_result_dir'),
                                perf['episode_window'])
    log = Path(plan['output']) / 'simulator.log'
    resources = resource_monitor.summary(
        gpu_report,
        log.read_text(errors='replace') if log.is_file() else '',
        perf['episode_window'])
    report = dict(
        performance=perf, local_resources=resources, remote_model=remote)
    path = Path(plan['output']) / 'performance_report.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    def show(value, scale=1):
        return '--' if value is None else f'{value / scale:.2f}'

    def describe(row, unit='', scale=1):
        return (f'mean {show(row["mean"], scale)}, '
                f'P95 {show(row["p95"], scale)}, '
                f'peak {show(row["peak"], scale)} {unit}')

    lines = [
        'Evaluation resources and throughput report',
        f'Complete: {result["complete"]}; episodes: {len(episodes)}; '
        f'simulator workers: {plan["simulator_workers"]}; '
        f'model replicas: {plan["model_workers"]}',
        f'Restored episodes: {perf["restored_episodes"]}; '
        f'executed this run: {len(current)}; '
        'resource, inference and throughput figures below cover '
        'the current execution window.',
        'Initialization to first episode start: '
        f'{show(perf["startup_to_first_episode_s"])} seconds; '
        f'First episode start to last episode end: '
        f'{show(perf["episode_execution_wall_s"])} seconds',
        f'Throughput: {show(perf["episodes_per_hour"])} ep/h',
        'Cumulative execution time across resumes: '
        f'{show(cumulative_wall)} seconds; '
        'cumulative throughput: '
        f'{show(perf["cumulative_episodes_per_hour"])} ep/h '
        '(excluding downtime)',
        'Full benchmark duration requires the complete execution window; '
        'throughput for a subset of tasks does not represent other tasks.',
    ]
    for device, g in resources['gpus'].items():
        lines.append(f'GPU {device}: baseline '
                     f'{show(g["baseline_mib"], 1024)} GiB, whole-GPU peak '
                     f'{show(g["whole_gpu_peak_mib"], 1024)} GiB, increment '
                     f'{show(g["incremental_peak_mib"], 1024)} GiB')
    phase = resources['phases']['total']
    cpu_key = 'process_tree_cpu_core_equivalents'
    memory_key = 'process_tree_pss_mib'
    scope = 'Simulator process tree'
    memory_label = 'PSS memory'
    lines += [
        scope + ' CPU (logical core equivalents): ' + describe(phase[cpu_key]),
        scope + ' ' + memory_label + ': ' +
        describe(phase[memory_key], 'GiB', 1024),
        f'Host memory baseline: '
        f'{show(resources["baseline"]["host_memory_mib"], 1024)} GiB; '
        f'Host memory incremental peak: '
        f'{show(resources["host_memory_incremental_peak_mib"], 1024)} GiB',
    ]
    execution = resources['phases'].get('episode_execution')
    if execution:
        stage_scope = 'process tree'
        lines += [
            'Evaluation phase ' + stage_scope +
            ' CPU (logical core equivalents): ' + describe(execution[cpu_key]),
            'Evaluation phase ' + stage_scope + ' ' + memory_label + ': ' +
            describe(execution[memory_key], 'GiB', 1024),
        ]
    lines.append(f'Local resource samples: {resources["samples"]}; '
                 f'samples with errors: {resources["samples_with_errors"]}')
    for worker in resources['processes']:
        if worker['worker_id']:
            lines.append(f'{worker["worker_id"]} PID={worker["pid"]}: CPU ' +
                         describe(worker['cpu_core_equivalents'], 'cores') +
                         '; RSS ' + describe(worker['rss_mib'], 'GiB', 1024) +
                         '; GPU ' +
                         describe(worker['gpu_memory_mib'], 'GiB', 1024))
    lines += [
        'CPU 100% = one logical core; RSS includes shared pages and cannot '
        'be summed directly. Process GPU memory uses NVML compute-app '
        'accounting and may exclude separate graphics allocations.',
        f'Inference wait fraction of simulator execution time: '
        f'{show(perf["simulator_predict_wait_fraction"], .01)}%; '
        f'Model worker service time fraction: '
        f'{show(perf["model_worker_busy_fraction"], .01)}%',
    ]
    for key, latency in perf['latency_s'].items():
        lines.append(key + ': ' + describe(latency, 'ms', .001))
    for device, gpu in remote['gpus'].items():
        lines.append(f'Remote GPU {device} utilization: ' +
                     describe(gpu['utilization_pct'], '%'))
    lines += [
        'Sampled remote queue length: ' + describe(remote['pending_requests']),
        'Queue sampling may miss short waits; network_and_queue includes '
        'network, serialization and queueing, so it does not measure '
        'queueing time alone.'
    ]
    if not remote['samples']:
        lines.append('Remote GPU data unavailable; cannot assess GPU '
                     'inference bottlenecks.')
    text = '\n'.join(lines) + '\n'
    (Path(plan['output']) / 'performance_report.txt').write_text(text)
    return str(path)
