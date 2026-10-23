"""Evaluate an already-running ZMQ model from a prepared simulator host."""

import argparse
import json
import re
import signal
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from evaluation import resume as resume_utils
from evaluation.config import DEFAULT_CONFIG, load_config
from evaluation.gpu_monitor import GpuMonitor, query_gpus
from evaluation.gpu_monitor import summarize as summarize_gpu
from evaluation.measurements import ProcessResourceMonitor, write_analysis
from evaluation.progress import (Events, Progress, SimulatorStartupLog,
                                 write_json)
from evaluation.runtime import ProcessRunner
from mmengine import DictAction
from preflight.environment import check_layouts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def simulator_resources(local):
    try:
        rows = query_gpus(local['gpus'].split(','))
        return dict(status='ok', gpus=rows)
    except Exception as exc:
        print(f'GPU capacity unavailable: {exc}', flush=True)
        return dict(status='unavailable', error=str(exc))


def probe(address, workers=None):
    from fluxthemis.transports.fluxvla_zmq import query_status
    try:
        status = query_status(address, workers=workers)
        return dict(
            state=status['status'],
            ready=status['workers_ready'],
            expected=status['workers_expected'],
            deployment=status['deployment_id'])
    except TimeoutError:
        return dict(
            state='unreachable',
            message=('No service response yet; '
                     'check remote logs, address and tunnel'))
    except (ValueError, TypeError, RuntimeError) as exc:
        return dict(state='invalid', message=str(exc))


def simulator_arguments(plan):
    local = plan['local']
    command = [
        '--config',
        local['frozen_config'],
        '--server-endpoint',
        plan['endpoint'],
        '--gpu-ids',
        local['gpus'],
        '--workers-per-gpu',
        str(local['workers']),
        '--scheduling',
        local['scheduling'],
        '--run-name',
        plan['run'],
        '--rollout-bind',
        f'tcp://127.0.0.1:{local["port"]}',
        '--rollout-endpoint',
        f'tcp://127.0.0.1:{local["port"]}',
        '--startup-timeout-s',
        str(plan['wait_s']),
        '--cfg-options',
        f'themis.evaluation.reporter.event_log={plan["output"]}/events.jsonl',
        f'themis.transport.connect_timeout_s={plan["wait_s"]}',
    ]
    if local.get('robodojo_root'):
        command.append('themis.evaluation.environment.robodojo_root=' +
                       local['robodojo_root'])
    if local.get('robotwin_root'):
        command.append('themis.evaluation.environment.robotwin_root=' +
                       local['robotwin_root'])
    if local['save_videos']:
        command += ['--save-videos', '--video-dir', f'{plan["output"]}/videos']
    if plan.get('resume'):
        command += ['--resume-from', f'{plan["output"]}/resume_input.json']
    return command


def preflight(plan, runtime, progress):
    runtime.connect()
    progress.show('Python, simulator dependencies and asset paths verified', 1,
                  3)
    output = Path(plan['output'])
    checks = [('config_check.log', [
        'bash',
        str(ROOT / 'scripts/distributed_eval.sh'), *simulator_arguments(plan),
        '--dry-run'
    ]),
              ('layout_check.log', [
                  plan['local']['python'], __file__, '--check-layouts',
                  plan['local']['frozen_config']
              ])]
    for index, (name, command) in enumerate(checks, 2):
        logfile = output / name
        if runtime.run_logged(command, logfile, progress, 120):
            raise RuntimeError(f'Simulator preflight failed; see {logfile}')
        progress.show(f'{name} passed (GPU not initialized yet)', index, 3)
    if plan['local']['environment_type'] == 'RoboTwinEnvironment':
        from preflight.robotwin import check_gpus
        check_gpus(plan['local'], runtime, progress, output)


def wait_remote(plan, runtime, progress):
    deadline = time.monotonic() + plan['wait_s']
    probe_log = Path(plan['output']) / 'probe.log'
    while time.monotonic() < deadline:
        command = [
            plan['local']['python'], __file__, '--probe', plan['endpoint']
        ]
        if plan.get('model_workers'):
            command += ['--expected-workers', str(plan['model_workers'])]
        code = runtime.run_logged(command, probe_log, progress, 20,
                                  lambda: None)
        if code:
            raise RuntimeError(
                f'Connection probe process failed; log: {probe_log}')
        try:
            status = json.loads(probe_log.read_text().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(
                f'Cannot read probe status; log: {probe_log}') from exc
        write_json(Path(plan['output']) / 'remote_status.json', status)
        if status['state'] == 'invalid':
            raise RuntimeError(status['message'])
        if status['state'] == 'unreachable':
            progress.show(status['message'], status='Waiting for connection')
        else:
            plan['deployment'] = status['deployment']
            plan['model_workers'] = status['expected']
            write_json(Path(plan['output']) / 'plan.json', plan)
            progress.show(
                f'Service {status["deployment"]}; loading model replicas',
                status['ready'], status['expected'],
                'Completed' if status['state'] == 'ready' else 'Loading')
        if status['state'] == 'ready':
            if (plan['local']['scheduling'] == 'fixed'
                    and status['expected'] != plan['simulator_workers']):
                raise ValueError('fixed scheduling requires equal model and '
                                 'simulator worker counts')
            return
        time.sleep(3)
    raise TimeoutError(
        'Timed out waiting for model; check remote model.log, IP/port '
        'and network/tunnel')


def evaluate(plan, runtime, progress):
    output = Path(plan['output'])
    journal = output / 'events.jsonl'
    evaluator_log = output / 'simulator.log'
    events = Events()
    monitor = GpuMonitor(output, plan['local']['gpus'].split(','))
    resources = ProcessResourceMonitor(output,
                                       plan['local']['gpus'].split(','))
    startup = SimulatorStartupLog()

    def tick():
        events.read(journal)
        if events.total is None:
            text = evaluator_log.read_text(
                errors='replace') if evaluator_log.exists() else ''
            startup_message = startup.update(text)
            progress.show('Simulator workers ready' + startup_message,
                          len(startup.ready), plan['simulator_workers'],
                          'Initializing')
        else:
            remaining = max(events.total - len(events.episodes), 0)
            eta_s = None if remaining else 0
            starts = [
                datetime.fromisoformat(value).timestamp()
                for key, value in events.episode_starts.items()
                if value and key not in events.restored
            ]
            current_completed = sum(key not in events.restored
                                    for key in events.episodes)
            restored_completed = len(events.episodes) - current_completed
            history_target = plan.get('resume', {}).get('completed', 0)
            replaying = restored_completed < history_target
            if (remaining and not replaying and starts
                    and current_completed >= plan['simulator_workers']):
                elapsed = max(time.time() - min(starts), .001)
                eta_s = remaining * elapsed / current_completed
            progress.show(
                'Episodes',
                len(events.episodes),
                events.total,
                'Finalizing' if events.end else
                ('Restoring' if replaying else 'Evaluating'),
                live=True,
                eta_s=eta_s)

    if plan['local'].get('cache'):
        Path(plan['local']['cache']).mkdir(parents=True, exist_ok=True)
    command = [
        'bash',
        str(ROOT / 'scripts/distributed_eval.sh'), *simulator_arguments(plan)
    ]
    write_json(output / 'simulator_command.json', command)
    result = None
    local_start = None
    try:
        progress.show(
            'Collecting CPU/RAM/GPU baseline before execution (~3 seconds)')
        resources.baseline()
        local_start = time.time()
        monitor.start()
        code = runtime.run_logged(
            command,
            evaluator_log,
            progress,
            plan['limit_s'],
            tick,
            on_start=resources.start)
        result = events.result(code)
    except BaseException as exc:
        result = events.result(-1)
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        progress.close()
        resources.stop()
        monitor.stop()
        if result is not None:
            if plan.get('resume'):
                result['resume'] = plan['resume']
            if evaluator_log.is_file():
                log_text = evaluator_log.read_text(errors='replace')
                result['simulator_restart_count'] = len(
                    re.findall(
                        r'\[FluxThemis\] restarting simulator-worker-\S+'
                        r' after exit code', log_text))
            report = {}
            try:
                report = summarize_gpu(monitor, journal, result['complete'])
                write_json(output / 'gpu_summary.json', report)
                result['gpu_monitor_status'] = report['monitor_status']
            except Exception as exc:
                result['gpu_monitor_status'] = 'unavailable'
                result['gpu_monitor_error'] = str(exc)
            try:
                result['performance_report'] = write_analysis(
                    plan, result, events.episode_starts, resources, report,
                    local_start)
            except Exception as exc:
                result['performance_report_error'] = str(exc)
                print(
                    f'Performance summary failed; raw samples retained: {exc}',
                    flush=True)
            write_json(output / 'result.json', result)
    return result


def run_plan(plan, progress):
    output = Path(plan['output'])
    progress.enter(2)
    runtime = ProcessRunner(plan)
    preflight(plan, runtime, progress)
    if plan.get('resume'):
        resume_utils.verify_environment(plan, runtime)
    else:
        progress.show(
            'Recording config, environment and asset fingerprints for resume')
        resume_utils.freeze_inputs(plan, runtime)
    write_json(output / 'resources.json', simulator_resources(plan['local']))
    progress.enter(3)
    progress.show(f'Connecting to {plan["endpoint"]}; '
                  'model service is started separately')
    wait_remote(plan, runtime, progress)
    resume_utils.verify_model(plan)
    write_json(output / 'plan.json', plan)
    progress.enter(4)
    result = evaluate(plan, runtime, progress)
    progress.enter(5)
    if not result['complete']:
        raise RuntimeError(
            f'Evaluation incomplete: {result["completed"]}/{result["total"]}; '
            f'exit code {result["simulator_exit_code"]}; '
            'see simulator.log / result.json')
    progress.show(
        f'Evaluation complete; successful episodes '
        f'{result["successes"]}/{result["total"]}', result['completed'],
        result['total'], 'Completed')
    print(f'Simulator results: {output / "result.json"}\n'
          f'Model service results: {result["remote_result_dir"]}\n'
          f'Resources and throughput: {output / "performance_report.txt"}')
    return 0


def run_config(config,
               dry_run=False,
               resume=None,
               result_pointer=None,
               mode=None,
               overrides=None):
    progress = Progress()
    source_lock = run_lock = None
    try:
        progress.enter(1)
        settings = load_config(config, mode=mode, overrides=overrides)
        configured_source = settings['resume']['from_dir']
        if resume is not None:
            resume = Path(resume).expanduser().resolve()
            if configured_source and resume != Path(configured_source):
                raise ValueError(
                    '--resume does not match YAML resume.from_dir')
        source = resume or configured_source
        if source and not dry_run:
            plan, source_lock = resume_utils.prepare_resume(source, settings)
            if plan is None:
                if result_pointer:
                    write_json(
                        Path(result_pointer), dict(directory=str(source)))
                return 0
        else:
            name = 'evaluation_' + time.strftime(
                '%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
            if dry_run:
                name += '_dry_run'
            output = Path(settings['output_root']) / name
            output.mkdir(parents=True)
            plan = dict(
                schema_version=4,
                run=name,
                output=str(output),
                local=settings['local'],
                simulator_workers=settings['simulator_workers'])
            snapshot = output / 'config_snapshot.py'
            settings['config'].dump(str(snapshot))
            plan['local']['frozen_config'] = str(snapshot)
        plan.update(
            endpoint=settings['endpoint'],
            wait_s=settings['wait_s'],
            limit_s=settings['limit_s'],
            launch_config=settings['source'],
            launch_mode=mode or 'config',
            model_result_dir=settings['model']['result_dir'])
        plan.pop('deployment', None)
        plan.pop('model_workers', None)
        if settings['model']['expected_workers'] is not None:
            plan['model_workers'] = settings['model']['expected_workers']
        output = Path(plan['output'])
        progress.output = output
        if result_pointer:
            write_json(Path(result_pointer), dict(directory=str(output)))
        suffix = Path(settings['source']).suffix
        (output / ('launch_config' + suffix)).write_text(
            settings['source_text'])
        write_json(output / 'plan.json', plan)
        print(f'Results: {output}', flush=True)
        print(
            f'Model endpoint: {plan["endpoint"]}; '
            f'simulator workers: {plan["simulator_workers"]}',
            flush=True)
        if dry_run:
            print(
                'Configuration resolved; no network, GPU, asset hashing '
                'or subprocess work. '
                'Use a smoke run to validate the prepared environment.',
                flush=True)
            if source:
                print(
                    f'Resume source to validate on execution: {source}',
                    flush=True)
            return 0
        run_lock = resume_utils.RunLock(output)
        return run_plan(plan, progress)
    except KeyboardInterrupt:
        progress.show(
            'Evaluation interrupted; cleaning up simulator processes',
            status='Cancelled')
        return 130
    except Exception as exc:
        progress.show(f'{type(exc).__name__}: {exc}', status='Failed')
        if progress.output:
            print(f'Run logs: {progress.output}')
        return 1
    finally:
        progress.close()
        if run_lock:
            run_lock.close()
        if source_lock:
            source_lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        type=Path,
        nargs='+',
        default=[DEFAULT_CONFIG],
        help='Launch YAML or Themis Python configs; evaluated sequentially.')
    parser.add_argument(
        '--server-endpoint',
        help='Reachable model ZMQ endpoint; never starts a model.')
    parser.add_argument('--gpu-ids', help='Simulator GPU IDs or UUIDs.')
    parser.add_argument(
        '--gpus', type=int, help='Simulator GPU count (default: 1).')
    parser.add_argument(
        '--workers-per-gpu',
        type=int,
        help='Simulator workers per GPU (default: 1).')
    parser.add_argument('--scheduling', choices=['dynamic', 'fixed'])
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument(
        '--output', type=Path, help='Simulator output parent directory.')
    parser.add_argument('--save-videos', action='store_true', default=None)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        '--smoke',
        dest='mode',
        action='store_const',
        const='smoke',
        help='One task, one episode per simulator worker.')
    scope.add_argument(
        '--full',
        dest='mode',
        action='store_const',
        const='full',
        help='All tasks using the Python evaluation config quotas.')
    parser.add_argument(
        '--resume', type=Path, help='Resume a stopped output directory.')
    parser.add_argument(
        '--result-pointer',
        type=Path,
        help='Write this evaluation output location.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Resolve config without starting workloads or connecting.')
    parser.add_argument('--probe', help=argparse.SUPPRESS)
    parser.add_argument('--expected-workers', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--check-layouts', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.check_layouts:
        check_layouts(args.check_layouts)
        return 0
    if args.probe:
        print(json.dumps(probe(args.probe, args.expected_workers)))
        return 0
    if len(args.config) > 1 and (args.resume or args.result_pointer):
        parser.error('--resume and --result-pointer require one config')
    overrides = {
        key: getattr(args, key)
        for key in ('server_endpoint', 'gpu_ids', 'gpus', 'workers_per_gpu',
                    'scheduling', 'cfg_options', 'output', 'save_videos')
    }
    for config in args.config:
        code = run_config(config, args.dry_run, args.resume,
                          args.result_pointer, args.mode, overrides)
        if code:
            return code
    return 0


if __name__ == '__main__':

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
