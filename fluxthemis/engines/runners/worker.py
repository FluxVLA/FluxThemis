"""Restartable simulator worker and its local process supervisor."""

from __future__ import annotations

import multiprocessing
import os
import signal
import time
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from fluxthemis.engines.utils import (build_environment_from_cfg,
                                      build_policy_client_from_cfg,
                                      to_jsonable)
from fluxthemis.environments.base import (BaseEnvironment,
                                          EnvironmentCaseRejected)
from fluxthemis.model_clients.base import BasePolicyClient
from fluxthemis.structures import TaskSpec

from ...transports.zmq_protocol import (decode_frame, decode_header,
                                        empty_payload, encode_frame,
                                        encode_header)
from .assignment import EpisodeAssignment, ExecutionPolicy
from .config import (distributed_video_settings,
                     load_distributed_evaluation_config)
from .episode import EpisodeJob, configure_worker_lifetime, execute_assignment

_SIMULATOR_STARTUP_TIMEOUT_S = 180.0


class SimulatorWorkerSupervisor:
    """Restart one disposable simulator worker process when it exits."""

    def __init__(
            self,
            config_path: str,
            rollout_endpoint: str,
            worker_id: str,
            deployment_id: str,
            device: str,
            episode_timeout_s: float,
            config_overrides: Mapping[str, Any] | None = None,
            startup_timeout_s: float = _SIMULATOR_STARTUP_TIMEOUT_S) -> None:
        self.config_path = config_path
        self.rollout_endpoint = _endpoint(rollout_endpoint, 'rollout_endpoint')
        self.worker_id = _nonempty(worker_id, 'worker_id')
        self.deployment_id = _nonempty(deployment_id, 'deployment_id')
        self.device = _nonempty(device, 'device')
        self.episode_timeout_s = _positive_float(episode_timeout_s,
                                                 'episode_timeout_s')
        self.startup_timeout_s = _positive_float(startup_timeout_s,
                                                 'startup_timeout_s')
        self.config_overrides = dict(config_overrides or {})
        self._context = multiprocessing.get_context('spawn')

    def run(self) -> None:
        import zmq

        os.environ['CUDA_VISIBLE_DEVICES'] = self.device
        os.environ.pop('LOCAL_RANK', None)
        os.environ['MUJOCO_EGL_DEVICE_ID'] = self.device
        with zmq.Context() as context, context.socket(zmq.DEALER) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.IDENTITY,
                              f'{self.worker_id}:supervisor'.encode())
            socket.connect(self.rollout_endpoint)
            self._run_generations(socket)

    def _run_generations(self, socket: Any) -> None:
        generation = 0
        while True:
            ready_event = self._context.Event()
            # Each child needs its own flag; a previous generation's shutdown
            # acknowledgement must not classify a later exit as intentional.
            stop_received = self._context.Event()
            process = self._context.Process(
                target=_run_simulator_worker,
                kwargs={
                    'config_path': self.config_path,
                    'rollout_endpoint': self.rollout_endpoint,
                    'worker_id': self.worker_id,
                    'deployment_id': self.deployment_id,
                    'generation': generation,
                    'device': self.device,
                    'episode_timeout_s': self.episode_timeout_s,
                    'config_overrides': self.config_overrides,
                    'parent_pid': os.getpid(),
                    'ready_event': ready_event,
                    'stop_received': stop_received,
                },
                name=self.worker_id,
            )
            process.start()
            startup_timed_out = False
            try:
                deadline = time.monotonic() + self.startup_timeout_s
                while process.is_alive() and not ready_event.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    ready_event.wait(timeout=min(1.0, remaining))
                if process.is_alive() and not ready_event.is_set():
                    print(
                        f'[FluxThemis] {self.worker_id} did not become ready '
                        f'within {self.startup_timeout_s:g}s; '
                        'terminating',
                        flush=True,
                    )
                    startup_timed_out = True
                    _terminate_process(process)
                process.join()
            except BaseException:
                _terminate_process(process)
                raise
            # Kit can terminate Python with code 0 before the job is done.
            # Only a coordinator shutdown makes that a normal pool exit.
            if (process.exitcode == 0 and not startup_timed_out
                    and stop_received.is_set()):
                return
            restart_reason = ('startup timeout' if startup_timed_out else
                              f'exit code {process.exitcode}')
            self._request_restart(socket, generation, restart_reason)
            generation += 1
            print(
                f'[FluxThemis] restarting {self.worker_id} after '
                f'{restart_reason} as generation {generation}',
                flush=True,
            )
            time.sleep(0.5)

    def _request_restart(self, socket: Any, generation: int,
                         reason: str) -> None:
        socket.send_multipart([
            encode_header(
                'worker_exit',
                deployment_id=self.deployment_id,
                worker_id=self.worker_id,
                generation=generation,
                device=self.device,
                reason=reason),
            empty_payload(),
        ])
        # The coordinator must finish failure reporting before shutting down
        # the rank; raising immediately could kill its reporter in mid-write.
        if not socket.poll(int(_SIMULATOR_STARTUP_TIMEOUT_S * 1000)):
            raise TimeoutError('Coordinator did not decide simulator recovery')
        frames = socket.recv_multipart()
        header = decode_header(frames[0])
        if (header.get('deployment_id') != self.deployment_id
                or header.get('worker_id') != self.worker_id
                or header.get('generation') != generation):
            raise RuntimeError('Simulator recovery response identity mismatch')
        if header['type'] == 'shutdown_worker':
            raise RuntimeError(
                f'Simulator {self.worker_id} stopped after {reason}')
        if header['type'] != 'restart_worker':
            raise RuntimeError('Unexpected simulator recovery response')


class _SimulatorWorker:

    def __init__(self, *, config_path: str, rollout_endpoint: str,
                 worker_id: str, deployment_id: str, generation: int,
                 device: str, episode_timeout_s: float,
                 config_overrides: Mapping[str, Any], ready_event: Any,
                 stop_received: Any) -> None:
        self.config_path = config_path
        self.rollout_endpoint = rollout_endpoint
        self.worker_id = worker_id
        self.deployment_id = deployment_id
        self.generation = generation
        self.device = device
        self.episode_timeout_s = episode_timeout_s
        self.config_overrides = config_overrides
        self.ready_event = ready_event
        self.stop_received = stop_received

    @property
    def _message_identity(self) -> dict[str, Any]:
        return {
            'deployment_id': self.deployment_id,
            'worker_id': self.worker_id,
            'generation': self.generation,
        }

    def run(self) -> None:
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(
            zmq.IDENTITY,
            f'{self.worker_id}:{self.generation}'.encode(),
        )
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.rollout_endpoint)
        environment: BaseEnvironment | None = None
        policy_clients: list[BasePolicyClient] = []
        # Save stderr before simulator redirection, so Python failures reach
        # the supervisor's log before os._exit bypasses normal cleanup.
        error_fd = os.dup(2)
        try:
            self._register(socket)
            evaluation_cfg = load_distributed_evaluation_config(
                self.config_path, config_overrides=self.config_overrides)
            save_videos, video_dir, video_fps = distributed_video_settings(
                evaluation_cfg)
            environment = build_environment_from_cfg(
                evaluation_cfg['environment'])
            policy = ExecutionPolicy(
                evaluation_cfg.get('episode_batch_size', 1))
            policy.validate_options(
                save_videos=save_videos,
                strict_worker_assignment=evaluation_cfg.get(
                    'strict_worker_assignment', False))
            if environment.num_envs != policy.batch_size:
                raise ValueError(
                    'Environment num_envs differs from episode_batch_size')
            environment.configure_worker(self.worker_id, self.deployment_id)
            environment.configure_simulator_device(self.device)
            environment.configure_video_recording(save_videos, video_dir)
            for _ in range(policy.batch_size):
                client = build_policy_client_from_cfg(
                    evaluation_cfg['policy_client'])
                policy_clients.append(client)
                client.setup()
            environment.setup()
            tasks = tuple(
                TaskSpec.from_value(task) for task in environment.tasks())
            socket.send_multipart([
                encode_header('ready', **self._message_identity),
                encode_frame({
                    'tasks': [to_jsonable(task) for task in tasks],
                    'capacity': environment.num_envs
                }),
            ])
            self.ready_event.set()
            while True:
                assignment = self._receive_assignment(socket)
                if assignment is None:
                    # Kit cleanup can terminate Python before run() returns.
                    # A zero exit is normal only after this protocol message.
                    self.stop_received.set()
                    return
                self._execute_assignment(
                    socket=socket,
                    environment=environment,
                    policy_clients=policy_clients,
                    evaluation_cfg=evaluation_cfg,
                    assignment=assignment,
                    save_videos=save_videos,
                    video_dir=video_dir,
                    video_fps=video_fps,
                )
        except BaseException:
            try:
                # Closing the stream flushes the traceback before os._exit;
                # normal interpreter cleanup may hang in native simulator code.
                with os.fdopen(error_fd, 'w', closefd=False) as error_log:
                    print(
                        f'[FluxThemis] simulator worker failed '
                        f'id={self.worker_id} generation={self.generation}',
                        file=error_log)
                    traceback.print_exc(file=error_log)
            finally:
                os._exit(1)
        finally:
            for component in (environment, *policy_clients):
                if component is None:
                    continue
                try:
                    component.close()
                except Exception:
                    pass
            socket.close(linger=0)
            context.term()
            os.close(error_fd)

    def _register(self, socket: Any) -> None:
        socket.send_multipart([
            encode_header(
                'register_worker',
                **self._message_identity,
                device=self.device,
            ),
            empty_payload(),
        ])

    def _receive_assignment(self, socket: Any) -> EpisodeAssignment | None:
        frames = socket.recv_multipart()
        header = decode_header(frames[0])
        self._validate_header(header)
        if header['type'] == 'shutdown_worker':
            return None
        if header['type'] != 'episode_assignment':
            raise RuntimeError(
                f'Unexpected simulator command {header["type"]!r}')
        payload = frames[1] if len(frames) > 1 else empty_payload()
        body = decode_frame(payload)
        if not isinstance(body, Mapping) or not isinstance(
                body.get('jobs'), list):
            raise TypeError('Assignment payload must contain a jobs list')
        assignment = EpisodeAssignment(
            tuple(_episode_job_from_wire(value) for value in body['jobs']),
            assignment_id=body.get('assignment_id'))
        if header.get('assignment_id') != assignment.assignment_id:
            raise RuntimeError('simulator assignment_id mismatch')
        return assignment

    def _execute_assignment(self, *, socket: Any, environment: BaseEnvironment,
                            policy_clients: list[BasePolicyClient],
                            evaluation_cfg: Mapping[str, Any],
                            assignment: EpisodeAssignment, save_videos: bool,
                            video_dir: str | None, video_fps: int) -> None:
        with _episode_timeout(self.episode_timeout_s):
            started = set()

            def report_episode_start(job: EpisodeJob) -> None:
                started.add(job.episode_id)
                _send_episode_event(
                    socket,
                    self._message_identity,
                    'episode_start',
                    job.episode_id,
                    assignment_id=assignment.assignment_id)

            def report_episode_end(job: EpisodeJob, result: Any) -> None:
                _send_episode_event(
                    socket,
                    self._message_identity,
                    'episode_end',
                    job.episode_id,
                    assignment_id=assignment.assignment_id,
                    payload=encode_frame(to_jsonable(result)))

            def report_case_rejected(job: EpisodeJob,
                                     info: Mapping[str, Any]) -> None:
                _send_episode_event(
                    socket,
                    self._message_identity,
                    'environment_case_rejected',
                    job.episode_id,
                    assignment_id=assignment.assignment_id,
                    replace_case=True,
                    invalidated=True,
                    message=str(
                        info.get('reason', 'Invalid benchmark sample')))

            try:
                execute_assignment(
                    environment=environment,
                    policy_clients=policy_clients,
                    assignment=assignment,
                    max_episode_steps=int(evaluation_cfg['max_episode_steps']),
                    execute_horizon=evaluation_cfg.get('execute_horizon'),
                    stop_on_success=bool(
                        evaluation_cfg.get('stop_on_success', True)),
                    on_episode_started=report_episode_start,
                    on_episode_finished=report_episode_end,
                    on_case_rejected=report_case_rejected,
                    save_videos=save_videos,
                    video_dir=video_dir,
                    video_fps=video_fps,
                    video_worker_id=self.worker_id,
                )
            except EnvironmentCaseRejected as exc:
                if started:
                    # Execution failures retry the same initial condition;
                    # only reset failures may consume a replacement layout.
                    raise RuntimeError(
                        'Environment case failed after episode start') from exc
                rejected = set(
                    range(len(assignment.jobs)) if exc.slots is
                    None else exc.slots)
                if not rejected <= set(range(len(assignment.jobs))):
                    raise ValueError(
                        'Rejected slot is outside the assignment') from exc
                for slot, job in enumerate(assignment.jobs):
                    _send_episode_event(
                        socket,
                        self._message_identity,
                        'environment_case_rejected',
                        job.episode_id,
                        assignment_id=assignment.assignment_id,
                        replace_case=slot in rejected,
                        message=str(exc))
            # Results alone do not release a worker. All simulator execution
            # and per-episode callbacks must return before accepting more work.
            socket.send_multipart([
                encode_header(
                    'assignment_end',
                    **self._message_identity,
                    assignment_id=assignment.assignment_id),
                empty_payload(),
            ])

    def _validate_header(self, header: Mapping[str, Any]) -> None:
        if header.get('deployment_id') != self.deployment_id:
            raise RuntimeError('ZMQ deployment_id mismatch')
        if header.get('worker_id') != self.worker_id:
            raise RuntimeError('simulator worker_id mismatch')
        if header.get('generation') != self.generation:
            raise RuntimeError('simulator worker generation mismatch')


def _run_simulator_worker(*, config_path: str, rollout_endpoint: str,
                          worker_id: str, deployment_id: str, generation: int,
                          device: str, episode_timeout_s: float,
                          config_overrides: Mapping[str, Any], parent_pid: int,
                          ready_event: Any, stop_received: Any) -> None:
    configure_worker_lifetime(parent_pid)
    print(
        f'[FluxThemis] simulator process id={worker_id} '
        f'pid={os.getpid()} generation={generation}',
        flush=True)
    _SimulatorWorker(
        config_path=config_path,
        rollout_endpoint=rollout_endpoint,
        worker_id=worker_id,
        deployment_id=deployment_id,
        generation=generation,
        device=device,
        episode_timeout_s=episode_timeout_s,
        config_overrides=config_overrides,
        ready_event=ready_event,
        stop_received=stop_received,
    ).run()


def _send_episode_event(socket: Any,
                        message_identity: Mapping[str, Any],
                        event_type: str,
                        episode_id: str,
                        payload: bytes | None = None,
                        **values: Any) -> None:
    socket.send_multipart([
        encode_header(
            event_type,
            **message_identity,
            episode_id=episode_id,
            **values,
        ),
        payload if payload is not None else empty_payload(),
    ])


def _episode_job_from_wire(value: Any) -> EpisodeJob:
    if not isinstance(value, Mapping):
        raise TypeError('episode job payload must be a mapping')
    data = dict(value)
    data['task'] = TaskSpec.from_value(data['task'])
    return EpisodeJob(**data)


@contextmanager
def _episode_timeout(timeout_s: float) -> Iterator[None]:
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _terminate_process(process: multiprocessing.Process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=10.0)
    if process.is_alive():
        process.kill()
        process.join()


def _endpoint(value: Any, name: str) -> str:
    value = _nonempty(value, name)
    if not value.startswith(('tcp://', 'ipc://', 'inproc://')):
        raise ValueError(f'{name} must be a complete ZMQ endpoint')
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value.strip()


def _positive_float(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or value <= 0):
        raise ValueError(f'{name} must be a positive number')
    return float(value)


__all__ = ['SimulatorWorkerSupervisor']
