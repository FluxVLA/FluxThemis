from fluxthemis.engines.runners.coordinator import (
    DistributedEvaluationCoordinator, _SchedulingState, _WorkerSlot)
from fluxthemis.engines.runners.episode import build_episode_jobs
from fluxthemis.engines.utils import to_jsonable
from fluxthemis.structures import TaskSpec
from fluxthemis.transports.zmq_protocol import encode_frame


class RouterSocket:

    def __init__(self, incoming=None):
        self.incoming = incoming
        self.sent = []

    def recv_multipart(self):
        return self.incoming

    def send_multipart(self, frames):
        self.sent.append(frames)


def make_jobs(count, tasks=('task', )):
    return build_episode_jobs([TaskSpec(t) for t in tasks], count, 7, 'run')


def make_coordinator(capacity,
                     jobs,
                     requires_single_task=False,
                     task_capacities=None,
                     *,
                     active=False):
    coordinator = DistributedEvaluationCoordinator(
        'inproc://assignments', 1, deployment_id='deployment')
    coordinator._scheduling_state = _SchedulingState.from_jobs(jobs)
    slot = _WorkerSlot('simulator-worker-0', b'worker', 0, '0')
    coordinator._worker_slots[slot.worker_id] = slot
    coordinator._worker_router = RouterSocket()
    tasks = {job.task.task_id: job.task for job in jobs}
    coordinator._mark_worker_ready(
        slot,
        encode_frame({
            'tasks': [to_jsonable(task) for task in tasks.values()],
            'capacity': capacity,
            'requires_single_task': requires_single_task,
            'task_capacities': task_capacities or {},
        }))
    if active:
        coordinator._dispatch_jobs()
        coordinator._worker_router.sent.clear()
    return coordinator, slot
