"""Durable episode recovery, independent of a model server's live session.

Only acknowledged episode_end events are reused. A fresh report session
receives them through the existing protocol. Unfinished episodes reset.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        # Persist complete contents before replacement, then sync the parent
        # directory so the replacement itself is durable across a crash.
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(',', ':'),
            allow_nan=False).encode()).hexdigest()


def read_journal(path):
    """Ignore only an incomplete last line; reject interior corruption."""
    rows = []
    for line in Path(path).read_text().splitlines(keepends=True):
        if line.endswith('\n'):
            rows.append(json.loads(line))
    return rows


def identity(payload):
    return payload['task_id'], payload['episode_index']


def manifest(start):
    # Run/session identities change on resume; the benchmark definition must
    # remain identical before old and new results can be combined.
    keys = ('base_seed', 'episodes_per_task', 'episodes_per_task_by_task',
            'max_episode_steps', 'execute_horizon', 'stop_on_success',
            'task_ids', 'tasks', 'total_episodes', 'strict_worker_assignment')
    value = {key: start.get(key) for key in keys}
    value['environment_case_rejection_limit'] = start.get(
        'environment_case_rejection_limit', 100)
    return value


def canonical_jobs(start):
    task_ids = [task['task_id'] for task in start['tasks']]
    counts = start['episodes_per_task_by_task']
    if (task_ids != start['task_ids'] or not task_ids
            or len(task_ids) != len(set(task_ids))
            or set(counts) != set(task_ids) or any(
                type(count) is not int or count < 1
                for count in counts.values())):
        raise ValueError('Invalid recovery task inventory')
    jobs = []
    # Derive ordinals and seeds from the full task list, before removing any
    # completed episodes, or the remaining episodes would receive new seeds.
    for task in start['tasks']:
        task_id = task['task_id']
        count = start['episodes_per_task_by_task'][task_id]
        for index in range(count):
            seed = start['base_seed'] + len(jobs)
            jobs.append(
                dict(
                    global_episode_number=len(jobs) + 1,
                    task_id=task_id,
                    episode_index=index,
                    policy_seed=seed,
                    environment_seed=seed,
                    environment_case_index=index))
    if len(jobs) != start['total_episodes'] or len({identity(j)
                                                    for j in jobs
                                                    }) != len(jobs):
        raise ValueError('Invalid recovery task manifest')
    return jobs


def validate_journal(rows):
    if not rows or rows[0]['event_type'] != 'run_start':
        raise ValueError('No acknowledged run_start; nothing to resume')
    session = rows[0]['run_session_id']
    start = rows[0]['payload']
    jobs = {identity(j): j for j in canonical_jobs(start)}
    starts, completed = {}, {}
    ended = False
    # Missing or mixed-session events make completion counts ambiguous;
    # reject them instead of inferring progress from the surviving records.
    for sequence, event in enumerate(rows, 1):
        if event['sequence'] != sequence or event['run_session_id'] != session:
            raise ValueError('Recovery journal sequence/session mismatch')
        if ended:
            raise ValueError('Recovery journal contains events after run_end')
        kind, body = event['event_type'], event['payload']
        if kind == 'run_start':
            if sequence != 1:
                raise ValueError('Duplicate run_start')
        elif kind in ('episode_start', 'episode_end'):
            key = identity(body)
            job = jobs.get(key)
            if job is None or key in completed:
                raise ValueError('Unknown or duplicate completed episode')
            for field in ('policy_seed', 'environment_seed'):
                if body[field] != job[field]:
                    raise ValueError(f'Episode {key}: {field} changed')
            expected_id = f'{session}:{job["global_episode_number"]}'
            if body['episode_id'] != expected_id:
                raise ValueError('Episode ID does not match canonical ordinal')
            if kind == 'episode_start':
                if key in starts:
                    previous = starts[key]['environment_case_index']
                    replacement = body.get('replaces_environment_case_index')
                    if (type(replacement) is not int or replacement != previous
                            or type(body['environment_case_index']) is not int
                            or body['environment_case_index'] == previous):
                        raise ValueError('Duplicate episode_start')
                starts[key] = body
            else:
                if key not in starts:
                    raise ValueError('episode_end without episode_start')
                for field in ('episode_id', 'environment_case_index'):
                    if body[field] != starts[key][field]:
                        raise ValueError(f'Episode start/end {field} mismatch')
                if (body['completed_episodes'] != len(completed) + 1
                        or body['total_episodes'] != len(jobs)):
                    raise ValueError('Recovery completion counts disagree')
                if type(body['success']) is not bool:
                    raise ValueError('Episode success must be a boolean')
                if not math.isfinite(
                        body['duration_s']) or body['duration_s'] < 0:
                    raise ValueError('Invalid episode duration')
                completed[key] = dict(start=starts[key], end=body)
        elif kind == 'run_end':
            if (body['completed_episodes'] != len(completed)
                    or body['total_episodes'] != len(jobs)):
                raise ValueError('Terminal completion counts disagree')
            ended = True
        else:
            raise ValueError(f'Unknown recovery event: {kind}')
    return start, starts, completed


def read_bundle(path):
    data = json.loads(Path(path).read_text())
    validate_bundle(data)
    return data


def build_bundle(directory):
    """Recover acknowledged results; rebuild pending jobs and reset retries."""
    directory = Path(directory).resolve()
    journal = directory / 'events.jsonl'
    rows = read_journal(journal)
    start, starts, completed = validate_journal(rows)
    inherited = {}
    if start.get('resumed_from'):
        inherited = read_bundle(directory / 'resume_input.json')
        if (manifest(inherited['start']) != manifest(start)
                or inherited['source'] != start['resumed_from']):
            raise ValueError('Inherited recovery task manifest/source changed')
    # Preserve results if a new attempt was interrupted during replay.
    merged = {
        identity(pair['end']): pair
        for pair in inherited.get('completed', [])
    }
    for key in merged.keys() & completed.keys():
        for field in ('policy_seed', 'environment_seed',
                      'environment_case_index', 'success', 'steps',
                      'model_calls', 'episode_return', 'duration_s',
                      'termination_reason'):
            if merged[key]['end'][field] != completed[key]['end'][field]:
                raise ValueError(f'Replayed result changed {field}')
    merged.update(completed)
    jobs = canonical_jobs(start)
    previously_started = {
        identity(job)
        for job in inherited.get('jobs', []) if job.get('started', False)
    }
    for job in jobs:
        job['started'] = (
            identity(job) in starts or identity(job) in previously_started)
    pending_cases = {
        identity(job): job['environment_case_index']
        for job in inherited.get('jobs', [])
    }
    pending_cases.update({
        key: body['environment_case_index']
        for key, body in starts.items()
    })
    used = {task: set() for task in start['task_ids']}
    # Reserve the actual completed layouts first: rejected defaults may have
    # been replaced with indexes that overlap another episode's default case.
    for job in jobs:
        key = identity(job)
        if key in merged:
            case = merged[key]['end']['environment_case_index']
            if type(case) is not int or case < 0 or case in used[key[0]]:
                raise ValueError('Invalid or duplicate completed layout')
            job['environment_case_index'] = case
            used[key[0]].add(case)
    # An acknowledged start fixes the initial condition even when its result
    # was lost. Reserve these before assigning defaults to unstarted episodes.
    for job in jobs:
        key = identity(job)
        if key not in merged and key in pending_cases:
            case = pending_cases[key]
            if type(case) is not int or case < 0 or case in used[key[0]]:
                raise ValueError(
                    'Invalid or duplicate pending recovery layout')
            job['environment_case_index'] = case
            used[key[0]].add(case)
    next_cases = {
        task: max([start['episodes_per_task_by_task'][task] - 1, *cases]) + 1
        for task, cases in used.items()
    }
    # Restart unfinished episodes from canonical cases unless already reserved.
    # Unrecorded layout rejections may be encountered again after recovery.
    for job in jobs:
        task = job['task_id']
        if identity(job) not in merged and identity(job) not in pending_cases:
            case = job['environment_case_index']
            if case in used[task]:
                case = next_cases[task]
                next_cases[task] += 1
            job['environment_case_index'] = case
            used[task].add(case)
            next_cases[task] = max(next_cases[task], case + 1)
    next_cases = {task: max(cases) + 1 for task, cases in used.items()}
    live_starts = [
        datetime.fromisoformat(e['payload']['started_at']).timestamp()
        for e in rows if e['event_type'] == 'episode_start'
        and not e['payload'].get('resumed_from')
    ]
    # Count only the recorded window of new execution in this attempt. Replay
    # and downtime are excluded; the unrecorded tail before a crash is unknown.
    end = datetime.fromisoformat(rows[-1]['recorded_at']).timestamp()
    if rows[-1]['event_type'] == 'run_end':
        end = datetime.fromisoformat(
            rows[-1]['payload']['finished_at']).timestamp()
    elapsed = max(0., end - min(live_starts)) if live_starts else 0.
    bundle = dict(
        schema_version=2,
        source=str(directory),
        start=start,
        jobs=jobs,
        completed=list(merged.values()),
        next_environment_case_index_by_task=next_cases,
        elapsed_execution_s=inherited.get('elapsed_execution_s', 0.) + elapsed,
        source_journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest())
    validate_bundle(bundle)
    return bundle


def validate_bundle(data):
    if data.get('schema_version') != 2:
        raise ValueError('Unsupported resume bundle version')
    if not isinstance(data.get('source'), str) or not data['source']:
        raise ValueError('Recovery source is missing')
    elapsed = data['elapsed_execution_s']
    if (not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed)
            or elapsed < 0):
        raise ValueError('Invalid recovery active duration')
    # Bundles must retain every canonical job, including completed ones, so
    # resumed dispatch cannot silently omit work or change its identity.
    original = {identity(j): j for j in canonical_jobs(data['start'])}
    jobs = {identity(j): j for j in data['jobs']}
    if jobs.keys() != original.keys() or len(jobs) != len(data['jobs']):
        raise ValueError('Recovery jobs do not match the full task manifest')
    cases = set()
    for key, job in jobs.items():
        if type(job.get('started', False)) is not bool:
            raise ValueError('Invalid recovery episode started flag')
        for field in ('global_episode_number', 'policy_seed',
                      'environment_seed'):
            if job[field] != original[key][field]:
                raise ValueError(f'Recovery changed canonical {field}')
        case = job['environment_case_index']
        if type(case) is not int or case < 0 or (key[0], case) in cases:
            raise ValueError('Invalid or duplicate recovery layout index')
        cases.add((key[0], case))
        if data['next_environment_case_index_by_task'][key[0]] <= case:
            raise ValueError('Recovery next layout index would reuse a case')
    done = set()
    for pair in data['completed']:
        start, end = pair['start'], pair['end']
        key = identity(end)
        if key in done or key not in jobs or identity(start) != key:
            raise ValueError('Duplicate or unknown restored result')
        for field in ('policy_seed', 'environment_seed',
                      'environment_case_index'):
            if end[field] != jobs[key][field] or start[field] != end[field]:
                raise ValueError(f'Restored result changed {field}')
        if type(end['success']) is not bool:
            raise ValueError('Invalid restored success value')
        for field in ('duration_s', 'episode_return'):
            if not math.isfinite(end[field]):
                raise ValueError(f'Invalid restored {field}')
        if end['duration_s'] < 0:
            raise ValueError('Invalid restored duration')
        for field in ('steps', 'model_calls'):
            if type(end[field]) is not int or end[field] < 0:
                raise ValueError(f'Invalid restored {field}')
        for body, field in ((start, 'started_at'), (end, 'finished_at')):
            if datetime.fromisoformat(body[field]).tzinfo is None:
                raise ValueError('Recovery timestamps need a timezone')
        done.add(key)
