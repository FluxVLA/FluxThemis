"""Run and account for local simulator processes in a prepared environment."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class ProcessTree:
    """Track descendants across torchrun's new sessions and reparenting."""

    def __init__(self, pid):
        self.known = {}
        row = self.snapshot().get(pid)
        if row:
            self.known[pid] = row[2]

    @staticmethod
    def snapshot():
        rows = {}
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields = path.read_text().rsplit(')', 1)[1].split()
                if fields[0] == 'Z':
                    continue
                # Parent PID, process group and start time (guards PID reuse).
                rows[int(path.parent.name)] = (int(fields[1]), int(fields[2]),
                                               int(fields[19]))
            except (OSError, ValueError, IndexError):
                continue
        return rows

    def members(self):
        rows = self.snapshot()
        owned = {
            pid
            for pid, started in self.known.items()
            if pid in rows and rows[pid][2] == started
        }
        while True:
            children = {pid for pid, row in rows.items() if row[0] in owned}
            added = children - owned
            if not added:
                break
            owned.update(added)
        self.known = {pid: rows[pid][2] for pid in owned}
        return {pid: rows[pid] for pid in owned}

    def signal_groups(self, signum):
        # Only groups containing still-live, identity-checked descendants.
        for group in {row[1] for row in self.members().values()}:
            try:
                os.killpg(group, signum)
            except ProcessLookupError:
                pass


class ProcessRunner:
    """Manage only processes launched by this evaluation."""

    def __init__(self, plan):
        self.plan = plan
        self.python = plan['local']['python']
        self.env = dict(os.environ)
        # One simulator server, independently of the model cluster's rank.
        # Reject multi-node jobs rather than run a full duplicate per node.
        for name in ('WORLD_SIZE', 'MLP_WORKER_NUM'):
            if int(self.env.get(name, '1')) != 1:
                raise ValueError(
                    'The evaluation launcher supports one simulator '
                    'node per evaluation; use a 1-node sim job. '
                    'distributed_eval.sh supports multi-node '
                    'launching independently.')
        for name in ('RANK', 'MLP_ROLE_INDEX'):
            if int(self.env.get(name, '0')) != 0:
                raise ValueError('Evaluation launcher must run on rank 0')
        self.env.update(
            FLUXTHEMIS_PYTHON=self.python,
            PYTHONPATH=os.pathsep.join((str(ROOT / 'scripts'), str(ROOT),
                                        self.env.get('PYTHONPATH', ''))),
            PYTHONUNBUFFERED='1',
            PYTHONDONTWRITEBYTECODE='1',
            MASTER_ADDR='127.0.0.1',
            MASTER_PORT=str(plan['local']['port'] + 1),
            WORLD_SIZE='1',
            RANK='0')
        if plan['local'].get('robodojo_root'):
            self.env['ROBODOJO_ROOT'] = plan['local']['robodojo_root']
        if plan['local'].get('robotwin_root'):
            self.env['ROBOTWIN_ROOT'] = plan['local']['robotwin_root']
        if plan['local'].get('cache'):
            self.env['XDG_CACHE_HOME'] = plan['local']['cache']
        self.env.pop('NPROC_PER_NODE', None)
        self.env.pop('LOCAL_RANK', None)
        self.env.pop('LOCAL_WORLD_SIZE', None)
        self._identity = None

    def connect(self):
        if self._identity is not None:
            return
        if not os.access(self.python, os.X_OK):
            raise ValueError(
                f'Simulator Python is not executable: {self.python}')
        local = self.plan['local']
        if local.get('robodojo_root'):
            root = Path(local['robodojo_root'])
            if (root / 'Assets').resolve() != Path(local['assets']).resolve():
                raise ValueError(
                    'RoboDojo/Assets must point to simulator.assets')
        result = subprocess.run([
            self.python, '-m', 'preflight.environment',
            local.get('frozen_config', local['config'])
        ],
                                env=self.env,
                                cwd=ROOT,
                                stdin=subprocess.DEVNULL,
                                capture_output=True,
                                text=True,
                                timeout=120)
        (Path(self.plan['output']) /
         'preflight.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError('Simulator environment check failed; see '
                               f'{self.plan["output"]}/preflight.log')
        self._identity = json.loads(result.stdout.splitlines()[-1])

    def identity(self):
        self.connect()
        return self._identity

    def run_logged(self,
                   args,
                   logfile,
                   progress,
                   timeout,
                   tick=None,
                   on_start=None):
        process = tree = None
        started = time.monotonic()
        next_update = 0
        try:
            with logfile.open('w') as stream:
                process = subprocess.Popen(
                    list(map(str, args)),
                    env=self.env,
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True)
                tree = ProcessTree(process.pid)
                if on_start:
                    on_start(process.pid)
                while process.poll() is None:
                    tree.members()
                    if timeout is not None and time.monotonic(
                    ) - started > timeout:
                        raise TimeoutError(
                            f'Exceeded {timeout}s; see {logfile}')
                    if time.monotonic() >= next_update:
                        if tick:
                            tick()
                        else:
                            progress.show(f'Running; log: {logfile}')
                        next_update = time.monotonic() + 5
                    time.sleep(.25)
                if tick:
                    tick()
                return process.returncode
        finally:
            if process:
                # torchrun workers have their own process groups/sessions.
                tree.signal_groups(signal.SIGTERM)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    process.poll()
                    if not tree.members():
                        break
                    time.sleep(.05)
                else:
                    tree.signal_groups(signal.SIGKILL)
                process.wait(timeout=5)
