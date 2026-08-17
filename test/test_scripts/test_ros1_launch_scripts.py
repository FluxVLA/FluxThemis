import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / 'scripts'


@pytest.mark.parametrize(
    'name',
    [
        'install.sh',
        'install_libero_plus.sh',
        'install_ros1_hook.sh',
        'run_ros1_eval.sh',
    ],
)
def test_ros1_launch_scripts_have_valid_bash_syntax(name):
    subprocess.run(
        ['/bin/bash', '-n', str(SCRIPTS / name)],
        check=True,
    )


def test_one_command_eval_help_is_environment_independent():
    completed = subprocess.run(
        ['/bin/bash', str(SCRIPTS / 'run_ros1_eval.sh'), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '--smoke' in completed.stdout
    assert '--service-timeout' in completed.stdout
    assert '--shutdown-master' in completed.stdout
    assert '--workers' in completed.stdout
    assert '--sim-gpus' in completed.stdout
    assert 'FLUXTHEMIS_PARALLEL_WORKERS' in completed.stdout
    assert 'FLUXTHEMIS_SIM_GPU_IDS' in completed.stdout
    assert '--service-name' not in completed.stdout
    assert 'runner-only --cfg-options' in completed.stdout
    assert 'fluxthemis-eval' in completed.stdout


def test_installer_documents_fluxvla_ros1_hook():
    completed = subprocess.run(
        ['/bin/bash', str(SCRIPTS / 'install.sh'), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '--fluxvla-env-name' in completed.stdout
    assert '--skip-fluxvla-hook' in completed.stdout


def test_installer_exposes_isolated_libero_plus_profile():
    completed = subprocess.run(
        ['/bin/bash', str(SCRIPTS / 'install.sh'), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )

    assert 'libero-plus' in completed.stdout
    assert '--libero-plus-asset-mode' in completed.stdout
    assert '--libero-plus-asset-root' in completed.stdout


def test_libero_plus_installer_dry_run(tmp_path):
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    fake_conda = fake_bin / 'conda'
    fake_conda.write_text(
        '#!/usr/bin/env bash\n'
        'case "$*" in\n'
        '  *sys.prefix*) echo /tmp/fake-libero-plus-prefix ;;\n'
        '  *sys.version_info*) echo 3.10 ;;\n'
        'esac\n',
        encoding='utf-8',
    )
    fake_conda.chmod(0o755)
    completed = subprocess.run(
        [
            '/bin/bash',
            str(SCRIPTS / 'install_libero_plus.sh'),
            '--env-name',
            'fake-libero-plus',
            '--asset-mode',
            'none',
            '--system-deps',
            'never',
            '--smoke-level',
            'none',
            '--dry-run',
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            'PATH': f'{fake_bin}:/usr/bin:/bin',
        },
    )

    output = completed.stdout + completed.stderr
    assert 'd7c160eca38bf268d5965aef6d4c6dca12d5537c' in output
    assert 'requirements-libero-plus.txt' in output
    assert 'fluxthemis-libero-plus.sh' in output
    assert 'Dry run complete' in output
