from fluxthemis.engines.runners import EvaluationRun, EvaluationSettings
from fluxthemis.evaluators import SuccessRateEvaluator
from fluxthemis.reporters.mock import MockEvaluationReporter
from fluxthemis.structures import EpisodeResult, TaskSpec


class _Reporter(MockEvaluationReporter):

    @property
    def evaluation_report_run_dir(self):
        return '/srv/fluxvla/evaluations/test-run'


def _result(task_id, episode_index, policy_seed, success=True):
    return EpisodeResult(
        task_id=task_id,
        episode_index=episode_index,
        policy_seed=policy_seed,
        environment_seed=policy_seed,
        environment_case_index=episode_index,
        success=success,
        episode_return=float(success),
        steps=3,
        duration_s=0.1,
        termination_reason='success' if success else 'max_steps',
        model_calls=1,
    )


def test_evaluation_run_tracks_metrics_and_reporting():
    evaluation_run = EvaluationRun(
        settings=EvaluationSettings(
            episodes_per_task=2,
            max_episode_steps=10,
            execute_horizon=2,
            run_name='test_run',
        ),
        evaluator=SuccessRateEvaluator(),
        reporter=_Reporter(run_session_id='reporting-run'),
    )
    tasks = (TaskSpec('task_a'), TaskSpec('task_b'))
    results = (
        _result('task_a', 0, 7),
        _result('task_b', 0, 9, success=False),
    )

    with evaluation_run:
        evaluation_run.start(tasks, total_episodes=len(results))
        for index, result in enumerate(results, start=1):
            evaluation_run.on_episode_end(result, f'run:{index}')
        summary = evaluation_run.finish(results[-1])

    assert summary['run']['tasks'] == ['task_a', 'task_b']
    assert summary['run']['completed_episodes'] == 2
    assert summary['metrics']['overall']['success_rate'] == 0.5
    assert summary['artifacts'] == {
        'remote_run_dir': '/srv/fluxvla/evaluations/test-run'
    }
