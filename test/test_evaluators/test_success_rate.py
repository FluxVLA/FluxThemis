import pytest

from fluxthemis.evaluators.success_rate import SuccessRateEvaluator
from fluxthemis.structures import EpisodeResult, TaskSpec


def _result(task_id, success):
    return EpisodeResult(
        task_id=task_id,
        episode_index=0,
        seed=7,
        success=success,
        episode_return=float(success),
        steps=2,
        duration_s=0.5,
        termination_reason='success' if success else 'max_steps',
        model_calls=1,
    )


def test_default_summary_remains_backward_compatible():
    evaluator = SuccessRateEvaluator()
    evaluator.on_run_start([TaskSpec('a', metadata={'category': 'Camera'})])
    evaluator.on_episode_end(_result('a', True))

    summary = evaluator.summarize()

    assert set(summary) == {'overall', 'tasks'}


def test_groups_success_by_each_metadata_field_including_null():
    evaluator = SuccessRateEvaluator(
        group_by_metadata=['category', 'difficulty_level'])
    evaluator.on_run_start([
        TaskSpec(
            'a',
            metadata={
                'category': 'Camera Viewpoints',
                'difficulty_level': 3,
            }),
        TaskSpec(
            'b',
            metadata={
                'category': 'Camera Viewpoints',
                'difficulty_level': None,
            }),
        TaskSpec(
            'c',
            metadata={
                'category': 'Light Conditions',
                'difficulty_level': 3,
            }),
    ])
    evaluator.on_episode_end(_result('a', True))
    evaluator.on_episode_end(_result('b', False))
    evaluator.on_episode_end(_result('c', False))

    groups = evaluator.summarize()['groups']

    assert groups['category']['Camera Viewpoints']['success_rate'] == 0.5
    assert groups['category']['Light Conditions']['success_rate'] == 0.0
    assert groups['difficulty_level']['3']['success_rate'] == 0.5
    assert groups['difficulty_level']['null']['episodes'] == 1


def test_grouping_rejects_unknown_metadata_field():
    evaluator = SuccessRateEvaluator(group_by_metadata=['unknown'])

    with pytest.raises(ValueError, match='Unknown group_by_metadata'):
        evaluator.on_run_start([TaskSpec('a', metadata={'category': 'x'})])
