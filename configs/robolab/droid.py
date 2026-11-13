"""RoboLab DROID scalar evaluation through the shared remote launcher."""

themis = dict(
    evaluation=dict(
        base_seed=0,
        environment=dict(
            headless=True,
            num_envs=1,
            output_dir='work_dirs/robolab',
            robolab_root=None,
            simulator_seed=0,
            task_name='all',
            type='RoboLabEnvironment'),
        episodes_per_task=10,
        evaluator=dict(type='SuccessRateEvaluator'),
        execute_horizon=None,
        max_episode_steps=4500,
        policy_client=dict(type='FluxVLAPolicyClient'),
        reporter=dict(type='FluxVLAEvaluationReporter'),
        stop_on_success=True,
        strict_worker_assignment=False),
    transport=dict(
        image_keys=[
            'over_shoulder_left_camera',
            'wrist_cam',
        ],
        state_keys=[
            'states',
        ],
        unnorm_key='robolab_droid'))
