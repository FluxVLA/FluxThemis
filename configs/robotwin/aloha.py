"""RoboTwin ALOHA simulator and ZMQ observation/action interface."""

themis = dict(
    transport=dict(
        image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        state_keys=['qpos', 'states'],
        unnorm_key='private'),
    evaluation=dict(
        environment=dict(
            type='RoboTwinEnvironment',
            task_suite='all',
            robotwin_root=None,
            robotwin_seed=7,
            instruction_type='unseen'),
        policy_client=dict(type='FluxVLAPolicyClient'),
        reporter=dict(type='FluxVLAEvaluationReporter'),
        evaluator=dict(type='SuccessRateEvaluator'),
        base_seed=7,
        episodes_per_task=100,
        max_episode_steps=2000,
        execute_horizon=10,
        stop_on_success=True,
        save_videos=False,
        video_fps=10),
)
