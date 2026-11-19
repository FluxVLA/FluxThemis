"""LIBERO-Plus LIBERO-10 simulator with PI0.5 evaluation defaults."""

themis = dict(
    transport=dict(
        image_keys=['agentview_image', 'robot0_eye_in_hand_image'],
        state_keys=[
            'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'
        ],
        unnorm_key='libero_10_no_noops'),
    evaluation=dict(
        environment=dict(
            type='LiberoPlusEnvironment',
            task_suite_name='libero_10',
            task_order_index=0,
            resolution=256,
            controller='OSC_POSE',
            simulator_seed=0,
            num_steps_wait=10),
        policy_client=dict(type='FluxVLAPolicyClient'),
        reporter=dict(type='FluxVLAEvaluationReporter'),
        evaluator=dict(
            type='SuccessRateEvaluator',
            group_by_metadata=['category', 'difficulty_level']),
        base_seed=7,
        episodes_per_task=1,
        episodes_per_task_overrides={},
        task_ids=None,
        task_metadata_filters=None,
        max_episode_steps=520,
        execute_horizon=10,
        stop_on_success=True,
        save_videos=False,
        video_fps=20),
)
