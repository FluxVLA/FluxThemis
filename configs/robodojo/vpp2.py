"""RoboDojo evaluation for VPP2's absolute EE16 actions and camera history."""

_base_ = ['./arx_x5.py']

themis = dict(
    transport=dict(
        state_keys=[
            'left_ee_pose', 'left_gripper_qpos', 'right_ee_pose',
            'right_gripper_qpos'
        ],
        unnorm_key='robodojo_ee16'),
    evaluation=dict(
        environment=dict(
            action_mode='ee',
            seed=int('{{$ROBODOJO_EVAL_SEED:1}}')),
        policy_client=dict(
            replan_steps=24,
            history=dict(num_frames=8, stride=25, include_anchor=True)),
        base_seed=1,
        execute_horizon=24))
