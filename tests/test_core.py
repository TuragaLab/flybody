"""Test core installation by creating an RL environment and stepping it."""

import numpy as np
from flybody.fly_envs import walk_on_ball


obs = ['accelerometer',
       'actuator_activation',
       'appendages_pos',
       'force',
       'gyro',
       'joints_pos',
       'joints_vel',
       'touch',
       'velocimeter',
       'world_zaxis',
       'ball_qvel']

obs = ['walker/' + s for s in obs]

def test_can_create_and_run_environment():

    env = walk_on_ball()
    
    obs_spec = env.observation_spec()
    assert list(obs_spec) == obs
    
    n_act = env.action_spec().shape
    assert n_act == (59,)

    timestep = env.reset()
    assert all(timestep.observation['walker/ball_qvel'] == [0, 0, 0])

    for _ in range(100):
        action = np.random.uniform(-1, 1, n_act)
        timestep = env.step(action)
        assert timestep.reward is not None

    # _ = env.physics.render()
