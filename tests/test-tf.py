"""Test: create and run tensorflow policy network in environment loop."""

import sonnet as snt
from acme import wrappers
from acme.tf import networks as network_utils
from acme.tf import utils as tf2_utils

from flybody.fly_envs import walk_on_ball
from flybody.agents.network_factory import make_network_factory_dmpo


def test_can_create_and_run_tf_policy():

    env = walk_on_ball()
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=True)

    network_factory = make_network_factory_dmpo()
    networks = network_factory(env.action_spec())
    assert set(networks.keys()) == set(('observation', 'policy', 'critic'))

    policy_network = snt.Sequential([
        networks['observation'],
        networks['policy'],
        network_utils.StochasticSamplingHead()
    ])

    timestep = env.reset()
    for _ in range(100):
        batched_observation = tf2_utils.add_batch_dim(timestep.observation)
        action = policy_network(batched_observation)
        action = tf2_utils.to_numpy_squeeze(action)
        timestep = env.step(action)
