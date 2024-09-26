"""Script for distributed reinforcement learning training with Ray.

This script trains the fly-on-ball RL task using a distributed version of the
DMPO agent. The training runs in an infinite loop until terminated.

For lightweight testing, run this script with --test argument. It will run
training with a single actor and print training statistics every 10 seconds.

This script is not task-specific and can be used with other fly RL tasks by
swapping in other environments in the environment_factory function. The single
main configurable component below is the DMPO agent configuration and
training hyperparameters specified in the DMPOConfig data structure.
"""

# ruff: noqa: F821, E722, E402

# Start Ray cluster first, before imports.
import ray
try:
    # Try connecting to existing Ray cluster.
    ray_context = ray.init(address='auto', 
                           include_dashboard=True,
                           dashboard_host='0.0.0.0')
except:
    # Spin up new Ray cluster.
    ray_context = ray.init(include_dashboard=True, dashboard_host='0.0.0.0')


import argparse
import time
import os

from acme import specs
from acme import wrappers
import sonnet as snt

import flybody
from flybody.agents.remote_as_local_wrapper import RemoteAsLocal
from flybody.agents.counting import PicklableCounter
from flybody.agents.network_factory import policy_loss_module_dmpo
from flybody.agents.losses_mpo import PenalizationCostRealActions

from flybody.fly_envs import walk_on_ball
from flybody.agents.network_factory import make_network_factory_dmpo

from flybody.agents.ray_distributed_dmpo import (
    DMPOConfig,
    ReplayServer,
    Learner,
    EnvironmentLoop,
)

PYHTONPATH = os.path.dirname(os.path.dirname(flybody.__file__))
LD_LIBRARY_PATH = (
    os.environ['LD_LIBRARY_PATH'] if 'LD_LIBRARY_PATH' in os.environ else '')
# Defer specifying CUDA_VISIBLE_DEVICES to sub-processes.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


parser = argparse.ArgumentParser()
parser.add_argument('--test', '-t',
    help='Run job in test mode with one actor and output to current terminal.',
    action='store_true')
args = parser.parse_args()
is_test = args.test
if parser.parse_args().test:
    print('\nRun job in test mode with one actor.')
    test_num_actors = 1
    test_log_every = 10
    test_min_replay_size = 40
else:
    test_num_actors = None
    test_log_every = None
    test_min_replay_size = None

print('\nRay context:')
print(ray_context)

ray_resources = ray.available_resources()
print('\nAvailable Ray cluster resources:')
print(ray_resources)

# Create environment factory for walk-on-ball fly RL task.
def environment_factory(training: bool) -> 'composer.Environment':
    """Creates replicas of environment for the agent."""
    del training  # Unused.
    env = walk_on_ball()
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    return env
# Create network factory for RL task.
network_factory = make_network_factory_dmpo()

# Dummy environment and network for quick use, deleted later.
dummy_env = environment_factory(training=True)
dummy_net = network_factory(dummy_env.action_spec())
# Get full environment specs.
environment_spec = specs.make_environment_spec(dummy_env)

# This callable will be calculating penalization cost by converting canonical
# actions to real (not wrapped) environment actions inside DMPO agent.
# Note that we need the action_spec of the underlying environment so we unwrap
# with dummy_env._environment.
penalization_cost = PenalizationCostRealActions(dummy_env._environment.action_spec())

# Distributed DMPO agent configuration.
dmpo_config = DMPOConfig(
    num_actors=test_num_actors or 32,
    batch_size=256,
    prefetch_size=4,
    num_learner_steps=100,
    min_replay_size=test_min_replay_size or 10_000,
    max_replay_size=4_000_000,
    samples_per_insert=15,
    n_step=5,
    num_samples=20,
    policy_loss_module=policy_loss_module_dmpo(
        epsilon=0.1,
        epsilon_mean=0.0025,
        epsilon_stddev=1e-7,
        action_penalization=True,
        epsilon_penalty=0.1,
        penalization_cost=penalization_cost,
    ),
    policy_optimizer=snt.optimizers.Adam(1e-4),
    critic_optimizer=snt.optimizers.Adam(1e-4),
    dual_optimizer=snt.optimizers.Adam(1e-3),
    target_critic_update_period=107,
    target_policy_update_period=101,
    actor_update_period=1000,
    log_every=test_log_every or 5*60,
    logger_save_csv_data=False,
    checkpoint_max_to_keep=None,
    checkpoint_directory='~/ray-ckpts/',
    checkpoint_to_load=None,
    print_fn=print
)

# Print full job config and full environment specs.
print('\n', dmpo_config)
print('\n', dummy_net)
print('\nobservation_spec:\n', dummy_env.observation_spec())
print('\naction_spec:\n', dummy_env.action_spec())
print('\ndiscount_spec:\n', dummy_env.discount_spec())
print('\nreward_spec:\n', dummy_env.reward_spec(), '\n')
del dummy_env
del dummy_net

# Environment variables for learner, actor, and replay buffer processes.
runtime_env_learner = {
    'env_vars': {
        'MUJOCO_GL': 'egl',
        'CUDA_VISIBLE_DEVICES': '0',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'PYTHONPATH': PYHTONPATH,
        'LD_LIBRARY_PATH': LD_LIBRARY_PATH,
    }
}
runtime_env_actor = {
    'env_vars': {
        'MUJOCO_GL': 'egl',
        'MUJOCO_EGL_DEVICE_ID': '0',
        'CUDA_VISIBLE_DEVICES': '-1',  # CPU-actors don't use CUDA.
        'PYTHONPATH': PYHTONPATH,
        'LD_LIBRARY_PATH': LD_LIBRARY_PATH,
    }
}

# === Create Replay Server.
runtime_env_replay = {
    'env_vars': {
        'PYTHONPATH': PYHTONPATH,  # Also used for counter.
    }
}
ReplayServer = ray.remote(
    num_gpus=0, runtime_env=runtime_env_replay)(ReplayServer)
replay_server = ReplayServer.remote(dmpo_config, environment_spec)
addr = ray.get(replay_server.get_server_address.remote())
print(f'Started Replay Server on {addr}')

# === Create Counter.
counter = ray.remote(PicklableCounter)  # This is class (direct call to
                                        # ray.remote decorator).
counter = counter.remote()  # Instantiate.
counter = RemoteAsLocal(counter)

# === Create Learner.
Learner = ray.remote(
    num_gpus=1, runtime_env=runtime_env_learner)(Learner)
learner = Learner.remote(replay_server.get_server_address.remote(),
                         counter,
                         environment_spec,
                         dmpo_config,
                         network_factory)
learner = RemoteAsLocal(learner)

print('Waiting until learner is ready...')
learner.isready(block=True)

checkpointer_dir, snapshotter_dir = learner.get_checkpoint_dir()
print('Checkpointer directory:', checkpointer_dir)
print('Snapshotter directory:', snapshotter_dir)

# === Create Actors and Evaluator.

EnvironmentLoop = ray.remote(
    num_gpus=0, runtime_env=runtime_env_actor)(EnvironmentLoop)

n_actors = dmpo_config.num_actors

def create_actors(n_actors):
    """Return list of requested number of actor instances."""
    actors = []
    for _ in range(n_actors):
        actor = EnvironmentLoop.remote(
            replay_server_address=replay_server.get_server_address.remote(),
            variable_source=learner,
            counter=counter,
            network_factory=network_factory,
            environment_factory=environment_factory,
            dmpo_config=dmpo_config,
            actor_or_evaluator='actor',
            )
        actor = RemoteAsLocal(actor)
        actors.append(actor)
        time.sleep(0.2)
    return actors

# Get actors.
actors = create_actors(n_actors)

# Get evaluator.
evaluator = EnvironmentLoop.remote(
    replay_server_address=replay_server.get_server_address.remote(),
    variable_source=learner,
    counter=counter,
    network_factory=network_factory,
    environment_factory=environment_factory,
    dmpo_config=dmpo_config,
    actor_or_evaluator='evaluator')
evaluator = RemoteAsLocal(evaluator)

print('Waiting until actors are ready...')
# Block until all actors and evaluator are ready and have called `get_variables`
# in learner with variable_client.update_and_wait() from _make_actor. Otherwise
# they will be blocked and won't be inserting data to replay table, which in
# turn will cause learner to be blocked.
for actor in actors:
    actor.isready(block=True)
evaluator.isready(block=True)

print('Actors ready, issuing run command to all')

# === Run all.
if hasattr(counter, 'run'):
    counter.run(block=False)
for actor in actors:
    actor.run(block=False)
evaluator.run(block=False)

while True:
    # Single call to `run` makes a fixed number of learning steps.
    # Here we need to block, otherwise `run` calls pile up and spam the queue.
    learner.run(block=True)
