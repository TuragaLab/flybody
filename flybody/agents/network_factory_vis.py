"""Two-level-controller networks for controller reuse and vision-guided flight."""

from typing import Sequence, Callable, Optional

import tensorflow as tf
import tensorflow_probability as tfp

import sonnet as snt
from acme import types
from acme.tf import networks as acme_networks
from acme.tf import utils as tf2_utils

from flybody.agents.utils_tf import restore_dmpo_networks_from_checkpoint
from flybody.tasks.task_utils import observable_indices_in_tensor

tfd = tfp.distributions


def make_vis_network_factory_two_level_controller(
    ll_network_ckpt_path,
    ll_network_factory,
    ll_environment_spec,
    hl_network_layer_sizes=(256, 256, 128),
    steering_command_dim=7,
    task_input_dim=2,
    vis_output_dim=8,
    critic_layer_sizes=(512, 512, 256),
    vmin=-150.,
    vmax=150.,
    num_atoms=51,
):
    """Closure for vis_network_factory_two_level_controller."""

    def network_factory(action_spec):
        return vis_network_factory_two_level_controller(
            action_spec,
            ll_network_ckpt_path=ll_network_ckpt_path,
            ll_network_factory=ll_network_factory,
            ll_environment_spec=ll_environment_spec,
            hl_network_layer_sizes=hl_network_layer_sizes,
            steering_command_dim=steering_command_dim,
            task_input_dim=task_input_dim,
            vis_output_dim=vis_output_dim,
            critic_layer_sizes=critic_layer_sizes,
            vmin=vmin,
            vmax=vmax,
            num_atoms=num_atoms,                        
        )
    return network_factory


def vis_network_factory_two_level_controller(
    action_spec,
    ll_network_ckpt_path,
    ll_network_factory,
    ll_environment_spec,
    hl_network_layer_sizes=(256, 256, 128),
    steering_command_dim=7,
    task_input_dim=2,
    vis_output_dim=8,
    critic_layer_sizes=(512, 512, 256),
    vmin=-150.,
    vmax=150.,
    num_atoms=51,
):
    """Two-level-controller network factory for vision-guided flight tasks.
    
    Args:
        action_spec: Vision task environment action spec, e.g. env.action_spec().
        ll_network_ckpt_path: Path to checkpoint with trained low-level controller
            network.
        ll_network_factory: Callable to create low-level controller network. The
            network architecture must match the checkpoint weights to be loaded
            with the `ll_network_ckpt_path` checkpoint.
        ll_environment_spec: Specs of the environment used to pre-train the low-level
            controller network. The specs can be obtained with:
            `ll_environment_spec = acme.specs.make_environment_spec(ll_env)`.
        hl_network_layer_sizes: List of MLP layer sizes for the high-level controller
            network. See the `TwoLevelController` class below.
        steering_command_dim: Dimension of the steering command output by the
            high-level and passed to the low-level controller.
        task_input_dim: Dimension of the optional task input passed to the
            high-level controller by the environment. For example, if
            task_input_dim == 2, the task input could be target speed and target
            altitude for the current episode.
        vis_output_dim: Output dimension of the visual processing network to be passed
            to the high-level controller. See the VisNet class below.
        critic_layer_sizes: List of MLP layer sizes for the critic network.
        vmin, vmax: Minimum and maximum values for the critic network value head.
        num_atoms: Number of atoms in the critic network value head.

    Returns:
        A dict with three networks: policy, critic, and observation.
    """

    # === Observation network.
    observation_network = VisNet(vis_output_dim=vis_output_dim)

    # === Build policy network.
    
    # Load trained networks from flight_imitation task.
    loaded_networks = restore_dmpo_networks_from_checkpoint(
        ckpt_path=ll_network_ckpt_path,
        network_factory=ll_network_factory,
        environment_spec=ll_environment_spec)
    ll_network = loaded_networks.policy_network

    # Initialize policy with flight_imitation policy as low-level controller.
    policy_network = TwoLevelController(
        hl_layer_sizes=hl_network_layer_sizes,
        ll_network=ll_network,
        ll_environment_spec=ll_environment_spec,
        steering_command_dim=steering_command_dim,
        task_input_dim=task_input_dim,
        vis_output_dim=vis_output_dim,
    )
    # Make sure all low-level network variables are frozen.
    for variable in policy_network._ll_network.variables:
        assert not variable._trainable

    # === Build critic network.
    
    # The multiplexer concatenates the (maybe transformed) observations/actions.
    critic_network = acme_networks.CriticMultiplexer(
        action_network=acme_networks.ClipToSpec(action_spec),
        critic_network=acme_networks.LayerNormMLP(layer_sizes=critic_layer_sizes,
                                                  activate_final=True),
    )
    critic_network = snt.Sequential([
        critic_network,
        acme_networks.DiscreteValuedHead(vmin=vmin, vmax=vmax, num_atoms=num_atoms)
    ])

    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network,
    }


class VisNet(snt.Module):
    """Visual network for "simple" eyes with no buffers."""

    def __init__(self, vis_output_dim=1):
        """Visual convolutional network. It separates the 'walker/left_eye' and
        'walker/right_eye' observables from the full input, processes them, and
        concatenates the result with the rest of the observations, which bypass
        the visual processing conv net.

        Args:
            vis_output_dim: Output dimension of the visual processing network.
                It will be concatenated with the rest of the observation, which
                just bypasses the visual processing.
        """

        super().__init__()

        # Mean and std from the "trench" task.
        self._mean = 77.
        self._std = 56.

        # Visual network.
        self._layers = [
            snt.Conv2D(output_channels=2, kernel_shape=(3, 3),
                       stride=1, padding='VALID', data_format='NHWC'),
            tf.keras.layers.ReLU(),
            snt.Conv2D(output_channels=4, kernel_shape=(3, 3),
                       stride=1, padding='VALID', data_format='NHWC'),
            tf.keras.layers.ReLU(),
            snt.Conv2D(output_channels=8, kernel_shape=(3, 3),
                       stride=2, padding='VALID', data_format='NHWC'),
            tf.keras.layers.ReLU(),
            snt.Conv2D(output_channels=16, kernel_shape=(3, 3),
                       stride=2, padding='VALID', data_format='NHWC'),
            tf.keras.layers.ReLU(),
            snt.Flatten(),
            snt.Linear(output_size=vis_output_dim),
        ]

    def __call__(self, observation):

        # Copy to prevent modifying observation in-place.
        # (the modification is done with .pop() below.)
        observation = observation.copy()

        if not hasattr(self, '_task_input'):
            # If task input is present in the observation, it will be popped
            # and concatenated at specific position in the output vector.
            self._task_input = 'walker/task_input' in observation.keys()

        # Pop eyes from `observation`.
        left_eye = tf.cast(observation.pop('walker/left_eye'), dtype=tf.float32)
        right_eye = tf.cast(observation.pop('walker/right_eye'), dtype=tf.float32)

        # If RGB, transform from RGB to 1-channel gray scale.
        if left_eye.shape[-1] == 3:  # Is RGB?
            left_eye = tf.reduce_mean(left_eye, axis=-1)
            right_eye = tf.reduce_mean(right_eye, axis=-1)
        # Normalize.
        left_eye = (left_eye - self._mean) / self._std
        right_eye = (right_eye - self._mean) / self._std
        # Stack the two eyes, shape (batch, height, width, channel=2).
        x = tf.stack((left_eye, right_eye), axis=-1)

        # Forward pass.
        for layer in self._layers:
            x = layer(x)

        if self._task_input:
            task_input = observation.pop('walker/task_input')
            # Concatenate the visual network output with the rest of
            # observations and task input.
            observation = tf2_utils.batch_concat(observation)
            out = tf.concat((task_input, x, observation), axis=-1)
        else:
            # Concatenate the visual network output with the rest of observation.
            observation = tf2_utils.batch_concat(observation)
            out = tf.concat((x, observation), axis=-1)

        return out


class TwoLevelController(snt.Module):
    """Module encapsulating the high-level and the low-level controllers."""

    def __init__(self,
                 hl_layer_sizes: Sequence[int],
                 ll_network: snt.Module,
                 ll_environment_spec,
                 steering_command_dim: int,
                 task_input_dim: int,
                 vis_output_dim: int,
                ):

        super().__init__()

        self._task_input_dim = task_input_dim
        self._vis_output_dim = vis_output_dim
        
        n_repeats = int(steering_command_dim / (3 + 4))  # This is future_steps + 1
        assert n_repeats == steering_command_dim / (3 + 4)
        steering_ballpark = n_repeats * [0., 0, 0] + n_repeats * [1., 0, 0, 0]

        # To make training possible, the HL conroller network is initialized with
        # (small) scale=0.01 so the network output, upon initialization, is close
        # to the meaningful no-op steering command with ~(0, 0, 0) displacement
        # vector and ~(1, 0, 0, 0) quaternion.
        self._hl_network = snt.Sequential([
            LayerNormMLP(
                layer_sizes=list(hl_layer_sizes) + [steering_command_dim],
                w_init=tf.initializers.VarianceScaling(
                    distribution='uniform', mode='fan_out', scale=0.01),
                activate_final=False),
            lambda x: x + tf.constant(steering_ballpark, dtype=x.dtype),
        ])
        self._ll_network = ll_network
        
        # Get steering command indices in LL-network input.
        sorted_obs_dict = observable_indices_in_tensor(
            ll_environment_spec.observations)
        ref_displacement_inds = sorted_obs_dict['walker/ref_displacement']
        ref_root_quat_inds = sorted_obs_dict['walker/ref_root_quat']
        assert ref_displacement_inds[1] == ref_root_quat_inds[0]
        self._steering_idx = ref_displacement_inds[0]

        # Make sure LL network variables are frozen.
        for variable in self._ll_network.variables:
            variable._trainable = False

    def __call__(self, x):
        """Forward pass through the two-level controller networks."""
        
        # Pass observation (with visual processed by VisNet) to HL network and
        # get steering command.
        steering = self._hl_network(x)

        if isinstance(steering, tfd.Distribution):
            steering = steering.sample()
        
        # Prepare input for LL controller.
        # Assume task_input and vis_output are the first observables in input tensor,
        # which is how the input tensor is be prepared in VisNet.
        offset = self._task_input_dim + self._vis_output_dim
        x = x[:, offset:]
        # Put steering command at its observation (ref_displacement, ref_root_quat)
        # position.
        x = tf.concat(
            (x[:, :self._steering_idx], steering, x[:, self._steering_idx:]),
            axis=-1)
        action = self._ll_network(x)
        
        return action


class LayerNormMLP(snt.Module):
    """Simple feedforward MLP torso with initial layer-norm.

    This module is an MLP which uses LayerNorm (with a tanh normalizer) on the
    first layer and non-linearities (elu) on all but the last remaining layers.
    """

    def __init__(self,
                layer_sizes: Sequence[int],
                w_init: Optional[snt.initializers.Initializer] = None,
                activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.elu,
                activate_final: bool = False):
        """Construct the MLP.

        Args:
            layer_sizes: a sequence of ints specifying the size of each layer.
            w_init: initializer for Linear weights.
            activation: activation function to apply between linear layers. Defaults
            to ELU. Note! This is different from snt.nets.MLP's default.
            activate_final: whether or not to use the activation function on the final
            layer of the neural network.
        """
        super().__init__(name='feedforward_mlp_torso')

        def _uniform_initializer():
            return tf.initializers.VarianceScaling(
                distribution='uniform', mode='fan_out', scale=0.333)

        self._network = snt.Sequential([
            snt.Linear(layer_sizes[0], w_init=w_init or _uniform_initializer()),
            snt.LayerNorm(
                axis=slice(1, None), create_scale=True, create_offset=True),
            tf.nn.tanh,
            snt.nets.MLP(
                layer_sizes[1:],
                w_init=w_init or _uniform_initializer(),
                activation=activation,
                activate_final=activate_final),
        ])

    def __call__(self, observations: types.Nest) -> tf.Tensor:
        """Forwards the policy network."""
        return self._network(tf2_utils.batch_concat(observations))
