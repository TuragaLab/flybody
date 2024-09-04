"""Arenas for the fruitfly walker."""

import numpy as np
from scipy import ndimage

from dm_control import composer
from dm_control.locomotion.arenas import assets as locomotion_arenas_assets
from dm_control.mujoco.wrapper import mjbindings

mjlib = mjbindings.mjlib


def pos_to_terrain_idx(x, y, size, nrow, ncol):
    """Return index in `terrain` array corresponding to world position x, y."""
    idx_y = int((y / size[0]) * (nrow / 2) + nrow / 2)
    idx_x = int((x / size[1]) * (ncol / 2) + ncol / 2)
    return idx_x, idx_y


def terrain_bowl(physics,
                 bump_scale=2.,
                 elevation_z=4.,
                 tanh_rel_radius=0.7,
                 tanh_sharpness=8.,
                 random_state=None):
    """Generate a bowl-shaped terrain.

    Args:
        physics: Current physics instance.
        bump_scale: Spatial extent of bumps.
        elevation_z: Returned terrain will be normalized between [0, elevation_z].
        tanh_rel_radius: Radius of bowl, relative to half-length of arena.
        tanh_sharpness: Sharpness of tanh.

    Returns:
        terrain (nrow, ncol).
    """
    size = physics.model.hfield_size[0, :2]  # half-lengths! (e.g., radius)
    nrow = physics.model.hfield_nrow[0]
    ncol = physics.model.hfield_ncol[0]

    # Fill arena with bumps, normalized between [0, elevation_z].
    assert nrow == ncol
    bump_res = int(2 * size[0] / bump_scale)
    bumps = random_state.uniform(0, 1, (bump_res, bump_res))
    terrain = ndimage.zoom(bumps, nrow / float(bump_res))
    terrain -= np.min(terrain)
    terrain /= np.max(terrain)
    terrain *= elevation_z

    # Add tanh-shaped bowl.
    axis = np.linspace(-1, 1, terrain.shape[0])
    xv, yv = np.meshgrid(axis, axis)
    r = np.sqrt(xv**2 + yv**2)
    bowl_shape = 0.5 * np.tanh(tanh_sharpness *
                               (r - tanh_rel_radius)) + 0.5  # Between 0 and 1.
    terrain *= bowl_shape

    return terrain  # Normalized between [0, elevation_z].


def add_sine_bumps(terrain, arena_size, wavelength=5., phase=0., height=1.):
    """Add sine-like bumps to an existing terrain.

    Args:
        terrain: Initial terrain, (nrow, ncol).
        arena_size: Half-length (aka radius) of arena, shape (2,).
        wavelength: Wavelength of bumps, in actual world length units.
        phase: Phase of sine.
        height: Amplitude of bumps, in actual world length units.

    Returns:
        terrain: Initial terrain with added sine bumps, (nrow, ncol).
    """
    _, ncol = terrain.shape
    x_axis = np.linspace(-arena_size[0], arena_size[0], ncol)
    bumps = height * 0.5 * (np.sin(2 * np.pi / wavelength * x_axis + phase) +
                            1)
    terrain = terrain.copy()
    terrain[:, :] = np.maximum(bumps, terrain)
    return terrain


def add_sine_trench(terrain,
                    arena_size,
                    wavelength=5,
                    phase=0.,
                    amplitude=1.,
                    start_x=0,
                    end_x=10.,
                    width=1.,
                    height=1.,
                    sigma=0.2):
    """Add sine-shaped trench to terrain.

    Args:
        terrain: Initial terrain, (nrow, ncol).
        arena_size: Half-lengths (aka radius) of arena, cm, shape (2,).
        wavelength: Sine wavelength, in cm.
        amplitude: Sine amplitude, in cm.
        phase: Sine phase, rad.
        start_x: x of trench entrance, cm.
        end_x: x of trench end, cm.
        width: Width of trench before smoothing, cm.
        height: Height of trench, cm.
        sigma: Terrain smoothing stddev, in cm.

    Returns:
        terrain: Initial terrain with added sine trench, (nrow, ncol).
        sine: Trench sine, not used in the task but can be used for analysis etc.
    """
    nrow, ncol = terrain.shape
    idx_from, _ = pos_to_terrain_idx(start_x, 0, arena_size, nrow, ncol)
    idx_to, _ = pos_to_terrain_idx(end_x, 0, arena_size, nrow, ncol)
    delta, _ = pos_to_terrain_idx(-arena_size[0] + width / 2, 0, arena_size,
                                  nrow, ncol)
    x_axis = np.linspace(0, end_x - start_x, idx_to - idx_from + 1)
    sine = amplitude * np.sin(2 * np.pi / wavelength * x_axis + phase)
    sine -= sine[0]  # Center tunnel entrance at zero.
    trench = np.zeros_like(terrain)
    trench[:, idx_from:idx_to] = height
    for idx_x in range(idx_from, idx_to):
        _, idx_y = pos_to_terrain_idx(0, sine[idx_x - idx_from], arena_size,
                                      nrow, ncol)
        trench[idx_y - delta:idx_y + delta + 1, idx_x] = 0.
    # Smoothing.
    trench = ndimage.gaussian_filter(trench, sigma=sigma / width * delta * 2)
    # Combine trench with existing terrain.
    terrain = np.maximum(trench, terrain)
    return terrain, sine


class Hills(composer.Arena):
    """A hilly arena.

    Args:
        name: Name of the arena.
        dim (tuple or int): Half-length of the actual arena (this is the `radius`).
            If a tuple is provided, then it's (radius_x, radius_y).
        aesthetic: Aesthetic of the arena.
        hfield_elevation_z, hfield_base_z: hfield asset parameters.
        grid_density (tuple or int): number of hfield grid points per unit length
            of the actual floor. For example, if grid density == 10, the number of
            hfield grid points in 1x1 square of actual floor is 100.
            If a tuple is provided, it's (density_x, density_y).
        elevation_z_range: Range of elevation of horizon mountains.
    """

    def _build(self,
               name='hills',
               dim=(20, 20),
               aesthetic='outdoor_natural',
               hfield_elevation_z=1,
               hfield_base_z=0.05,
               grid_density=(10, 10),
               elevation_z_range=(4., 5.)):
        super()._build(name=name)

        if isinstance(dim, tuple):
            # Potentially rectangular arena.
            size = dim
        else:
            # Square arena.
            size = (dim, dim)
        if not isinstance(grid_density, tuple):
            grid_density = (grid_density, grid_density)
        self._elevation_z_range = elevation_z_range

        self._hfield = self._mjcf_root.asset.add(
            'hfield',
            name='terrain',
            nrow=((2 * grid_density[1] * size[1]) // 2) * 2 + 1,
            ncol=((2 * grid_density[0] * size[0]) // 2) * 2 + 1,
            size=size + (hfield_elevation_z, hfield_base_z))

        if aesthetic != 'default':
          
            ground_info = locomotion_arenas_assets.get_ground_texture_info(
                aesthetic)
            sky_info = locomotion_arenas_assets.get_sky_texture_info(aesthetic)
            texturedir = locomotion_arenas_assets.get_texturedir(aesthetic)
            self._mjcf_root.compiler.texturedir = texturedir

            self._texture = self._mjcf_root.asset.add('texture',
                                                      name='aesthetic_texture',
                                                      file=ground_info.file,
                                                      type=ground_info.type)
            self._material = self._mjcf_root.asset.add(
                'material',
                name='aesthetic_material',
                texture=self._texture,
                texuniform='true')
            self._skybox = self._mjcf_root.asset.add(
                'texture',
                name='aesthetic_skybox',
                file=sky_info.file,
                type='skybox',
                gridsize=sky_info.gridsize,
                gridlayout=sky_info.gridlayout)
            self._terrain_geom = self._mjcf_root.worldbody.add(
                'geom',
                name='terrain',
                type='hfield',
                pos=(0, 0, -0.01),
                hfield='terrain',
                material=self._material)
            self._ground_geom = self._mjcf_root.worldbody.add(
                'geom',
                type='plane',
                name='groundplane',
                size=list(size) + [0.5],
                material=self._material)

        else:

            self._ground_texture = self._mjcf_root.asset.add(
                'texture',
                rgb1=[.2, .3, .4],
                rgb2=[.1, .2, .3],
                type='2d',
                builtin='checker',
                name='groundplane',
                width=200,
                height=200,
                mark='edge',
                markrgb=[0.8, 0.8, 0.8])
            self._ground_material = self._mjcf_root.asset.add(
                'material',
                name='groundplane',
                texrepeat=[2, 2],  # Makes white squares exactly 1x1 length units.
                texuniform=True,
                reflectance=2,
                texture=self._ground_texture)

            self._terrain_geom = self._mjcf_root.worldbody.add(
                'geom',
                name='terrain',
                type='hfield',
                hfield='terrain',
                material=self._ground_material)
            self._ground_geom = self._mjcf_root.worldbody.add(
                'geom',
                type='plane',
                name='groundplane',
                pos=(0, 0, -0.01),
                rgba=(0.2, 0.3, 0.4, 1),
                size=list(size) + [0.5])

        self._mjcf_root.visual.headlight.set_attributes(ambient=[.4, .4, .4],
                                                        diffuse=[.8, .8, .8],
                                                        specular=[.1, .1, .1])

        self._regenerate = True

    def regenerate(self, random_state):
        # Regeneration of the bowl requires physics, so postponed to initialization.
        self._regenerate = True

    def initialize_episode(self, physics, random_state):
        if self._regenerate:
            self._regenerate = False

            # Create bowl arena.
            # Elevation of horizon mountains.
            elevation_z = random_state.uniform(*self._elevation_z_range)
            terrain = terrain_bowl(physics,
                                   elevation_z=elevation_z,
                                   random_state=random_state)

            start_idx = physics.bind(self._hfield).adr
            res = physics.bind(self._hfield).nrow
            physics.model.hfield_data[start_idx:start_idx +
                                      res**2] = terrain.ravel()

            # If we have a rendering context, we need to re-upload the modified
            # heightfield data.
            if physics.contexts:
                with physics.contexts.gl.make_current() as ctx:
                    ctx.call(mjlib.mjr_uploadHField, physics.model.ptr,
                             physics.contexts.mujoco.ptr,
                             physics.bind(self._hfield).element_id)

    @property
    def ground_geoms(self):
        """Returns the geoms that make up the ground."""
        return (self._terrain_geom, self._ground_geom)


class SineTrench(Hills):
    """A hilly arena.

    Args:
        name: Name of the arena.
        dim: Half-length of the actual arena (this is the `radius`.)
        aesthetic: Aesthetic of the arena.
        hfield_elevation_z=1, hfield_base_z: hfield asset parameters.
        grid_density: number of hfield grid points per unit length of the actual
            floor. For example, if grid density == 10, the number of hfield grid
            points in 1x1 square of actual floor is 100.
        elevation_z_range: Range of elevation of horizon mountains.
        start_offset_range: Range of x-offset of trench entrance.
        trench_len_range: Range of trench length.
        phase_range: Range of sine phase.
        wavelength_range: Range of sine wavelength.
        amplitude_range: Range of sine amplitude.
        width_range: Range of trench width (see implementation how it's calculated).
        height_range: Range of trench height.
        sigma_range: Range of terrain smoothing stddev.
    """

    def _build(self,
               name='sine_trench',
               dim=20,
               aesthetic='outdoor_natural',
               hfield_elevation_z=1,
               hfield_base_z=0.05,
               grid_density=10,
               elevation_z_range=(4., 5.),
               start_offset_range=(-5., -3.),
               trench_len_range=(4., 10.),
               phase_range=(0., 2 * np.pi),
               wavelength_range=(5., 8.),
               amplitude_range=(0.7 / 2, 1.2 / 2),
               width_range=(0.5, 1),
               height_range=(1.3, 1.3),
               sigma_range=(0.2, 0.2)):

        super()._build(dim=dim,
                       aesthetic=aesthetic,
                       name=name,
                       hfield_elevation_z=hfield_elevation_z,
                       hfield_base_z=hfield_base_z,
                       grid_density=grid_density,
                       elevation_z_range=elevation_z_range)

        self._start_offset_range = start_offset_range
        self._trench_len_range = trench_len_range
        self._phase_range = phase_range
        self._wavelength_range = wavelength_range
        self._amplitude_range = amplitude_range
        self._width_range = width_range
        self._height_range = height_range
        self._sigma_range = sigma_range
        self._trench_specs = None
        self._regenerate = None

    def initialize_episode(self, physics, random_state):
        if self._regenerate:
            self._regenerate = False

            # Create bowl arena.
            elevation_z = random_state.uniform(*self._elevation_z_range)
            bowl = terrain_bowl(physics,
                                elevation_z=elevation_z,
                                random_state=random_state)
            size = physics.model.hfield_size[0, :2]

            # Add sine trench.
            start_x = random_state.uniform(*self._start_offset_range)
            end_x = start_x + random_state.uniform(*self._trench_len_range)
            # Make sure the choice of amplitude and width don't allow for "trivial"
            # straight fly-through solution. 0.604 is the fly model wing span.
            amplitude = random_state.uniform(*self._amplitude_range)
            width = 2 * amplitude + 0.604 * random_state.uniform(
                *self._width_range)
            terrain, sine = add_sine_trench(
                bowl,
                size,
                start_x=start_x,
                end_x=end_x,
                phase=random_state.uniform(*self._phase_range),
                wavelength=random_state.uniform(*self._wavelength_range),
                amplitude=amplitude,
                width=width,
                height=random_state.uniform(*self._height_range),
                sigma=random_state.uniform(*self._sigma_range))
            # True-coordinate x-axis of trench, cm.
            trench_x = np.linspace(start_x, end_x, sine.shape[0])

            self._trench_specs = {'x_coords': trench_x, 'y_coords': sine}

            start_idx = physics.bind(self._hfield).adr
            res = physics.bind(self._hfield).nrow
            physics.model.hfield_data[start_idx:start_idx +
                                      res**2] = terrain.ravel()

            # If we have a rendering context, we need to re-upload the modified
            # heightfield data.
            if physics.contexts:
                with physics.contexts.gl.make_current() as ctx:
                    ctx.call(mjlib.mjr_uploadHField, physics.model.ptr,
                             physics.contexts.mujoco.ptr,
                             physics.bind(self._hfield).element_id)

    @property
    def trench_specs(self):
        """Returns the specs of the trench."""
        return self._trench_specs


class SineBumps(Hills):
    """A hilly arena with sinusoidal bumps.

    Args:
        name: Name of the arena.
        dim: Half-length of the actual arena (this is the `radius`.)
        aesthetic: Aesthetic of the arena.
        hfield_elevation_z=1, hfield_base_z: hfield asset parameters.
        grid_density: number of hfield grid points per unit length of the actual
            floor. For example, if grid density == 10, the number of hfield grid
            points in 1x1 square of actual floor is 100.
        elevation_z_range: Range of elevation of horizon mountains.
        phase_range: Range of sine phase.
        wavelength_range: Range of sine wavelength.
        height_range: Range of sine amplitude.
    """

    def _build(self,
               name='sine_bumps',
               dim=20,
               aesthetic='outdoor_natural',
               hfield_elevation_z=1,
               hfield_base_z=0.05,
               grid_density=10,
               elevation_z_range=(4., 5.),
               phase_range=(0., 2 * np.pi),
               wavelength_range=(10., 15.),
               height_range=(0.5, 1.0)):

        super()._build(dim=dim,
                       aesthetic=aesthetic,
                       name=name,
                       hfield_elevation_z=hfield_elevation_z,
                       hfield_base_z=hfield_base_z,
                       grid_density=grid_density,
                       elevation_z_range=elevation_z_range)

        self._phase_range = phase_range
        self._wavelength_range = wavelength_range
        self._height_range = height_range

    def initialize_episode(self, physics, random_state):
        if self._regenerate:
            self._regenerate = False

            # Create bowl arena.
            elevation_z = random_state.uniform(*self._elevation_z_range)
            bowl = terrain_bowl(physics,
                                elevation_z=elevation_z,
                                random_state=random_state)
            size = physics.model.hfield_size[0, :2]

            # Add sine trench.
            terrain = add_sine_bumps(
                bowl,
                size,
                wavelength=random_state.uniform(*self._wavelength_range),
                phase=random_state.uniform(*self._phase_range),
                height=random_state.uniform(*self._height_range))

            start_idx = physics.bind(self._hfield).adr
            res = physics.bind(self._hfield).nrow
            physics.model.hfield_data[start_idx:start_idx +
                                      res**2] = terrain.ravel()

            # If we have a rendering context, we need to re-upload the modified
            # heightfield data.
            if physics.contexts:
                with physics.contexts.gl.make_current() as ctx:
                    ctx.call(mjlib.mjr_uploadHField, physics.model.ptr,
                             physics.contexts.mujoco.ptr,
                             physics.bind(self._hfield).element_id)
