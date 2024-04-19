## Steps to generate the MuJoCo fruit fly model

1. Starting point: raw data in `assets/`:
    * Raw fly model consisting of `drosophila.xml` and `.msh` meshes, as [exported](https://github.com/google-deepmind/dm_control/tree/main/dm_control/blender/mujoco_exporter) from [Blender](https://www.blender.org/).
    * Defaults file `drosophila_defaults.xml`.
2. Run `fuse_fruitfly.py`, it will create `assets/drosophila_fused.xml`.
3. Run `make_fruitfly.py`, it will create `fruitfly.xml`, a final fly model MJCF.
4. Copy the generated `fruitfly.xml` over to `flybody/fruitfly/assets` where the expected `.obj` meshes are. This is also the "official" directory for the final fly MJCF.
5. Your fruit fly `flybody/fruitfly/assets/fruitfly.xml` is ready to go!
