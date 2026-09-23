# Object assets — sources and licenses

All meshes are stored in the GraspXL preprocessing format (`top_watertight_tiny` /
`bottom_watertight_tiny` watertight halves, `keypoint_*.npy`, URDF, `obj_only/` MJCF).
**None of them is covered by the repository's MIT license**; each folder keeps the terms of
its source below.

| Folder | Objects | Source | License / terms |
|---|---|---|---|
| `new_training_set/002_*` … `061_*` (19 objects, numeric YCB ids) | YCB Object and Model Set, processed by the [GraspXL](https://github.com/zdchan/graspxl) pipeline | [YCB](https://www.ycbbenchmarks.com/object-models/) | YCB models: [CC BY 4.0](https://registry.opendata.aws/ycb-benchmarks/); GraspXL processing: CC BY-NC 4.0 |
| `new_training_set/` other 15 folders (`mouse`, `hammer`, `car_down`, `gun_functional`, `brush_functional`, `loopy_head_side`, `off_water_body`, `blue_pitcher`, `fan_small_head`, `big_tape`, `small_tape`, `small_wood_block`, `*_oriented`) | processed with the same pipeline — **mesh origin to be confirmed** | — | to be confirmed |
| `mixed_train/` (49 objects; ShapeNet hashes such as `Basket_121cf1ae…`, PartNet-Mobility ids such as `Knife_115`, `Earphone_10112`) | [RobustDexGrasp](https://github.com/zdchan/RobustDexGrasp) training objects | [ShapeNet](https://shapenet.org/terms), [PartNet-Mobility](https://sapien.ucsd.edu/about) | RobustDexGrasp release: CC BY-NC 4.0; ShapeNet and PartNet-Mobility terms: **non-commercial research and education only** |
| `base_table*.xml` | table / floor MJCF | this repository | MIT |

If you need a commercially clean subset, keep only the 19 YCB folders (attribute YCB) and
the table; the training configs accept any object list through `obj_idxs`.
