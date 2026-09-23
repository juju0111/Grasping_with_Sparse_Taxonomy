# Hand assets — sources and licenses

Every hand in this folder is a MuJoCo MJCF adapted from a publicly released robot
description: the fingers, joints, actuators, inertials and collision meshes are the
vendor's; this repository adds the mocap-welded wrist, the fore-arm box, touch/contact
sensors, fingertip bodies and servo gains, and renames joints/actuators to the common
`rh_*` / `lh_*` scheme. **The code of this repository is MIT-licensed, the hand assets are
not** — each keeps the license of its upstream description, reproduced in the `LICENSE`
file next to the MJCF. Keep those files if you copy a hand elsewhere.

| Folder | Hand (vendor) | Upstream description | License | License file |
|---|---|---|---|---|
| `shadow/` | Shadow Hand E3M5 (Shadow Robot Company) | [MuJoCo Menagerie `shadow_hand`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand), URDF and meshes provided by Shadow Robot Company | Apache-2.0 | [`shadow/LICENSE`](shadow/LICENSE), notes in [`shadow/SOURCE.md`](shadow/SOURCE.md) |
| `wonik_allegro/` | Allegro Hand v3 (Wonik Robotics) | [MuJoCo Menagerie `wonik_allegro`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/wonik_allegro), derived from the SimLab / Wonik URDF | BSD-2-Clause (© 2016 SimLab) | [`wonik_allegro/LICENSE`](wonik_allegro/LICENSE) |
| `tesollo/` | DG-5F (Tesollo) | [`tesollodelto/tesollo_model`](https://github.com/tesollodelto/tesollo_model) (dg5f MuJoCo model) | BSD-3-Clause (© 2026 Tesollo) | [`tesollo/LICENSE`](tesollo/LICENSE) |
| `robotis_sh5/` | FFW SH5 hand (ROBOTIS) | [`ROBOTIS-GIT/ai_worker`](https://github.com/ROBOTIS-GIT/ai_worker) (FFW description) | Apache-2.0 | [`robotis_sh5/LICENSE`](robotis_sh5/LICENSE) |
| `wuji_hand2/` | Wuji Hand 2 beta 1 (Wuji Technology) | [`wuji-technology/wuji-description`](https://github.com/wuji-technology/wuji-description) v2026.7.23 (commit `1407bee`), `hand2/hand2_beta1/body/mjcf/right.xml` + `meshes/right/` | MIT (© 2025 Wuji Technology) | [`wuji_hand2/LICENSE`](wuji_hand2/LICENSE), notes in [`wuji_hand2/SOURCE.md`](wuji_hand2/SOURCE.md) |
| `inspire/` | RH56 (Inspire Robots) | vendor URDF, converted to MJCF — **upstream package and license to be confirmed** (the widely used [`dexsuite/dex-urdf`](https://github.com/dexsuite/dex-urdf) copy is CC BY-NC-SA 4.0) | to be confirmed | — |
| *(not included)* | ALLEX hand (WIRobotics) | [`wirobotics-rih/allex_model`](https://github.com/wirobotics-rih/allex_model): description BSD-3-Clause, but the STL meshes are evaluation-only and may not be redistributed (`MESHES-LICENSE`) | — | the hand appears in the videos only; asset, config and checkpoint are not distributed |

The Inspire row is distributed here for research use only until its upstream terms are
recorded; if you need a clean permissive license chain, drop that folder and its
`config/inspire` / `grit/util/hand_info/inspire` entries — every other hand works without it.

Object meshes live in [`../object`](../object) and have their own terms (see the README
there and the *Assets & license* section of the top-level README).
