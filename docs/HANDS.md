# Hands

| `using_hand_name_list` | hand | actuators | MJCF | notes |
|---|---|---|---|---|
| `tesollo` | Tesollo DG-5F | 20 | `tesollo/tesollo_5g_w_fore_arm_touch_contact.xml` | |
| `robotis_sh5` | ROBOTIS FFW SH5 | 20 | `robotis_sh5/ffw_sh5_rh_w_fore_arm_touch_contact.xml` | `hand_info/robotis` |
| `allegro` | Wonik Allegro | 16 | `wonik_allegro/allegro_rh_w_fore_arm_touch_contact_kp20.xml` | 4 fingers |
| `inspire` | Inspire RH56 | 6 | `inspire/inspire_rh_w_fore_arm_touch_contact_kp20.xml` | under-actuated, coupled joints |
| `shadow` | Shadow Dexterous Hand | 18 | `shadow/right_hand_w_fore_arm_touch_contact.xml` | |
| `wuji_hand2` | Wuji Hand 2 | 20 | `wuji_hand2/wuji_hand2_rh_w_fore_arm_touch_contact.xml` | |

(Actuator counts are printed at env build as `action = 9 + n_ctrl`.) Only right hands are shipped.

## What a hand needs

Every hand is described in three places:

1. **MJCF + meshes** — `asset/dextrous_hand/<dir>/…`. Conventions the env relies on:
   * the palm is a free body welded to a mocap body (`<hand>_rh_mocap`) — the wrist action
     moves the mocap;
   * position actuators on the finger joints (`kp`, `forcerange`; `finger_gain` in the hand
     yaml can override them at build time);
   * one `touch` sensor site per finger link + `contact` sensors on the same sites (names
     ending in `_touch` / `_contact`), a fore-arm sensor, and `*_end` fingertip bodies;
   * geom groups: 0 floor · 1 collision+visual · 2 visual · 3 collision-only.
2. **Hand yaml** — `config/<hand>/<hand>_config.yaml` (asset path, wrist body name, gains)
   and `config/<hand>/grit_right.yaml` (Hydra entry that `HandUtils` composes).
3. **`hand_info/<pkg>/`** — `rh_info.py` (site / body / actuator name tables, finger groups,
   hand-centre frame, contact weights) and `<hand>_taxonomy_annotation.py` (finger joint
   targets and active-finger masks for each grasp taxonomy class used as the mimic /
   finger-mask targets). Register the `hand_name → pkg` pair in `HandUtils.__init__`
   (`grit/util/hand_utils.py`).

Then `config/training/grit_<hand>.yaml` (env-only) and any training yaml with
`using_hand_name_list: [<hand>]`. The observation and action sizes adapt automatically; the
same reward weights worked for every hand above without per-hand tuning (only
`finger_gain.forcerange` differs per hand).
