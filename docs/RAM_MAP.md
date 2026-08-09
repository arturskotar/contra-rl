# Contra NES RAM Map

This is a living translation table for RAM addresses used by the Contra RL environment.

Confidence levels:

- `hypothesis`: copied from old code or observed once, not proven.
- `candidate`: seems useful across a few runs, still needs controlled validation.
- `validated`: confirmed by deliberate manual tests and stable enough for reward/training.

Validation rule: no RAM address should drive important reward logic until it is at least `candidate`, and ideally `validated`.

## Current RAM translations

| Name | Address / formula | Type | Meaning | Current use | Confidence | Validation notes |
|---|---:|---|---|---|---|---|
| Player local X | `0x0334` | byte | Player horizontal position within current screen/page. | Part of absolute X. | candidate | Old project used it. New smoke/watch logs show X changes when moving right. |
| Screen/page number | `0x0064` | byte | Coarse horizontal screen/page index. | Part of absolute X. | hypothesis | Old project used it. Need verify across screen transition. |
| Horizontal scroll offset | `0x00FD` | byte | Camera/scroll offset. | Part of absolute X. | candidate | Old project used it. New preview logs show it changes during title/menu animation, so must validate during gameplay. |
| Absolute X | `ram[0x0334] + ram[0x0064] * 255 + ram[0x00FD]` | derived int | Combined horizontal progress estimate. | Progress reward, info metric, stuck detection. | candidate | Good concept from old project. Needs validation at first screen boundary. |
| Player Y | `0x031A` | byte | Player vertical position. | Info/debug only. | hypothesis | Old project used it. Needs jump/fall validation. |
| Lives | `0x0032` | byte | Remaining lives / life-like counter. | Info/debug; possible life-loss detection. | hypothesis | Old project used it. New startup logs showed odd values on title/menu, so only trust during gameplay. |
| Score info candidate | `read_digits(0x07E0, 2)` | 2-byte digit string | Score-like value exposed by old `_score` property. | Info candidate. | hypothesis | Old project used this for info, but reward used `0x07E2`. Need compare during enemy kills. |
| Score reward candidate | `read_digits(0x07E2, 2)` | 2-byte digit string | Score-like value used by old reward function. | Score reward candidate. | hypothesis | Need validate by killing first enemy and checking delta. |
| Weapon / pickup candidate | `0x00AA` | byte | Old code treated increases as weapon strength / pickup upgrade. | Possible pickup reward. | hypothesis | Need validate across weapon pickup, death, and normal movement. Do not reward yet without logs. |
| Dying flag candidate | `0x00D6 == 12` | byte flag | Old code treated value `12` as dying animation. | Death termination / death penalty candidate. | hypothesis | Need validate by intentionally dying. |
| Dead flag candidate | `0x00B4 != 0` | byte flag | Old code treated nonzero as dead. | Death termination / death penalty candidate. | hypothesis | Need validate by intentionally dying. |
| Pause/menu candidate | `0x0025 == 1` | byte flag | Old code treated this as paused/menu-ish state and tried to unpause/clear it. | Debug only. | hypothesis | Dangerous to mutate. Do not write to RAM until validated. |

## Old project reward formula

The old project effectively used:

```python
reward = 0

if previous_weapon_strength < weapon:
    reward += weapon / 10
    previous_weapon_strength = weapon

if previous_score < score:
    reward += (score - previous_score) / 1000
    previous_score = score

reward += max(0, current_x - max_x_seen) / 10

if is_dying or is_dead:
    reward -= 100
```

Time penalty existed in code, but it was commented out.

## Proposed v1 reward usage

| Reward component | Source | Formula | Status |
|---|---|---|---|
| Progress | Absolute X | `max(0, x - max_x_seen) * progress_scale` | Use now, but validate screen transition. |
| Score | Score reward candidate | `max(0, score - previous_score) * score_scale` | Add only after score RAM validation. |
| Death | Dying/dead/lives candidates | `-50` on life loss/death | Add after death validation. |
| Stuck | Absolute X | penalty + truncation after no new max X for N steps | Use after X validation. |
| Time pressure | Step counter | small fixed negative per env step | Safe. |
| Weapon/pickup | `0x00AA` candidate | positive reward on reliable upgrade increase | Later only. |

## Validation checklist

### Horizontal progress

- [ ] Start at playable level state.
- [ ] Stand still for 100 frames and confirm absolute X is stable or explain drift.
- [ ] Hold right for 300 frames and confirm absolute X increases.
- [ ] Cross first screen boundary and confirm absolute X remains monotonic.
- [ ] Move left/backtrack and confirm max-X logic avoids negative progress reward.

### Score

- [ ] Log `0x07E0`, `0x07E1`, `0x07E2`, `0x07E3`, and derived score candidates.
- [ ] Kill first enemy.
- [ ] Confirm which address/range changes.
- [ ] Confirm score does not change from movement alone.
- [ ] Confirm score reset behavior after environment reset.

### Lives / death

- [ ] Record `0x0032`, `0x00D6`, and `0x00B4` during normal gameplay.
- [ ] Intentionally die.
- [ ] Confirm which values change before, during, and after death animation.
- [ ] Decide whether termination should happen on dying animation, life decrement, or dead flag.

### Weapon / pickup

- [ ] Record `0x00AA` during normal gameplay.
- [ ] Pick up a weapon/powerup.
- [ ] Confirm whether `0x00AA` changes.
- [ ] Confirm whether it resets on death.
- [ ] Confirm whether it can decrease for reasons unrelated to losing a pickup.

## Notes

- Title/menu RAM values are noisy and should not be trusted as gameplay values.
- Avoid writing to RAM unless we know exactly what the address controls.
- Every reward component should be logged separately in `info["reward_parts"]`.
- If an address only works for one ROM revision, document the ROM hash before relying on it.

