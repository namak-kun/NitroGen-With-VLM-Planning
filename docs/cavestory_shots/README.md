# Cave Story (doukutsu-rs) eval screenshots

Frames from the headless eval harness, for assessing **asset fidelity** and NitroGen's
behavior in the loop.

| file | what it shows |
|---|---|
| `01_freeware_first_cave.png` | Clean First Cave gameplay — the **original freeware** Cave Story pixel art (Quote, classic HUD, "Start Point"). This is the asset-fidelity reference: doukutsu-rs renders the original `data/` assets faithfully (it is NOT a downgraded reimplementation). |
| `02_nitrogen_start.png` | Start of a NitroGen-in-the-loop run (First Cave). |
| `03_nitrogen_mid.png` | Mid-run. |
| `04_nitrogen_end.png` | End of the 24-step run. |
| `nitrogen_run.mp4` | Short video of the full run. |

## Asset-fidelity note
The rendering above is **original freeware** Cave Story (low-res pixel art, 4:3, classic HUD).
The visual risk for the model is **version mismatch**, not a downgraded reimplementation:
- **Freeware** (shown): original pixel art, 4:3.
- **Cave Story+** (Nicalis HD remaster on Steam/Switch): redrawn HD art, widescreen, different
  HUD — what most modern Twitch/YouTube streamers play.

If NitroGen's training footage was Cave Story+, the freeware look is off-distribution (different
art/aspect/UI), which — combined with NitroGen being near-Markov (no history) — could explain
the imperfect play (wall-pushing, down-bias) better than model incompetence. doukutsu-rs also
supports Cave Story+ / Switch data files, so the remaster look is reproducible if those assets
are available.
