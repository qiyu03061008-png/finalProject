Place generated or recorded voice cues in this folder.

Supported formats:

- `.wav`
- `.mp3`

Expected base file names:

- `squat_depth_not_enough`
- `torso_lean_forward`
- `knees_caving_in`
- `knees_too_far_forward`
- `pushup_depth_not_enough`
- `hip_sag`
- `hip_too_high`
- `elbows_flared`
- `plank_hip_sag`
- `plank_hip_too_high`
- `neck_not_neutral`
- `squat_keep_steady`
- `pushup_keep_steady`
- `plank_keep_steady`

Prompt text is defined in `cue_manifest.json`.

If you want to generate the files with `edge-tts`, run:

```bash
python tools/generate_audio_cues.py
```

The app will try `.wav` first and then `.mp3` for each cue.
