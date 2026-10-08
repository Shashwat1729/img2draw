# Animation

The project format is frame-agnostic. Per-frame overrides live under
`project["frames"][str(t)]` and are applied over shared layers/operations,
so objects keep identity across frames instead of being rebuilt
(`animation.project_for_frame`).

`animation.temporal_consistency(frames_a, frames_b)` reports mean/max pixel
drift between corresponding frames — the seed of the flicker/consistency
evaluation.

Not yet implemented: tracking, frame-consistency optimization, video export.
The schema leaves room: shared objects/layers/strokes + per-frame deltas.
