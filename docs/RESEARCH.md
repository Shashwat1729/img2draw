# Research Notes

**Question.** Can an AI automatically infer a compact, semantically
meaningful, editable, replayable sequence of drawing operations from a
finished raster image such that rendering it faithfully reconstructs the
source?

**Stance.** The drawing representation is the means; faithful
reconstruction is the hard constraint. Complexity regularization must never
silently degrade reconstruction.

**What is genuinely new here.** Not "combining GitHub repos" — a working
*inverse drawing representation*: hybrid layers (vector structure + raster
shading splits), an explicit signed-residual correction cascade guaranteeing
quantitative fidelity, and an editable/replayable project format with a
deterministic renderer and measured evaluation.

**Baselines to compare against** (see docs/BASELINES.md): VTracer,
DiffVG-optimization, LayerTracer, COVec, naive raster-to-SVG, raster baseline.

**Fidelity vs complexity.** Modes trade the two. In `high_fidelity`, the
correction cascade is always fitted to the remaining error — fidelity is
never sacrificed for fewer ops.
