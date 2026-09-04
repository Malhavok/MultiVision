# ADR-0005: Full Camera Calibration Workflow

**Status:** Proposed
**System:** MultiVision
**Scope:** One-shot camera/projector calibration for mixed wide and zoom cameras
**Decision type:** Calibration workflow / quality gates

## 1. Context

Some cameras intentionally observe only a small, zoomed-in part of the table –
for example, to read text on cards. Requiring every camera to see the full
projector pattern makes those cameras impossible to calibrate even though a
local mapping is useful.

The current calibration operation also treats cameras independently and
requires a separate metric-target workflow. This ADR concerns only the
projected-tag camera/projector calibration. Metric surface calibration remains
a separate operation and is not required by `full-calibration`.

## 2. Decision

Add a service-owned `full-calibration` workflow which presents one projected
pattern and analyses all available cameras from the same stable capture
window.

```text
projected pattern
       ↓
shared projector-native coordinates
       ↓
per-camera calibration
       ↓
wide or local usable region
```

The workflow must:

1. present the pattern and allow the projector/cameras to settle;
2. capture a stable burst for every available camera;
3. detect and score complete tags, corner stability and spatial coverage;
4. select the best camera as the session's **master**;
5. calibrate every camera independently from its own detected projector-tag
   correspondences;
6. perform a fresh verification burst for every accepted camera; and
7. hide the pattern and publish a per-camera result report.

The master is selected by measured quality rather than by a fixed camera name.
The default full-session gate is that the master sees at least 80% of the
pattern's unique tags, with acceptable stability, coverage, inlier ratio and
reprojection error.

The master is a quality and session-readiness reference. It does **not** repair
or mathematically alter another camera's transform. Every camera has its own
camera-native ↔ projector-native homography because each camera has different
pose, optics and distortion.

## 3. Camera acceptance levels

The workflow must distinguish global and local calibration:

- **Global calibration** – enough tags and spatial coverage to support the
  camera's observed region broadly.
- **Local calibration** – one or more complete tags support a restricted region
  around the observed tag cluster.
- **Local low-confidence calibration** – one complete tag is accepted as a
  deliberate fallback. Its four corners are mathematically sufficient for a
  homography, but they do not validate extrapolation away from the tag.

A camera seeing one full tag may therefore become usable for a nearby zoomed-in
text-reading region, but it must not silently be treated as calibrated over its
whole native frame or over the whole table. Its stored valid region and status
must make this limitation explicit.

A camera with no complete tag remains uncalibrated. A failed camera must not
invalidate independently accepted calibrations from other cameras. The command
may complete with a partial result, but must report that the session did not
meet the full master gate when applicable.

Verification is per-camera. Successful master verification is never a
substitute for verifying the other cameras, and calibration frames must not be
reused as the verification evidence.

## 4. Pattern experiment

Before selecting a new production pattern, test a candidate with approximately
twice as many tags and half the tag width and height. The change is accepted
only if real cameras demonstrate sufficient pixels per tag, complete-corner
detection, temporal stability and improved useful coverage.

Twice as many tags do not automatically compensate for four times less marker
area. The decision must be based on hardware observations, not tag count alone.
The pattern layout, marker size, dictionary and quality thresholds should remain
configuration choices where practical.

## 5. Consequences

The normal operator workflow becomes one command and one pattern presentation.
Wide cameras can establish a strong master calibration while zoom cameras can
obtain useful local mappings without seeing the full table.

The common coordinate frame is still projector-native space. This workflow does
not infer physical millimetres, tabletop axes or metric scale. Those require the
separate printed-target metric calibration defined by ADR-0002.

The implementation must add a capture-and-analysis orchestration path, fresh
per-camera verification, explicit global/local calibration statuses and a
transparent partial-result report. It must not copy a master homography into
other cameras or widen a one-tag camera's valid region by assumption.

## 6. Open implementation details

The implementation plan must settle:

- the exact tag-count, coverage and stability thresholds for each acceptance
  level;
- the representation of global versus local valid regions;
- whether a local one-tag result is enabled by default or requires an explicit
  option;
- the concrete command/API name and response schema; and
- the candidate dense-pattern layouts to test on the target hardware.
