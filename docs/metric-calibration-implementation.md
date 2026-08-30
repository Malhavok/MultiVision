# Plan4 metric-calibration implementation boundary

This document records the implementation boundary for Plan4. The governing
architecture and geometry decisions remain in [ADR-0001](ADR-0001_%20MultiVision%20MVP0%20Architecture%20and%20Implementation%20Contract.md),
[ADR-0002](ADR-0002_%20Metric%20Surface%20Calibration.md) and
[ADR-0003](ADR-0003_%20Physical%20Geometry%20Overlays.md); this document does
not amend any ADR. The implementation observables are defined by the [shared
validation contract](../plans/plan4-validation-contract.md).

## Shared metric authority

Plan4 creates exactly one session-local metric calibration for the active
projector and physical tabletop. It is owned by `MultiVisionService`, not by
a camera slot:

```text
selected calibrated camera
  → existing per-camera camera-native → projector-native transform
  → shared projector-native ↔ surface-mm transform pair
  → service-owned ruler geometry
  → projector-native renderer
```

The record stores both directions explicitly:

```text
projector-native → surface-mm
surface-mm → projector-native
```

The inverse is calculated and validated when the shared calibration is
published. No camera owns a metric transform, `mm_per_pixel` value or
camera-specific physical scale. The existing per-camera camera/projector
calibration remains unchanged and is only an input to metric calibration.

Applicability is tied to one authoritative session projector descriptor,
including resolution and logical output identity. A descriptor change makes
camera and metric geometry unusable together and removes the ruler. Metric
state is not persisted or trusted across process restarts.

## Target and validation boundary

The printable calibration target is deterministic A4 portrait at exactly
210 × 297 mm. It carries its format/version and supported marker-family
metadata, known surface-mm marker corners, an orientation cue and a labelled
exactly-100-mm reference segment. The generated artifact must instruct the
operator to print at 100% / Actual size and warn against scaling; a generated
file or synthetic test is not evidence of physical printer accuracy.

Quality and lifecycle rules fail closed. A metric transform is usable only
when it is finite, structurally valid, applicable to the current projector
descriptor and in the configured quality state. Unidentified or unreliable
target detections, invalid correspondences, insufficient coverage, bad fit,
invalid inverses, horizon crossings and out-of-bounds projector geometry are
rejected rather than approximated. The observable states are `UNCALIBRATED`,
`CALIBRATED` and `STALE`; unavailable or stale state never supplies a ruler
transform.

Plan4 includes only the minimal validation ruler: two finite surface-mm
endpoints, `mm`/`cm`/`in` output units, the requested physical length,
projected endpoints and deterministic tick/marker primitives. A physical
measurement may be recorded separately as requested length, observed length
and absolute error in millimetres. Fit residuals are calibration diagnostics,
not physical-validation evidence, and physical accuracy is never inferred
from them.

## Explicit exclusions

Plan4 does not change per-camera camera/projector calibration, introduce
camera-owned metric state, or create a second camera/projector calibration.
It does not implement ADR-0003 gameplay, grids, circles, generic lines,
generic polygons, a generic overlay registry, movement or line-of-sight rules.
The existing point overlay remains separate; Plan4 adds only the session-local
metric ruler needed to validate the shared transform.

Every Plan4 implementation task is checked against the [shared validation
contract](../plans/plan4-validation-contract.md). Deterministic
checks establish software and mathematical behaviour only. Hardware checks
must separately record printer, projector, tabletop, camera and physical
ruler observations; this document claims none of those checks have passed.
