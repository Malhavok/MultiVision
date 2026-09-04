# Plan7 / ADR-0004 implementation boundary

This document records the implementation boundary for ADR-0004. ADR-0004 and
ADRs 0001–0003 remain authoritative; this document does not amend them. The
runtime remains game-agnostic, session-owned, and based on the existing camera
→ projector and metric calibration authorities.

## Group configuration

`fiducial_groups` in the JSON file provides optional startup defaults, but group
definitions are session-local and may be added, replaced or removed at runtime
without restarting or recalibrating. The runtime API is `PUT
/fiducial-groups/{group_name}` and `DELETE /fiducial-groups/{group_name}`; the
CLI equivalents are `fiducial-groups set` and `fiducial-groups remove`.

`fiducial_groups` is a JSON object keyed by a non-empty namespace. Every value
has **exactly** these two fields:

```json
{
  "cards": {
    "dictionary": "DICT_5X5_1000",
    "marker_size_mm": 38.0
  }
}
```

`dictionary` uses the canonical Plan6 detector dictionary validation and
`marker_size_mm` is positive and finite. An empty object is valid and means
that no startup namespace is configured. There is no default group, no
per-group default dictionary or marker size, and no ID-only fallback. A
fiducial identity is always `(group, id)`, so equal numeric IDs in different
groups remain distinct. Runtime registration still requires the dictionary and
physical marker size explicitly; metric calibration supplies the coordinate
transform, but does not silently infer or persist those group semantics. Unknown fields inside a group definition are rejected;
unrelated top-level configuration fields continue to be preserved when the
configuration file is updated.

The new validated startup values are:

| Field | Default | Accepted values |
| --- | ---: | --- |
| `fiducial_history_length` | `8` | positive integer, at most `32` |
| `fiducial_tracking_rate_hz` | `30.0` | finite `1.0..60.0` |
| `fiducial_grace_period_seconds` | `5.0` | finite `0.1..60.0` |
| `fiducial_protection_margin_mm` | `5.0` | finite `0.1..1000.0` |
| `max_batch_operations` | `100` | positive integer, at most `1000` |
| `preview_mode` | `active` | `active`, `low_rate` or `off` |
| `preview_low_rate_hz` | `10.0` | finite `1.0..15.0` |

The five-second grace period is the user-approved default and is measured from
the last usable observation. It is not shortened by frame rate or preview
settings. Preview values are read once at process startup. There is deliberately
no runtime, API or CLI preview-mode mutation route. `tag_dictionary` remains
the independent Plan6 inspection setting, with its existing default and
canonical validation.

## Spatial selection and freshness

The tracker publishes immutable copy-on-write snapshots. History is bounded by
`fiducial_history_length` for each `(group, id, camera slot)`. A usable dynamic
observation requires usable metric calibration because stability is measured in
`surface_mm`; camera capture timestamps are diagnostic only and do not control
expiry when their clock can jump.

For a candidate with at least two retained centres, its stability score is the
arithmetic mean of Euclidean displacement between consecutive retained
`surface_mm` centres. Fewer than two samples gives an infinite score. For each
compound identity, choose the usable candidate with the lowest score. Break
ties by most recent received monotonic timestamp, then lowest stable session
slot ID, then highest frame counter. If every current candidate is unwarmed,
use the deterministic fallback of most recent received monotonic timestamp,
then lowest stable session slot ID, then highest frame counter. A current usable
observation beats a retained grace observation.

When all current observations disappear, the last selected pose is retained
without prediction for exactly the configured grace period. It then becomes
unresolved/hidden, while the declarative overlay remains stored. A newly usable
observation resolves that same overlay automatically. No temporal averaging,
Kalman filter or motion prediction is part of this boundary.

## Anchors and arrows

Anchors are explicit discriminated JSON objects. The supported forms are:

```json
{"type": "surface", "x": 420.0, "y": 180.0, "unit": "mm"}
{"type": "projector", "x": 800.0, "y": 400.0, "unit": "px"}
{"type": "fiducial", "group": "cards", "id": 17}
```

A fiducial anchor may additionally contain a metric marker-local offset and
rotation following:

```json
{
  "type": "fiducial",
  "group": "cards",
  "id": 17,
  "local_offset": {"x": 20.0, "y": 0.0, "unit": "mm"},
  "follow_rotation": true
}
```

The fiducial group and non-negative integer ID are mandatory. Coordinates,
offsets and angles are finite; units are explicit. Existing Plan5
`projector_px`, `camera_px` and `surface_mm` point references remain supported.
Metric offsets are applied in the marker's `surface_mm` frame and then passed
through the existing metric authority. A local offset or `follow_rotation`
requires usable metric calibration. Unknown groups or IDs, missing authorities,
unusable observations and invalid geometry fail closed; temporary absence is an
unresolved runtime state rather than deletion of the request.

An arrow request is declarative and contains `kind: "arrow"`, independent
`start` and `end` anchors, a style, positive bounded `head_length` and
`head_width` in the declared geometry space, and an optional label. Its two
anchors are resolved independently, including surface/projector/fiducial
combinations and fiducials from different groups. Materialisation consists of
one shaft segment and one triangle head. A zero-length resolved arrow is
explicitly rejected or suppressed and never produces NaN geometry. Physical
sizes use millimetres and the existing metric transform, not a pixel-scale
shortcut. Repeated resolutions never mutate the stored request.

## Registry and publication

The registry stores immutable normalised requests, not historical projector
coordinates. Dynamic requests remain listed while a marker is in grace and
after expiry. Listing can distinguish static and dynamic entries and report
anchor definitions, visibility, resolution state and freshness.

`POST /overlays/batch` accepts ordered `create`, `update` and `remove`
operations. Create supplies a complete request and UUID. Update supplies the
target UUID and a complete replacement with the same UUID. Remove supplies the
target UUID. The service applies operations in order to a private candidate
registry, allowing create-then-update/remove in one batch, then publishes once
only after all schema, dependency, bounded-work and static-geometry checks
succeed. Any invalid selector, duplicate or unknown ID, invalid specification,
unsafe geometry or over-limit batch rejects the whole batch with no registry,
render or visibility change. Updates retain insertion order, and
`max_batch_operations` is the single bounded operation limit. Legacy single
overlay routes use the same candidate machinery.

## Presentation and render ownership

Each projector frame consumes one immutable complete render snapshot. The
service owns one registry snapshot, one spatial snapshot, the active projector
descriptor, resolved projector-native primitives, legacy Plan3–Plan6 layers,
calibration/blank-capture suppression, global intensity, protected regions and
the camera preview/status snapshot. Snapshot assembly uses authority version
checks and bounded retry or discard behaviour, so it cannot mix generations.
Resolution and materialisation may happen outside a short publication lock.
The renderer only draws service-produced projector-native output: it does not
capture cameras, detect fiducials, calculate homographies, convert units,
resolve anchors or size shapes.

Global overlay intensity and each generic style intensity are bounded to
`0.0..1.0`; their product changes colour/opacity only. It never changes source
geometry, clipping dimensions or requested physical sizes. The global control
also applies to legacy point, area and ruler layers. Calibration patterns and
fiducial protection masks are not dimmed, and this setting is not a claim about
projector lamp brightness.

Every selected or retained usable fiducial creates a projector-space footprint
from its observed calibrated corners. When metric calibration is usable, the
footprint is expanded by `fiducial_protection_margin_mm` in surface space and
mapped back through the existing metric transform. Without metric calibration,
the unexpanded calibrated projector footprint is protected; no pixel-scale
margin is invented. Ordinary overlay presentation is clipped or suppressed in
protected regions without changing its requested geometry, so a crossing line
may be discontinuous. Protection starts with the usable observation and does
not wait for marker loss. Existing calibration-pattern and metric blank-capture
exclusivity remains authoritative.

## Preview, performance and claims

Capture, retained-frame publication, tracking, calibration validity, projector
rendering and API mutations are independent of diagnostic preview. `active`
uses the normal preview cadence, `low_rate` limits only preview conversion,
scaling and blitting, and `off` performs no preview conversion or blit. Preview
is reduced or disabled for presentation cost, not removed as a diagnostic
capability.

The realtime requirement is for a running service: at least 30 accepted and
published overlay state mutations per second while capture, tracking,
projector rendering and normal preview are active, without a mutation-caused
multi-frame projector stall. Benchmarks report HTTP, validation,
mutation/publication, spatial resolution/materialisation, presentation and CPU
components separately. CLI cold-start latency is measured separately. A pure
registry microbenchmark, or spawning 30 CLI processes, is not evidence of the
running-service target.

Synthetic tests establish software and mathematical behaviour only. Hardware
claims are manual-only: physical marker readability, projector/camera
coexistence, tracking across cameras, preview-mode operation on the target
setup and measured running-service performance must be observed and recorded
on the actual hardware. Unperformed checks remain unclaimed.

The project-owned benchmark is run against an already-running service. The
following command emits the deterministic smoke schema only and cannot pass
the realtime target:

```sh
.venv/bin/python benchmarks/benchmark_realtime_overlays.py --injected
```

For manual evidence, start three otherwise identical services with startup
preview modes `active`, `low_rate` and `off`, then drive all three endpoints in
one report. The service must have capture, tracking and projector presentation
active; the benchmark does not reduce those paths or change preview mode at
runtime:

```sh
.venv/bin/python benchmarks/benchmark_realtime_overlays.py \
  --service-url http://127.0.0.1:8000 --preview-mode active \
  --mode-url low_rate=http://127.0.0.1:8001 \
  --mode-url off=http://127.0.0.1:8002 \
  --samples 30 --warmup-requests 2 \
  > benchmark-realtime-overlays.json
```

The report keeps single-mutation accepted/published rate separate from atomic
batch object throughput and records CLI process startup under `cold_start` for
context only. A real-service result with missing diagnostics, inactive runtime
paths or presentation stalls is evidence against the acceptance criterion,
not a passing microbenchmark.

## ADR-0004 non-goals

This implementation does not add:

- detection of all rectangles, boards or board cells from camera imagery;
- generic contour or object recognition;
- game rules;
- identification of miniatures without explicit fiducials;
- path planning;
- motion prediction or Kalman filtering;
- persistence of overlay scenes across service restart;
- arbitrary scene graphs;
- projector lamp brightness control unless an existing trivial capability
  already provides it;
- a Python no-GIL migration; or
- AI/LLM logic.

It also does not introduce a second geometry or transform authority, 3D pose or
height correction, card/game semantics, synchronous camera reads in the
renderer, or a later ADR-0005 rectangle/board-detection feature. The existing
session camera ownership, Plan3 diagnostic areas, Plan4 metric ruler, Plan5
generic overlays and Plan6 tag inspection remain compatible boundaries.
