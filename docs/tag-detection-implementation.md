# Plan6 tag-detection implementation boundary

This document records the implementation boundary for Plan6. The governing
architecture remains in [ADR-0001](ADR-0001_%20MultiVision%20MVP0%20Architecture%20and%20Implementation%20Contract.md),
[ADR-0002](ADR-0002_%20Metric%20Surface%20Calibration.md) and
[ADR-0003](ADR-0003_%20Physical%20Geometry%20Overlays.md); this document does
not amend any ADR. The deterministic observables are defined by the [shared
validation contract](../plans/plan6-validation-contract.md).

## Query and detector boundary

Plan6 adds a read-only query for one camera's one retained latest frame:

```text
session camera reference
  → persistent camera runtime
  → exactly one usable latest frame
  → request-selected tag detector
  → camera-native observations
  → optional shared camera → projector projection
```

The service resolves the camera reference using the existing session identity
rules, including renamed slots, and remains the only owner of camera handles.
The HTTP route and CLI are thin clients; neither may open a camera, read a
handle, run OpenCV or calculate geometry. The query does not retry or average
frames, track detections or mutate overlays, calibration records or other
persistent state.

Generic tag inspection has its own `tag_dictionary` configuration. Its default
is `DICT_5X5_1000`; the validated canonical set also includes the currently
supported AprilTag-family names. Request overrides use the same validated
names. Unsupported names are rejected explicitly, and a supported dictionary
whose OpenCV constant is unavailable is an explicit detector error.

Calibration-pattern detection remains independently configured by
`calibration_pattern.marker_family`, including its existing default. The
calibration detector must never be replaced by the request's tag dictionary.
Generic inspection uses a separate dictionary-aware detector factory/cache with
an injectable factory seam for tests.

## Detection and observation semantics

The detector path is not calibration detection. It does not filter detections
to calibration correspondences, reject IDs unknown to calibration or
deduplicate IDs. Every valid detection is retained, including several
observations with the same marker ID. Individual malformed entries may be
omitted during permissive normalisation or pure geometry validation; structural
detector evidence and detector execution failures remain explicit errors.

Valid observations retain the detector's ordered four corners. Their camera
geometry contains:

- the diagonal-intersection centre of the quadrilateral, not a bounding-box
  centre;
- absolute polygon area in native camera pixels; and
- the angle of the ordered corner `0 → 1` edge.

All values must be finite and the corners must be non-degenerate. Results are
deterministically ordered by `(marker_id, centre_y, centre_x)`, using the
validated camera-space centre to order duplicate IDs.

`camera.orientation_degrees` is an apparent camera-image angle measured with
+X to the right and +Y down, normalised to `[-180, 180)`. It is not physical
table yaw: perspective and camera viewpoint can make it differ from the tag's
physical rotation.

When projection is available, all required points are transformed through the
current shared camera → projector homography. Projector geometry is calculated
from the transformed ordered corners, including a new centre, area and an
independently recomputed `projector.orientation_degrees`; the camera angle is
never copied or adjusted. This is the rectified planar orientation in native
projector coordinates for evidence on the calibrated plane. It is not a 3D
pose estimate and is not reliable for tags materially above or below that
plane. A known fixed tag-to-card mounting rotation may be applied by a caller;
Plan6 stores no mounting-offset database.

## Response shape

The primary service operation is exposed as:

```text
GET /cameras/{camera}/tags?dictionary=DICT_5X5_1000
multivision tags list --camera <camera> [--dictionary <dictionary>]
```

The JSON-safe document contains the requested `camera`, resolved `camera_id`
where available, the selected dictionary, retained-frame counter and capture
timestamp, all valid `tags`, and a top-level `projection_status`:

```json
{
  "camera": "overhead",
  "camera_id": "camera-0",
  "dictionary": "DICT_5X5_1000",
  "frame_counter": 42,
  "captured_at_seconds": 1710000000.0,
  "tags": [
    {
      "id": 23,
      "camera": {
        "corners": [[100.0, 80.0], [140.0, 85.0], [135.0, 125.0], [95.0, 120.0]],
        "centre": [117.5, 102.5],
        "orientation_degrees": 7.125,
        "area_px": 1700.0
      },
      "projector": null,
      "projection_status": null
    }
  ],
  "projection_status": null
}
```

A tag's `projector` object has the same `corners`, `centre`,
`orientation_degrees` and `area_px` shape in projector-native pixels. Raw
camera geometry remains available whenever capture, detection and that
individual observation are valid.

## Projection and failure boundary

The current session camera calibration, projector descriptor, camera
resolution, persisted `valid_region`, homography and projector bounds remain
the sole projection authority. Every tag's four corners and diagonal
intersection centre must project successfully before any projector geometry is
returned. Projection is atomic per tag: partial transformed corners are never
returned.

A camera-wide projection failure sets the top-level `projection_status` to a
structured `{ "code": ..., "message": ... }` object and gives every detected
tag `projector: null` with the same status. This applies when calibration is
unavailable, uncalibrated, stale, invalid, resolution-inapplicable, outside
its supported region, or otherwise cannot produce finite projector geometry.
An empty valid frame still reports the camera-wide status but has no
per-tag statuses.

With usable calibration, the top-level status is `null`. An individual tag has
a `null` status only when all required points project successfully. Otherwise
its projector geometry is `null` and its status uses the first applicable code
in this order:

```text
POINT_OUTSIDE_CALIBRATED_REGION
INVALID_HOMOGRAPHY
POINT_OUTSIDE_PROJECTOR_BOUNDS
```

The same rules apply to finite results and horizon-safe transforms. The
projection read is side-effect free.

Plan6 deliberately tightens the legacy point-projection path: the persisted
calibrated `valid_region` is now the single support-region rule for both point
requests and tag projection. A camera-native point inside image bounds but
outside that region must fail with `POINT_OUTSIDE_CALIBRATED_REGION`; it must
not be extrapolated merely because it is inside the frame. Supported in-region
points retain their existing projection semantics. The implementation should
share the existing point-projection authority rather than introduce a second
homography, local scale, DPI estimate or height correction.

## Explicit non-goals

Plan6 does not add:

- card identity, card metadata or card semantics;
- uniqueness assumptions for marker IDs;
- stack detection or stack-height estimation;
- tracking, temporal smoothing, debounce or frame synchronisation;
- multi-camera fusion or multi-dictionary scanning in one request;
- true 3D pose, camera-intrinsic calibration or height correction;
- physical yaw inferred directly from unrectified camera orientation;
- card mounting-offset storage;
- projector overlays, rendering changes or tag-observation persistence;
- a second geometry/transform authority; or
- edits to ADR-0001, ADR-0002, ADR-0003 or `harness.toml`.

Deterministic tests establish software and mathematical behaviour only. The
hardware smoke procedure remains separate and must not claim physical tag or
card accuracy without observations on the target setup.
