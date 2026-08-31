# Plan6 validation contract: configurable planar tag observations

## Intent and scope

Plan6 adds one read-only capability for inspecting fiducial tags in one camera's latest retained frame:

```text
multivision tags list --camera <camera> [--dictionary <dictionary>]
```

The capability returns every valid detected ID, including IDs unknown to the calibration pattern, with camera-native planar geometry and, when available, the corresponding projector-native planar geometry. It does not persist observations, identify cards, infer stacks, track tags across frames, or mutate overlays.

The query uses the existing persistent camera runtime and latest-frame ownership. The CLI and HTTP API never open or read camera handles directly.

## Accepted geometry meaning

“Pose” means planar pose only:

- centre position;
- ordered four-corner polygon;
- apparent polygon area;
- yaw/orientation around the axis perpendicular to the table.

It does not mean camera-relative 3D translation, height, tilt, roll, pitch, lens pose or physical dimensions.

The tag's reported yaw represents the card's yaw only when the tag is printed with a known fixed rotation relative to the card. Plan6 does not store card-specific mounting offsets; callers may apply that known offset themselves.

## Dictionary contract

Calibration-pattern detection remains independently configured around the existing `DICT_APRILTAG_36h11` family and must not be changed by this feature.

Generic tag inspection has a separate `tag_dictionary` configuration field whose default is:

```text
DICT_5X5_1000
```

The request may override that value with `--dictionary` or the equivalent HTTP query parameter. The initial supported set includes `DICT_5X5_1000` and the currently supported AprilTag-family names. Configuration/request validation rejects names outside that canonical set; OpenCV availability is checked when the detector is constructed at request time, and an unavailable supported constant is an explicit detection error. A typo must not silently select another dictionary.

One request uses one dictionary. Multi-family scanning is out of scope.

## Detection and frame contract

The service resolves one camera reference using the existing session identity rules, requires the camera to be open, available and to have a usable latest frame, then runs the configured detector on that frame.

The response includes:

- requested camera reference as `camera`;
- resolved session slot/stable identity as `camera_id` where available;
- dictionary name;
- frame counter;
- capture timestamp;
- tags sorted deterministically by marker ID.

No temporal averaging, tracking, debounce or latest-frame retry is introduced. An empty valid frame produces an empty `tags` list. Detector failures remain explicit errors.

The existing calibration detector seam remains available for calibration, but generic inspection has a separate dictionary-aware detector factory/cache (plus an injectable factory for tests). Calibration explicitly constructs or selects a detector for `calibration_pattern.marker_family`; a card-tag dictionary must never replace that detector. Generic inspection must not pass detections through calibration-pattern correspondence assembly, because unknown IDs are valid here.

## Planar observation schema

Each tag contains:

```json
{
  "id": 23,
  "camera": {
    "corners": [[x, y], [x, y], [x, y], [x, y]],
    "centre": [x, y],
    "orientation_degrees": 12.5,
    "area_px": 1234.5
  },
  "projector": {
    "corners": [[x, y], [x, y], [x, y], [x, y]],
    "centre": [x, y],
    "orientation_degrees": 11.8,
    "area_px": 987.0
  },
  "projection_status": null
}
```

The containing response also has `projection_status` for camera-wide failures, while a tag-level status explains an individual unsupported marker geometry.

When the camera-wide calibration is unusable, the top-level status contains the applicable code and every detected tag has `projector: null` with the same status. With usable calibration, the top-level status is `null`; an individual tag has a `null` tag status only when all four corners and its centre project successfully. An individual failure uses the first applicable code in this order: `POINT_OUTSIDE_CALIBRATED_REGION`, `INVALID_HOMOGRAPHY`, then `POINT_OUTSIDE_PROJECTOR_BOUNDS`. An empty tag list has no tag-level statuses, but still reports the camera-wide status.

`projector` is `null` when the current camera/projector calibration is unavailable, stale, invalid, outside its supported calibrated region, or produces an out-of-bounds/non-finite result. `projection_status` is either `null` or an object of the form `{"code": "CALIBRATION_STALE", "message": "..."}` carrying the structured error needed to explain that absence. Raw camera geometry remains available unless camera capture or detection itself fails.

The response also contains a top-level `projection_status` (`null` or a `{code, message}` object) for camera-wide projection availability. The service/API always expose camera geometry and attempt projector geometry. There is no second output-space authority and no need for a separate `--space` mode in this milestone.

## Geometry definitions

- Coordinates are native pixels: camera coordinates use the captured camera resolution; projector coordinates use the active projector output resolution.
- Corners retain the detector's ordered convention and are emitted in that order.
- `centre` is the projectively correct centre of the quadrilateral, defined by the intersection of its diagonals; it is not an arbitrary bounding-box centre.
- `area_px` is the absolute polygon area in the relevant pixel space.
- `orientation_degrees` is the angle of the ordered edge from corner 0 to corner 1, measured with +X right and +Y down, normalised to `[-180, 180)` degrees. Projector orientation is calculated after transforming the ordered corners, not by copying the camera angle.
- Projector centre is obtained by transforming the camera-space centre through the existing camera→projector homography.

All values must be finite. Individual malformed detector entries are omitted by the existing permissive detector normalisation and are never fabricated into a pose; structural detector failures, mismatched evidence and detector execution failures remain explicit errors.

## Projection authority and failure semantics

The current session camera calibration, projector descriptor, camera resolution, persisted `valid_region` and projector bounds remain authoritative. Plan6 must make that persisted calibrated region the single support-region interpretation for both the new tag projection and the legacy point-projection path; it must not leave the legacy path using full camera bounds while tags use a different rule. A non-mutating multi-point helper may be added to the existing point-projection authority so point requests and tag-corner projection share validation and transform logic.

Projection/status reads must be side-effect free: inspecting tags must not invalidate, replace or clear overlays, calibration records or persistent state. Lifecycle and calibration transitions remain responsible for their existing invalidation outside the query.

Plan6 must not introduce another homography, local scale, DPI estimate, camera intrinsic model or card-height correction.

For a calibrated camera, all four corners and the centre must project successfully before a tag receives a projector observation. If any required point is unsupported, the tag receives no projector geometry and an explicit per-tag projection status. A camera-wide calibration failure is represented consistently for every detected tag.

Changing camera/projector lifecycle or calibration state must affect only the current query result; tag observations are not persisted and do not become overlays.

## HTTP and CLI contract

Primary HTTP route:

```text
GET /cameras/{camera}/tags?dictionary=DICT_5X5_1000
```

`dictionary` is optional and defaults to the validated configuration value. The route returns the JSON-safe observation document above and preserves existing camera/snapshot/error routes.

Primary CLI route:

```text
multivision tags list --camera overhead
multivision tags list --camera overhead --dictionary DICT_5X5_1000
```

The CLI only validates arguments, encodes the camera/dictionary request, sends HTTP, prints the service response or structured failure, and returns a non-zero status for transport, validation or service failures. It performs no OpenCV, homography, pose or camera work locally.

## Deterministic validation matrix

Tests must cover:

- dictionary configuration round-trip and invalid dictionary rejection;
- `DICT_5X5_1000` detector initialisation and request selection without changing calibration-pattern defaults;
- empty, single and multiple detections with deterministic ID ordering;
- unknown IDs retained;
- malformed individual entries being omitted while structural evidence and detector failures remain explicit;
- detector construction being separate for calibration and card-tag inspection, including dictionary-aware injectable factories;
- diagonal-intersection centres, polygon areas and the stated orientation convention;
- projective transforms changing both position and orientation correctly;
- non-finite, horizon-crossing, calibrated-region and projector-bound failures;
- raw results remaining available without usable calibration;
- persistent latest-frame use without camera reopen/read ownership leaks;
- session camera references, renamed slots and stale lifecycle/calibration state;
- API schema/query validation and JSON-safe output;
- CLI delegation, URL encoding, structured failures and no camera/display imports.

Synthetic tests prove deterministic behaviour only. Hardware smoke instructions must remain separate and must not claim physical card/stack accuracy without execution on the target setup.

## Explicit non-goals

- card identity or card metadata;
- stack detection or stack-height estimation;
- tag tracking or temporal smoothing;
- multi-camera fusion;
- multi-dictionary scanning in one request;
- true 3D pose or camera intrinsic calibration;
- card mounting-offset databases;
- projector overlays or rendering changes;
- persistence of tag observations (the configured default dictionary is persisted as ordinary configuration);
- edits to ADR-0001, ADR-0002, ADR-0003 or `harness.toml`.
