# Plan4 validation contract

This sidecar defines the implementation observables for [plans/plan4.md](plan4.md). ADR-0002 is the accepted design authority for this plan. ADR-0001 remains authoritative for camera ownership and coordinate-space boundaries; Plan2/Plan3 remain authoritative for session-local camera slots, camera calibration state, pointing and diagnostic areas. ADRs and `harness.toml` must not be modified.

## Scope and ownership

Plan4 adds one shared metric calibration for the active projector/table geometry and one session-local validation ruler. It does not add a metric transform to a camera, a `mm_per_pixel` scalar, or a second camera/projector calibration.

The authority chain is:

```text
printed metric target
  ↓ known target surface-mm coordinates
selected calibrated camera frame
  ↓ existing per-camera camera-native → projector-native transform
projector-native correspondences
  ↓ shared RANSAC metric calibration
surface-mm ↔ projector-native transform pair
  ↓ service-owned metric state
ruler geometry → projector renderer
```

The existing camera/projector transform is an input and must not be changed by metric calibration. Its applicability metadata must include the same logical projector-output descriptor used by metric calibration; changing that descriptor stales camera and metric geometry together. Any later physical geometry must consume the shared surface-mm transform; Plan4 itself must not implement ADR-0003's grid, circle, generic line, highlighted region or multi-overlay registry.

Metric state is session-local for this implementation. The existing camera calibration persistence code is not repurposed to persist metric state. A new process starts `UNCALIBRATED`; no persisted metric matrix is trusted or loaded. The metric record is one shared object, not one record per camera.

## Metric thresholds and applicability

Configuration has a separate validated metric threshold group. The initial defaults are `ransac_reprojection_threshold_mm = 3.0`, `max_capture_corner_jitter_pixels = 2.0`, `max_mean_fit_error_mm = 2.0`, `max_fit_error_mm = 5.0`, `min_inlier_ratio = 0.5`, `min_unique_target_fiducials = 4` and `min_spatial_coverage = 0.5`; they remain configuration rather than scattered constants. Thresholds are in physical units unless explicitly named otherwise:

- positive RANSAC reprojection threshold in mm;
- positive maximum accepted camera-corner jitter across the three capture frames in camera pixels;
- maximum mean fit residual in mm;
- maximum individual fit residual in mm;
- minimum RANSAC inlier ratio;
- minimum unique target-fiducial count;
- minimum spatial coverage fraction across the target's known metric page area;
- positive metric-calibration format/version values as applicable.

The projector descriptor is resolved once for the running service and is shared with the display runtime. It contains the configured projector resolution and a logical selected-output identity. That descriptor is stored in every camera calibration record and every metric record. The service exposes a synchronised update operation for same-process output/reconfiguration tests and future display selection; it atomically marks every camera calibration whose descriptor no longer matches as not spatially usable, marks the metric record `STALE`, and removes the ruler. A normal restart starts a fresh session and therefore starts `UNCALIBRATED`, not `STALE`.

A metric calibration record contains at least:

- `projector_to_surface` and `surface_to_projector`, both finite, normalised 3×3 matrices;
- metric state and calibration/version metadata;
- projector resolution and logical projector-output identity used for applicability;
- target format/version and marker family;
- unique fiducial count, corner count, RANSAC inlier count and ratio;
- `fit_error_mm` as mean and maximum residual information;
- metric spatial coverage;
- observation-camera session slot and diagnostics identifying the camera calibration used;
- timestamp/session metadata;
- `latest_physical_validation_error_mm` as `null` until an independent ruler measurement is recorded;
- zero or more append-only session-local validation records containing requested length, observed length and absolute error in mm.

`latest_physical_validation_error_mm` must never be copied from or calculated from homography fit residuals. It is only populated from an explicitly supplied physical measurement.

The record is usable only when:

- its state is `CALIBRATED`;
- the current projector resolution equals the recorded resolution;
- the current logical projector-output identity equals the recorded identity;
- the matrices and metadata remain finite and structurally valid.

A resolution or output-identity mismatch changes the observable state to `STALE` and blocks ruler operations. The same descriptor mismatch also stales each affected camera/projector calibration, so metric recalibration cannot use a transform for another output. Explicit clear returns to `UNCALIBRATED`. A camera close, rename, reconnect, recalibration or movement does not automatically invalidate an already established metric surface transform: the observation camera is diagnostic provenance, not metric ownership. A new metric calibration still requires a currently `CALIBRATED` observation camera/projector transform.

## Printable target contract

The generated target is deterministic A4 portrait:

- page width: exactly `210.0 mm`;
- page height: exactly `297.0 mm`;
- target coordinate origin: page top-left;
- positive x points right and positive y points down;
- format version: a constant owned by the target module and emitted in SVG metadata;
- marker family: the existing supported AprilTag family `DICT_APRILTAG_36h11`;
- marker IDs: exactly 20 unique IDs, `0..19`;
- layout: 4 columns × 5 rows;
- marker square side: exactly `22.0 mm` in the target definition;
- marker starts: x positions linearly distributed from `8.0` through `180.0` mm and y positions linearly distributed from `40.0` through `263.0` mm, inclusive; each marker's four target corners are derived from its start and 22-mm side;
- the visible orientation cue and version text are asymmetric and do not overlap marker ink;
- the labelled reference segment runs from `(55.0, 20.0)` to `(155.0, 20.0)` in target coordinates and is labelled `Expected reference: 100.0 mm`.

The target's known coordinates are the metric surface coordinates used by calibration. The sheet may be translated or rotated on the tabletop; callers must not assume it is aligned with projector axes. Marker IDs plus target coordinates establish the target frame.

The SVG must have explicit physical dimensions and a millimetre viewBox, include a 100%/Actual-size instruction and a warning against `Fit to page`, printer scaling and browser scaling. If marker images are embedded as raster data, their source pixel dimensions and displayed millimetre dimensions must be explicit and deterministic. The SVG must carry target format/version and marker-family metadata. Tests check generated structure and dimensions; they do not prove printer output.

## Target detection and corner orientation

The detector boundary remains `FiducialDetector`/`DetectedMarker`; no camera or Pygame code is allowed in target assembly.

Assembly must:

- use only IDs defined by the current target version/family;
- retain all four corners for every accepted marker;
- reject unknown IDs, duplicate markers, partial markers, malformed/non-finite/non-convex corners and insufficient spread;
- associate detections with the target's known marker coordinates;
- tolerate arbitrary sheet rotation without silently treating image-order corners as target-order corners.

The accepted orientation procedure is a two-stage deterministic association:

1. match detected marker centres to known target marker centres and estimate a provisional spread-marker projective mapping;
2. transform each marker's four known target corners through that provisional mapping and choose the cyclic corner permutation with the smallest total predicted-to-detected distance, rejecting ambiguous or excessively poor assignments; produce the final camera-native target correspondences in target-corner order.

The final calibration estimator, not the provisional association, is authoritative for quality metrics. A target that cannot be identified or oriented reliably fails closed. The permissive existing projector-pattern normaliser must not be used as the only metric detector path: malformed/partial evidence must remain observable to strict metric validation.

## Metric calibration algorithm

For a selected camera:

1. require the current session slot to be `OPEN` and runtime `AVAILABLE`;
2. require its existing camera/projector calibration status to be `CALIBRATED` and applicable to its current native resolution and the current projector configuration;
3. capture one or more stable retained frames after the operator has placed the verified printed target flat and stationary;
4. detect and assemble target marker correspondences in camera-native coordinates;
5. map every accepted camera-native corner through that camera's existing camera→projector transform, with finite/projector-bound checks; a target correspondence outside projector-native bounds invalidates the attempt rather than being silently extrapolated;
6. estimate `projector-native → surface-mm` with RANSAC using all valid four-corner correspondences and the configured positive `ransac_reprojection_threshold_mm`;
7. require a non-degenerate matrix, enough unique markers/corners, meaningful coverage measured against the full `210 × 297 mm` target page area, and configured fit quality;
8. calculate the explicit inverse `surface-mm → projector-native` and validate round-trip behaviour;
9. publish the complete shared record atomically only after every check succeeds.

Before the capture, the service clears the current ruler and marks the metric transform unavailable. It requests an exclusive blank projector frame from the main-thread display so existing areas, point overlays and any ruler cannot contaminate target observation; the display acknowledges the blank frame, then the service settles and captures exactly three consecutive candidate frames. Accepted target IDs must match across all three frames and each corresponding camera corner must stay within the configured camera-pixel jitter tolerance of the first frame; the averaged correspondences are then used for estimation. A supplied correspondence fixture bypasses hardware capture but still follows the same atomic publish contract. A failed attempt does not restore or leave an older record usable. Camera calibration matrices are not changed on metric success/failure, but their output-applicability status may already be stale if the projector descriptor changed.

Fit residuals are distances in surface millimetres between known target points and their reprojected points through `projector_to_surface`, calculated over RANSAC inliers only. Coverage is the convex-hull area of accepted/inlier target points divided by the full known `210 × 297 mm` target page area. It is not projector-pixel coverage and not a physical ruler result.

## Units and ruler contract

Accepted external units are exactly `mm`, `cm` and `in` (with any aliases limited to explicitly tested spellings). Internal values are millimetres:

```text
1 mm = 1.0 mm
1 cm = 10.0 mm
1 in = 25.4 mm
```

All coordinates, lengths and observed measurements must be finite. Requested lengths must be positive where a length is supplied directly; an endpoint-derived ruler must have distinct endpoints and a positive Euclidean surface-mm distance.

The minimum ruler request uses two surface points:

```json
{
  "from": {"x": 100.0, "y": 100.0},
  "to": {"x": 300.0, "y": 100.0},
  "unit": "mm",
  "observed_length": null,
  "observed_unit": "mm"
}
```

The service calculates the requested length in mm, formats the label in the requested unit, maps the endpoints and deterministic tick positions through `surface_to_projector`, and rejects a non-finite, horizon-crossing or out-of-projector result. Plan4 does not shrink or clip a requested physical line to make it fit. The service also validates the complete line, tick segments and start/end marker extents against projector-native bounds after raster-safe rounding. Decorative labels may be clamped inside the projector surface and do not alter the requested line. A later ADR may add clipping for generic overlays.

Tick generation is deterministic in surface space before projection. It may use major ticks every 10 mm and minor ticks every 5 mm, omitting ticks that would be too dense for the requested segment; this is presentation only and must not change the endpoint length. The renderer receives complete service-produced projector-native draw primitives and must not infer a pixel scale.

A ruler is one session-local replaceable metric ruler, independent of the existing red point overlay. Creating a new ruler replaces the previous ruler. `DELETE /metric/ruler` removes only the ruler. Adding or clearing a ruler must not modify camera calibration, metric calibration or the point overlay. A projector descriptor change removes the ruler as part of atomic spatial invalidation.

If `observed_length` is supplied, normalise it with `observed_unit`, calculate:

```text
absolute_error_mm = abs(observed_length_mm - requested_length_mm)
```

and append/return the validation record; `latest_physical_validation_error_mm` is exactly the absolute error from the most recently recorded observation. This is an operator observation, not an automatic pass/fail threshold and not a substitute for the required manual evidence.

## API and CLI surface

The runtime API is:

```text
POST   /metric/calibration
GET    /metric/calibration/status
DELETE /metric/calibration
POST   /metric/ruler
DELETE /metric/ruler
```

Metric calibration has an internal/existing-display capture-state handshake: the service requests an exclusive blank projector frame, and the main-thread display acknowledges it before capture. The handshake is not a second geometry authority and need not be a separate public endpoint.

`POST /metric/calibration` accepts a selected camera reference. An optional correspondence payload is an injected deterministic test seam only; normal CLI use sends no camera-native detections. Target generation is not a camera operation and remains a pure artifact operation.

Metric calibration status responses expose state, applicability/error code, transform directions, target/version, projector metadata, observation-camera diagnostics and fit/physical-validation metrics. Ruler responses expose requested surface endpoints, projector endpoints/ticks, length in mm and selected units, label and optional observed/absolute error. Serialisation must be JSON-safe and reject NaN/infinity.

The CLI remains a thin client for all running-service operations:

```text
multivision metric calibrate --camera camera-0
multivision metric status
multivision metric clear
multivision metric ruler --from-mm 100,100 --to-mm 300,100 --unit mm
multivision metric ruler clear
```

`multivision metric target generate --output metric-target.svg` may call the shared deterministic SVG generator locally and write the artifact; it must not open a camera, initialise Pygame, call the service or calculate a homography. Runtime metric commands must use HTTP and must not duplicate service state or geometry.

## Rendering contract

The projector surface remains projector-native. The display runtime owns Pygame and reads immutable service snapshots on the main thread.

Normal render order is:

```text
clear
→ calibration pattern exclusively when camera calibration is active
→ diagnostic areas
→ metric ruler
→ existing red point overlay
```

During metric blank capture, suppress every normal projector layer and render a blank surface until the main-thread acknowledgement is complete. While the existing camera calibration pattern is visible, also suppress metric ruler drawing so no extra projector marks contaminate camera fiducial capture. Area suppression remains governed by Plan3. Ruler rendering must tolerate a missing/stale/malformed service snapshot by drawing nothing and surfacing the existing display error path rather than inventing approximate geometry. Labels use the requested unit and computed length. Rendering commands must be deterministic and separate from calibration mathematics.

## Deterministic test matrix

Tests must cover:

- A4 page dimensions, exact reference segment, marker IDs, marker corners, version/family/orientation metadata and byte-stable SVG generation;
- target-aware assembly under rotation, corner permutation, unknown/duplicate/partial/malformed markers and weak spread;
- unit conversion and exact inch arithmetic;
- metric homography recovery under perspective, noisy corners and RANSAC outliers, including configured mm RANSAC thresholds, inlier-only residuals and full-page coverage denominator;
- explicit inverse and round trips;
- projective scale varying across projector pixels, proving no scalar `mm_per_pixel` implementation can pass;
- degenerate matrices, horizon crossings, non-finite values, invalid camera calibration and projector-bound failures;
- fit-error metrics versus null/independent physical-validation error;
- one shared registry/record, no camera-owned metric state, reset and restart-local behaviour;
- same-process resolution/output-identity invalidation staling both camera and metric geometry atomically and `STALE` fail-closed ruler requests;
- target capture blank acknowledgement, three-frame stability/movement rejection, atomic success/failure and unchanged camera calibration;
- complete raster-safe ruler endpoint/tick/marker/label geometry, observed-length recording with latest-error semantics, replacement, clear and point-overlay coexistence;
- API schemas, structured errors and JSON-safe output;
- CLI target writing, HTTP delegation, unit parsing and no hardware ownership;
- display order, pattern suppression, stale-ruler removal and projector failure recovery.

Synthetic tests establish mathematical and software behaviour only. They do not establish that a printer produced 1:1 output, that a target is flat, that the projector/table moved, or that a physical ruler matches the requested length.

## Manual acceptance boundary

The updated smoke procedure must record, on the target Mac:

1. generated target path, printer and actual-size setting;
2. measured printed 100-mm reference and deviation before calibration;
3. target placement/orientation and selected already-calibrated observation camera;
4. calibration state, target version, unique markers/corners, inliers, fit errors and coverage;
5. rulers at separated positions and orientations, including near intended usable-area edges;
6. requested length, selected display unit, physically measured length and absolute error for each ruler;
7. use of a second calibrated camera as an observation device without creating a second metric transform;
8. explicit clear and deliberate same-process stale/invalidation checks refusing both camera-dependent recalibration and metric operations.

No automated test, screenshot, generated SVG, API response or mock projector is evidence of physical metric accuracy.
