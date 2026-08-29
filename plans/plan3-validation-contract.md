# Plan3 validation contract

This sidecar defines the shared observables for `plans/plan3.md`. It extends the Plan2 session model without editing ADR-0001. Plan2 remains authoritative for camera slots, camera ownership, lifecycle and session-local calibration.

## Scope and compatibility boundary

Plan3 has two related goals:

- make the projector calibration pattern edge-complete and raster-safe, so its tags reach as close as safely possible to the edges of the configured usable area;
- let the operator enable or disable a per-camera diagnostic area on the projector, showing the calibrated surface area associated with that camera.

The area visualisation is diagnostic only. It must not alter camera selection, overlay ownership or calibration mathematics. Pointing uses the same full native camera-frame footprint as the diagnostic polygon, subject to native-camera and projector bounds. It is session-local and must not add camera-area persistence.

ADR-0001 is not edited. Where Plan2 already supersedes its persisted camera identity/calibration behaviour, Plan3 continues the Plan2 boundary.

## Calibration-pattern edge coverage

- Use a default 5×4 layout of 20 AprilTags for calibration. Retain the existing explicit 9–12 marker layouts for focused callers and compatibility unless implementation evidence requires otherwise.
- Place the 20 default markers using the full configured `usable_area`, not an additional large implicit inset. Retained layouts follow the same rule. Outer marker rectangles must be at the maximum raster-safe proximity to each usable-area edge, with intermediate centres evenly distributed across the remaining span.
- Respect the existing half-open coordinate convention and marker-size fit rules. No marker corner may equal or exceed the usable area's right or bottom boundary, and rendered integer pixels must remain inside the usable area and projector surface.
- The renderer and metadata must use the same raster-safe rectangle interpretation. Do not claim exact boundary contact if rounding would clip a marker.
- Pattern generation remains independent of camera frames, camera identity and Pygame window layout.

## Available-area meaning

- A camera's available area is derived from its current calibration, not from a manually authored polygon.
- For this diagnostic-only footprint, start with the camera's full native image rectangle, transform it with that camera's current `camera_to_projector` homography, and clip it to projector bounds. This deliberately may extrapolate beyond the tag-supported region.
- The result must contain at least three finite, non-collinear vertices. Projection must fail closed for invalid matrices, non-finite points, a projective horizon crossing or an empty/degenerate clipped result.
- The polygon estimates the camera-frame footprint on the projector. Camera clicks use that same native-frame footprint, with points still rejected when their projection falls outside the projector bounds. The calibration's tag-supported `valid_region` remains metadata for calibration quality and is not a separate click gate.
- Polygon vertices remain in projector-native coordinates. API responses serialise them as ordered `[x, y]` pairs.

## Area lifecycle and ownership

- Every session camera has an area-enabled flag owned by the same session/runtime state that owns its calibration. The default is disabled.
- Enabling requires the slot to be current-session `OPEN` and `CALIBRATED`, and requires a valid current available-area polygon. It is a desired-state operation: repeating enable or disable is a successful no-op response.
- Disabling hides the polygon but does not change calibration.
- Rename preserves calibration, area-enabled state, polygon and colour identity; rendering and responses use the current display name.
- Close, reopen, disconnect and any recalibration attempt clear the enabled flag and polygon. Reopening or reconnecting does not restore it. A successful recalibration remains disabled until explicitly enabled and cannot be enabled while its calibration is only `UNVERIFIED`.
- Area state is cleared before a recalibration capture/presentation begins, so an old polygon cannot remain visible during or after a recalibration attempt.
- No area state or polygon is persisted across process restarts, and newly discovered session slots never inherit old area data.

## Area control surface

The local API adds:

```text
POST /cameras/{slot}/area    {"enabled": true}
POST /cameras/{slot}/area    {"enabled": false}
```

The response includes the slot, current display name, lifecycle state, calibration state, `area_enabled`, `area_colour` and `available_area` (or `null` when disabled/invalid/unavailable). Existing camera list/status responses expose the same area observables.

The CLI adds:

```text
multivision cameras area enable camera-0
multivision cameras area disable camera-0
```

Controls address immutable session slots. Unknown slots, malformed bodies, unavailable/uncalibrated slots and invalid polygon state return structured non-success responses without mutating camera, calibration or overlay state. The API and CLI do not open cameras, initialise Pygame or calculate independent homographies.

## Projector rendering

- Render enabled areas on the projector in a dedicated diagnostic layer, using ordered projector-native polygon outlines and the current camera display name as a label.
- Use a deterministic palette assignment over currently renderable enabled slots in slot order. Simultaneously visible areas must have distinct colours; renaming must not change the assignment when the enabled set is unchanged. The area palette is separate from the red point-overlay colour.
- Draw enabled areas in deterministic slot order. Overlap is allowed and has no special arbitration, blending policy or preferred-camera semantics.
- While the calibration pattern is visible, suppress area polygons and labels so diagnostics cannot interfere with tag detection. In normal projector rendering, draw areas before the existing point overlay.
- Projector rendering remains main-thread-owned and must not make camera handles, API calls or calibration decisions.

## Deterministic validation matrix

Tests must cover:

- edge-near marker placement for every supported marker count, configured usable-area insets, marker-size limits, deterministic output and integer rendering bounds;
- polygon projection through identity and perspective homographies, native-camera intersection, projector clipping, invalid/non-finite/horizon-crossing transforms and empty/degenerate results;
- default disabled state, enable/disable idempotence, calibrated-only enablement, response serialisation and no mutation on failure;
- rename preservation of area state and current-name labels;
- close, reopen, disconnect and recalibration clearing area state, including clearing before a recalibration attempt;
- independent per-camera polygons and transforms, overlap, deterministic unique colours and full native-frame pointing behaviour;
- API and CLI requests against the running service without camera reopening or Pygame ownership leaks;
- area suppression during calibration-pattern presentation, normal projector layer ordering, labels and stale-area removal.

## Manual acceptance boundary

Automated checks establish state, geometry, rendering commands and ownership. They do not prove physical camera coverage or projector alignment. On the target Mac, record:

1. the updated calibration tags visibly reaching the usable surface edges without clipping;
2. at least two independently calibrated cameras with `area enable` showing distinct coloured outlines and names on the projector;
3. overlapping areas remaining distinguishable by colour;
4. disabling an area hiding it without changing calibration or pointing;
5. rename updating only the displayed name;
6. close/reopen, disconnect and recalibration hiding the old area until a valid calibration is explicitly enabled again.
