# Plan5 validation contract: coordinate-aware physical geometry overlays

This sidecar defines the implementation observables for [plans/plan5.md](plan5.md). ADR-0003 remains the generic-overlay source decision, Plan4 remains authoritative for metric calibration, and ADR-0001/Plans 2–4 remain authoritative for camera ownership, projector geometry, session lifecycle and display threading. Do not modify ADRs or `harness.toml`.

## Intent of the reduction

Plan5 deliberately implements a small geometry layer, not a general graphics framework.

The useful new capability is:

```text
camera says where
physical/projector geometry says what and how large
```

For example, a caller may identify a miniature centre in camera pixels and request a true physical 6-inch circle around that point. The implementation must convert the camera point through the existing calibration chain and construct the circle in the canonical metric frame. It must never approximate a physical radius from camera pixels.

The following ideas are intentionally deferred from Plan5:

- `surface_edge_mm` or any second physical coordinate frame;
- `camera_px` as a shape-size space;
- `camera_px` as a ruler measurement space;
- implicit or automatic **infinite**-grid extent; a named projector-footprint mode may derive one finite extent from the current metric homography;
- generic z-index/scene graph/plugin rendering;
- alpha/compositing systems;
- update/replace semantics for existing overlays;
- game rules, piece identity, movement legality, LOS legality, AI state or multiplayer state.

These can be added later if a concrete use case justifies them without changing the core authority chain below.

## Scope and authority

Plan5 adds five generic overlay kinds:

```text
grid
circle
rect
line
ruler
```

The service is the only owner of coordinate resolution, calibration checks, source-space shape construction, clipping, dependency tracking and overlay state. The display consumes already-resolved projector-native primitives. API and CLI remain thin boundaries.

The final authority chain is:

```text
request point references + declared geometry/measurement space
  ↓
existing camera/projector/metric calibration authorities
  ↓
source-space geometry construction
  ↓
projector-native materialisation and clipping
  ↓
session-local overlay registry
  ↓
main-thread renderer
```

No overlay may calculate `pixels_per_cm`, DPI, a local camera physical scale or a second homography.

The existing `/overlay/point` and Plan4 metric-ruler paths remain supported. They may reuse shared pure helpers where practical but must not create a second geometry authority.

## Coordinate model

There are exactly three public point-reference spaces:

- `projector_px` — projector-native pixel position; no camera or metric calibration required;
- `camera_px` — native pixel position from one explicit camera; requires that camera to be open/available and currently camera→projector calibrated;
- `surface_mm` — canonical ADR-0002/Plan4 physical surface frame; accepts `mm`, `cm` or `in`, normalized to millimetres; requires usable shared metric calibration.

There are exactly two public geometry/measurement spaces:

- `projector_px` — dimensions or distances are pixels;
- `surface_mm` — dimensions or distances are physical values normalized to millimetres.

`camera_px` is intentionally **point-only**. A camera pixel can locate a centre/origin/endpoint, but Plan5 does not define a circle radius, rectangle size, grid spacing or ruler measurement in camera pixels.

A point reference is conceptually:

```json
{
  "space": "camera_px",
  "camera": "camera-1",
  "x": 812.0,
  "y": 443.0
}
```

A physical point may include a unit shared by both coordinates:

```json
{
  "space": "surface_mm",
  "x": 5.0,
  "y": 3.0,
  "unit": "cm"
}
```

Pixel-space points use pixel values and reject physical units. Camera points require a camera identity. Projector points must not carry a camera identity. All numeric values must be finite; booleans are not numeric values.

## Mixed anchor conversion

A circle, rect or grid declares a geometry space of `projector_px` or `surface_mm`. Its centre/origin may be in any supported point space.

The service converts the point into the declared geometry space using existing transforms only.

Supported conversion logic is conceptually:

```text
projector_px -> projector_px
camera_px    -> projector_px
surface_mm   -> projector_px

projector_px -> surface_mm   (requires usable inverse metric transform)
camera_px    -> projector_px -> surface_mm
surface_mm   -> surface_mm
```

A mixed camera-anchor/physical-circle request therefore requires:

1. the selected camera's valid camera→projector calibration;
2. Plan4's current metric projector↔surface transform;
3. successful finite conversion of the anchor into canonical `surface_mm`;
4. circle construction in `surface_mm`;
5. surface→projector projection of the sampled result.

It must not use local pixel scale, DPI or a camera-space radius approximation.

Dependencies are exactly those implied by conversions actually required by the request. A pure `projector_px` overlay requires no camera or metric calibration.

## Primitive contracts

### Circle

```text
centre: point reference
geometry_space: projector_px | surface_mm
radius: positive quantity in geometry-space units
style
```

A physical circle is sampled deterministically in canonical `surface_mm` before projection. A projector circle is sampled in projector pixels. Circle tessellation must be deterministic and bounded by configured error/budget limits.

### Rotated rect

```text
centre: point reference
geometry_space: projector_px | surface_mm
width: positive quantity in geometry-space units
height: positive quantity in geometry-space units
angle_deg: finite angle
style
```

The four corners are constructed in the declared geometry space around the converted centre, then rotated and projected if required.

Angle convention is explicit and shared across primitives: zero points along positive x; positive angle is counter-clockwise in a conventional x-right/y-up interpretation. Where stored coordinates increase downward, implementation must compensate consistently rather than silently reversing the public convention.

### Grid

```text
origin: point reference representing one grid intersection
geometry_space: projector_px | surface_mm
spacing: positive quantity in geometry-space units
extent: {width, height} positive finite quantities in geometry-space units
angle_deg: finite angle
style
```

**Extent is required for a normal `GridRequest`.** The named projector-footprint grid capability derives one finite surface-mm extent by inverse-projecting the four projector-output corners; it is not an automatic/infinite extent mode.

The grid is square in its declared source space. `origin` is one deterministic grid intersection. Spacing and extent are applied before rotation and projection. The generated segment collection must be finite before projection and must respect configured budgets. The derived footprint bounding box must be finite and must not cross the homography horizon; projector-native clipping remains authoritative.

A physical 1-inch grid means exactly 25.4 mm source spacing before projection.

### Line

```text
start: point reference
end: point reference
label: optional string
style
```

Endpoints may independently use any of the three point spaces. Each is resolved to projector-native coordinates using existing authorities, then the projector-native segment is clipped.

A line is literal geometry only. It does not decide LOS legality, path legality or obstruction.

### Ruler

```text
start: point reference
end: point reference
measurement_space: projector_px | surface_mm
unit: px when projector_px; mm|cm|in when surface_mm
label: optional override
style
```

For `projector_px`, both endpoints are converted to projector coordinates and Euclidean pixel distance is measured there.

For `surface_mm`, both endpoints are converted to canonical metric surface coordinates and physical Euclidean distance is measured there. This may require camera calibration, metric calibration or both depending on endpoint point spaces.

Ticks are deterministic and bounded. Existing Plan4 physical ruler calculations should be reused or factored into a shared pure physical-distance/tick path rather than duplicated.

There is no camera-pixel ruler measurement in Plan5.

## Strict quantity and style model

Physical quantities accept only `mm`, `cm` or `in` and normalize to millimetres. Projector quantities accept only `px`.

Unknown units, unit/space mismatches, non-finite values, booleans, zero/negative dimensions and unsupported fields fail closed.

Minimal style:

```json
{
  "colour": "#RRGGBB",
  "fill": false,
  "line_width_px": 2
}
```

Requirements:

- `colour` is strict six-digit HEX and normalizes to RGB;
- `line_width_px` is a positive integer in final projector pixels;
- `fill` applies to circle/rect only;
- line, grid and ruler reject `fill: true`;
- defaults are deterministic by primitive kind;
- style never changes physical/source geometry;
- alpha and separate stroke/fill colours are out of scope.

## Bounded generation

Configuration must bound work before registry mutation.

Initial implementation should expose bounded defaults equivalent in spirit to:

```text
max_overlay_vertices = 10000
max_overlay_segments = 5000
max_overlay_ticks = 200
max_overlay_label_characters = 256
```

Exact names may follow existing config conventions, but they must be positive, test-injectable and round-trip safely.

Circle sampling, grid generation, tick generation and labels must reject over-budget requests before allocating an unsafe collection or partially mutating registry state.

## Projector materialisation and clipping

All source geometry is constructed at requested size before projector clipping.

Requirements:

- all transformed coordinates must be finite;
- horizon-crossing or non-finite projective results fail the request;
- line segments are clipped independently to output bounds;
- filled polygons are polygon-clipped;
- outline closed shapes clip their source edges independently so clipping does not invent a screen-edge closing stroke;
- an entirely off-screen but otherwise valid overlay may remain registered and produce zero draw calls;
- clipping never shrinks, re-centres or otherwise reinterprets the requested geometry;
- raster rounding happens only after source construction, transformation and clipping;
- renderer code must not recompute geometry.

## Registry model

One `OverlayRegistry` is owned by `MultiVisionService` for new generic overlays.

Each entry stores at minimum:

```text
id
optional unique name
kind
visible
immutable normalized request
immutable materialised projector primitives
camera dependencies
metric dependency flag
projector-output dependency
insertion sequence
```

IDs are UUID4 unless a deterministic test seam injects one. Names are optional, session-local and unique; names containing `/` or canonical UUID syntax are rejected so selectors remain unambiguous.

Supported operations:

```text
create
list
show
hide
remove
clear
```

There is no implicit replace/update in Plan5. To change geometry, remove and recreate the object.

Show/hide is idempotent. Listing and rendering use insertion order within the explicit primitive layer order, never UUID lexical order.

## Dependency invalidation

No stale materialised projector geometry may remain visible.

The simple Plan5 rule is to **remove affected generic overlays** when an authority they depend on becomes invalid.

Examples:

- camera-dependent overlay: remove before/when that camera calibration becomes stale/reset/changed or camera lifecycle makes the dependency unusable;
- metric-dependent overlay: remove before metric recalibration/reset and when metric calibration becomes stale;
- all generic overlays: remove when projector output identity/resolution changes because their final projector primitives are tied to that descriptor;
- unrelated projector-only overlays remain when only a camera or metric calibration changes.

A mixed camera-anchor/physical shape depends on both that camera and metric calibration.

Invalidation must be atomic relative to visible service state: stale materialised geometry must not survive one frame after the dependency transition becomes observable.

## Compatibility boundaries

Plan3 diagnostic camera areas remain diagnostics, not generic overlays and not physical gameplay regions.

The legacy point overlay remains independently addressable and must coexist with generic overlays.

Plan4's existing metric-ruler endpoint remains supported. Its physical mathematics should reuse the same pure ruler/metric helpers where practical, but legacy compatibility must not create a second registry or second transform/scale authority.

Normal generic overlays are suppressed during camera calibration patterns and metric blank capture.

## Display contract

The display/runtime receives immutable service-produced projector-native primitives only.

It may:

- draw polygons, circles/polylines, segments, ticks and labels;
- apply already-normalized RGB/style values;
- obey visibility and deterministic layer ordering;
- round/clamp raster-safe final values as specified by the primitive snapshot.

It must not:

- resolve camera points;
- call homographies;
- convert units;
- derive physical scale;
- choose source-space dimensions;
- rebuild a grid/circle/rect/ruler.

Normal order should preserve current semantics while adding generic layers approximately as:

```text
grid/background
diagnostic areas
rects/circles
lines/rulers
labels
legacy point emphasis
```

Exact placement may adapt to the current renderer, but it must be deterministic and keep the legacy point visible as emphasis.

## API contract

Primary routes:

```text
POST   /overlays/grid
POST   /overlays/circle
POST   /overlays/rect
POST   /overlays/line
POST   /overlays/ruler
GET    /overlays
POST   /overlays/id/{id}/show
POST   /overlays/id/{id}/hide
DELETE /overlays/id/{id}
POST   /overlays/name/{name}/show
POST   /overlays/name/{name}/hide
DELETE /overlays/name/{name}
DELETE /overlays
```

Schemas reject unknown fields and malformed nested references.

Responses expose normalized public overlay state:

```text
id
name
kind
visible
normalized request/style
dependency/status information where useful
```

**Do not return the full materialised projector primitive collection as the normal API representation.** That is an internal service→display contract and may be very large for grids/curves.

API code performs no homography, camera, Pygame, unit conversion, clipping or shape mathematics; it delegates to the service.

Existing point/metric/calibration/camera/area routes retain their established behaviour.

## CLI contract

Creation uses the canonical thin form:

```text
multivision overlay <kind> --spec-json '<json>'
```

Lifecycle operations:

```text
multivision overlays list
multivision overlay show --id ... | --name ...
multivision overlay hide --id ... | --name ...
multivision overlay remove --id ... | --name ...
multivision overlays clear
```

Exactly one selector is accepted for lifecycle commands. CLI prints returned IDs/names and structured errors but performs no geometry/calibration work locally.

## Deterministic test matrix

Tests should be consolidated around behaviours rather than one near-duplicate suite per layer.

### Pure geometry tests

Cover:

- unit conversion and strict unit/space matching;
- point validation and camera identity rules;
- projector/camera/surface point resolution;
- projector→surface inverse conversion for mixed anchors;
- mixed `camera_px` centre + `surface_mm` circle/rect/grid;
- mixed line endpoints;
- physical and projector-pixel rulers;
- deliberate rejection of camera-pixel shape sizes/measurement;
- circle sampling before projective transformation;
- rotated rect geometry;
- finite explicit grid extent and deterministic ordering;
- angle convention positive/negative cases;
- style normalization;
- generation budgets;
- clipping without size mutation or synthetic screen-edge strokes;
- off-screen valid geometry producing zero primitives/draws;
- skewed homographies proving that physical geometry cannot pass via constant pixel scale.

### Service/registry tests

Cover:

- one registry owner;
- UUID/name behaviour and duplicate rejection;
- insertion ordering;
- show/hide idempotence;
- remove/clear;
- atomic failed create;
- camera/metric/projector dependency recording;
- invalidation removing affected overlays only;
- camera-anchor + physical-shape dependence on both authorities;
- projector-only overlays without calibration;
- coexistence with point, area and legacy metric ruler;
- calibration-pattern/blank-frame suppression.

### Boundary/display/integration tests

Cover:

- strict API schemas and JSON-safe normalized responses;
- absence of full projector primitive dumps from normal API responses;
- thin CLI delegation;
- no camera/homography/Pygame work in API/CLI;
- fake-Pygame deterministic layer order, visibility, fills, colours, widths and labels;
- running-service create/list/show/hide/remove/clear flow;
- invalidation leaves no stale draw calls;
- projector-only and physical overlays can coexist;
- existing endpoint regressions remain green.

Synthetic tests prove software/mathematical behaviour only and must not claim physical accuracy.

## Manual hardware acceptance

Update `docs/mvp0-manual-smoke-check.md` to record at minimum:

1. a rotated physical grid with explicit extent and known spacing, measured at separated tabletop positions;
2. a known-size physical circle and rectangle, both outline and filled where useful;
3. physical line/ruler checks at several positions/orientations;
4. a camera-pixel centre used to place a physical circle/rect after both calibrations;
5. direct projector-pixel geometry demonstrating intentionally pixel-based sizing;
6. clipping at projector edges without changing requested physical size;
7. several named/unnamed overlays coexisting, including show/hide/remove;
8. camera invalidation removing camera-dependent geometry while unrelated geometry remains;
9. metric invalidation removing physical geometry;
10. projector descriptor change clearing materialised generic overlays.

Hardware evidence, not mocks, screenshots or synthetic projector commands, establishes physical accuracy.

## pi-harness completion expectations

The implementation agent should treat the numbered Plan5 lines as execution steps and this sidecar as their shared acceptance contract.

Before Plan5 is complete it must:

- run the full deterministic repository suite from a clean root;
- keep changes inside the stated scope;
- remove duplicate scale/transform/unit logic discovered during integration;
- preserve ADR-0001/0002/0003 and `harness.toml` unchanged;
- report which manual physical checks remain unperformed rather than claiming them from automated tests;
- explicitly flag any necessary deviation from this sidecar instead of silently generalising the architecture.

## Adversarial guardrails

The implementation should be rejected or corrected if it introduces any of these:

1. camera pixels as physical shape size or ruler distance;
2. a second metric frame such as `surface_edge_mm` without a new approved decision;
3. local `pixels_per_cm`, DPI or constant pixel-scale approximation;
4. auto/infinite grid generation when explicit extent suffices;
5. generic scene graph, z-index or plugin renderer architecture;
6. a second homography/metric authority in overlay, API, CLI or display code;
7. API responses coupling clients to thousands of renderer primitives;
8. stale materialised geometry after calibration/output invalidation;
9. clipping that changes requested physical dimensions;
10. game/AI/piece/LOS/movement semantics in the generic geometry layer.

The intended result is intentionally modest: a reliable generic tabletop geometry API with camera-addressable points and honest physical/projector sizing, suitable as a foundation for later piece tracking without prematurely becoming a graphics engine.
