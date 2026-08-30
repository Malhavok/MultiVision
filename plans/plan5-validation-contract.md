# Plan5 validation contract: coordinate-aware physical geometry overlays

This sidecar defines the implementation observables for [plans/plan5.md](plan5.md). ADR-0003 remains the generic-overlay source decision, Plan4 remains authoritative for metric calibration, and ADR-0001/Plans 2–4 remain authoritative for camera ownership, projector geometry, session lifecycle and display threading. Do not modify ADRs or `harness.toml`.

## Scope and authority

Plan5 adds five generic overlay kinds:

```text
grid
circle
rect
line
ruler
```

The service is the only owner of coordinate resolution, calibration checks, shape construction, clipping and overlay state. The display only consumes already-resolved projector-native draw primitives. The API and CLI are thin boundaries.

The final authority chain is:

```text
request quantities and anchors
  ↓ explicit coordinate-space resolution
camera/projector/metric calibration authorities
  ↓ pure shape construction in declared shape/measurement space
projector-native geometry
  ↓ output clipping and raster-safe draw primitives
session-local overlay registry
  ↓ main-thread renderer
Pygame projector output
```

No overlay may calculate `pixels_per_cm`, DPI, a camera-local physical scale or a second homography. No overlay owns game semantics, piece identity, movement, grenade rules, line-of-sight legality, AI state or multiplayer state.

The existing `/overlay/point` path and Plan4 `/metric/ruler` path remain supported. They must coexist with generic objects without silently becoming a second geometry authority. The generic registry owns new objects; compatibility adapters may map old requests into the same builders where practical.

## Coordinate spaces and typed quantities

Supported coordinate spaces are exactly:

- `projector_px`: projector-native pixel coordinates; direct final-output space; no calibration required for geometry resolution, though output bounds still apply;
- `camera_px`: native pixel coordinates from one named/selected camera before any transform; requires that camera to be open, available and currently camera/projector-calibrated;
- `surface_mm`: ADR-0002's canonical metric surface frame; coordinates may be supplied in `mm`, `cm` or `in` and are normalised to millimetres; requires a current usable shared metric calibration;
- `surface_edge_mm`: a point-reference-only derived physical frame; coordinates may be supplied in `mm`, `cm` or `in`; requires a current usable metric calibration and a usable mapping of projector output corners; it is not valid as a shape or measurement space.

A point reference has this conceptual shape:

```json
{
  "space": "camera_px",
  "camera": "camera-1",
  "x": 812.0,
  "y": 443.0
}
```

A physical point may include one unit for both coordinates:

```json
{
  "space": "surface_mm",
  "x": 5.0,
  "y": 3.0,
  "unit": "cm"
}
```

Pixel-space points use pixel values and must not silently accept physical units. Camera-space points always name the camera; projector-space points must not name one. All values must be finite and booleans are not numbers.

A primitive has an anchor/endpoint space and, where it constructs dimensions around an anchor, a declared `shape_space` of `camera_px`, `projector_px` or canonical `surface_mm`. This permits mixed requests without implicit guesses:

```json
{
  "kind": "circle",
  "centre": {
    "space": "camera_px",
    "camera": "camera-1",
    "x": 812,
    "y": 443
  },
  "shape_space": "surface_mm",
  "radius": {"value": 10, "unit": "cm"}
}
```

The camera point is resolved to a projector point, then to the corresponding canonical metric point, and the true physical circle is constructed there before being projected. This requires both camera and metric calibration. It must never use a local pixel-to-mm approximation.

For a pixel shape, dimensions are in `px` and refer to the declared pixel shape space. Thus a `projector_px` radius is projector pixels and a `camera_px` radius is camera pixels. A `surface_edge_mm` anchor is first converted to canonical `surface_mm`; it cannot define physical shape dimensions or measurements directly. If the anchor and shape spaces differ, the anchor is converted into the declared shape space using an applicable inverse transform; missing inverse calibration is an explicit failure.

The same principle applies to rotated rects and grids. A line's two endpoints may independently use any supported point space; a ruler additionally declares the space in which its distance is measured.

## Transform and dependency rules

Resolution is explicit and fail-closed:

```text
projector_px point
  → direct projector-native point

camera_px point
  → selected camera's camera-native → projector-native transform

surface_mm point
  → shared metric surface-mm → projector-native transform

surface_edge_mm point
  → derived edge frame → canonical surface-mm
  → shared metric surface-mm → projector-native transform
```

A camera-space request requires only the selected camera's current camera/projector calibration and camera lifecycle availability. A physical request requires a current `CALIBRATED` shared metric record whose projector descriptor matches the running output. A mixed camera-anchor/physical-shape request requires both. A projector-space request requires neither calibration. Camera and metric records remain owned by their existing registries.

When a source calibration changes:

- camera-dependent generic overlays are removed or made non-renderable before the changed state is visible;
- metric-dependent generic overlays are removed before metric recalibration/reset and after metric staling;
- projector-dependent geometry is removed when the projector resolution or output identity changes;
- unrelated overlays remain intact where their dependencies are still valid.

No stale materialised projector geometry may remain visible after invalidation.

## `surface_edge_mm` frame

The edge frame is a point-reference convenience, not a replacement for ADR-0002's canonical frame and not a shape/measurement space.

Its contract is:

- `(0, 0)` is the canonical physical surface point obtained by inverse-projecting projector-native `(0, 0)`;
- its positive x direction is the physical direction of the projector's top output edge at that origin;
- its positive y direction is the physical direction of the projector's left output edge at that origin;
- the two axes may be oblique in the physical plane because perspective does not generally preserve right angles;
- coordinates along those axes are physical millimetres, not projector pixels or normalised UV values;
- every edge-frame anchor or endpoint is converted into ADR-0002's canonical Euclidean metric frame before constructing physical shapes, calculating physical distances or applying physical rotation; the conversion uses unit-length canonical basis vectors from the inverse-mapped projector top-left point towards the top-right and bottom-left points, so an x or y coordinate denotes physical millimetres along that oblique basis;
- the frame is unavailable if the projector output corners cannot be inverse-mapped to finite, consistent surface points.

This means direct projector alignment is available through `projector_px`, while physical values remain honest physical values. A perspective rectangle in projector pixels is not treated as a physical rectangle. Physical angle zero is the positive canonical surface x direction, and positive `angle_deg` rotates counter-clockwise in the physical plane (towards decreasing canonical surface y, whose stored coordinates increase downwards). Pixel-space angles use the same physical-screen convention: positive x right and positive y up for angle arithmetic.

## Primitive contracts

### Circle

```text
centre: point reference
shape_space: projector_px | camera_px | surface_mm
radius: positive quantity in shape-space units
angle_deg: optional metadata, normally irrelevant to a circle
style
```

A metric circle is sampled in canonical physical surface space before projection. Pixel circles are sampled in their declared pixel space before projection. Sampling is deterministic, uses a configured/default bounded tessellation tolerance and preserves the requested radius in its source space.

### Rotated rect

```text
centre: point reference
shape_space: projector_px | camera_px | surface_mm
width: positive quantity in shape-space units
height: positive quantity in shape-space units
angle_deg: finite angle, counter-clockwise, around centre, in declared shape-space orientation
style
```

The four source-space corners are generated around the centre, rotated counter-clockwise by `angle_deg` using the documented positive-x/right, positive-y/up angle convention, then transformed. Width/height are not altered to fit the projector. Filled rects render a clipped polygon; outline rects render each clipped source edge independently so clipping cannot invent a screen-boundary closing stroke.

### Grid

```text
origin: point reference, representing one grid intersection
shape_space: projector_px | camera_px | surface_mm
spacing: positive quantity in shape-space units
angle_deg: finite counter-clockwise rotation
extent: {width, height} in shape-space units, or omitted for automatic finite extent
style
```

The grid is square in its declared source space. Its local extent is finite. When extent is omitted, the service derives a sufficiently large finite source rectangle to cover the relevant projector output after transformation, then clips all generated lines; it never creates an infinite line collection. The origin/phase is stable in source coordinates and never aligned to rounded projector pixels.

For physical grids, spacing is exact in canonical metric space before projection. For camera/projector grids, spacing is in the declared pixel space.

### Line

```text
start: point reference
end: point reference
label: optional string
style
```

Endpoints can use different spaces if both can be resolved. A line is literal geometry and does not determine visibility, terrain or legality. It is clipped in projector space while retaining endpoint/request semantics in the response.

### Ruler

```text
start: point reference
end: point reference
measure_space: projector_px | camera_px | surface_mm
measurement_camera: required when measure_space == camera_px
unit: px for pixel measurement, mm|cm|in for physical measurement
label: optional override or default calculated label
style
```

A ruler is a specialised line with deterministic tick marks and a calculated distance. Physical measurement is computed in canonical metric space; projector measurement uses Euclidean distance in projector pixels; camera measurement uses Euclidean distance in the specifically named `measurement_camera`'s native pixels and records that camera as a dependency. Existing Plan4 physical ruler validation remains available and its observed physical errors remain independent operator observations.

Rulers use bounded tick generation in source measurement space. Tick decoration never changes the requested endpoints or measured length. Labels are decorative and may be clamped to the output surface.

## Style contract

The minimal public style is intentionally Pygame-friendly:

```json
{
  "colour": "#RRGGBB",
  "fill": false,
  "line_width_px": 2
}
```

`colour` is a strict six-digit HEX string and is normalised to an RGB tuple. `fill` controls filled versus outline rendering for circles and rects; lines and grid lines are outline-only and reject a supplied `fill: true`. `line_width_px` is a positive integer projector-raster width. Defaults are deterministic by primitive kind.

Alpha is not part of the initial public contract. The implementation should not invent translucent compositing that the current Pygame path cannot render reliably. Stroke/fill colour separation is also deferred; one colour keeps the first API small and direct.

## Generation budgets

Every request is validated against bounded configuration before it is inserted into the registry. Initial defaults are:

```text
max_overlay_vertices = 10000
max_overlay_segments = 5000
max_overlay_ticks = 200
max_overlay_label_characters = 256
```

The values must be positive, finite where applicable and preserved through configuration round trips. Circle tessellation, grid auto-extent, tick generation and label handling fail closed when a budget would be exceeded; they must not allocate an unbounded intermediate collection or partially mutate registry state. Tests may use smaller injected budgets to exercise the boundaries.

## Clipping and raster safety

Source geometry is generated at the requested size and transformed before output clipping. Projector clipping may remove invisible portions but must never shrink, re-centre or otherwise reinterpret requested physical/pixel geometry.

All final draw primitives must be finite. Lines are clipped to projector bounds; closed polygonal geometry is polygon-clipped; sampled curves are clipped after transformation. A valid entirely off-screen object may remain in registry state but produces no draw calls. Horizon crossings, non-finite transforms, degenerate geometry and unsafe projected primitives fail the request rather than producing plausible output.

Rounding to raster pixels occurs only after source geometry, projective transformation and clipping. The renderer must not recompute positions, dimensions or scales.

## Overlay registry and lifecycle

Each new object receives a UUID4 ID unless a test seam supplies an ID. A name is optional. Names are unique within the session and duplicate names are rejected; IDs remain the unambiguous machine reference. There is no implicit replacement. To change geometry, remove the object and create another.

The registry stores:

```text
id
optional name (no slash, not canonical UUID syntax)
kind
visible
immutable request/specification
materialised projector-native draw primitives
calibration/output dependencies
insertion sequence
```

ID and name selectors are never ambiguous: lifecycle routes use an explicit `id` or `name` path segment, and the service never guesses between them.

Supported state operations are:

```text
create
list
show(id-or-name)
hide(id-or-name)
remove(id-or-name)
clear(all)
```

Show/hide of an already-visible/already-hidden object is idempotent. Listing and rendering use deterministic insertion order, not UUID lexical order. Layer order is explicit by primitive category, then insertion order. Names are not required for operation; IDs returned by creation are sufficient.

The existing point overlay remains independently addressable through its current compatibility API and remains visually usable alongside generic objects. The existing metric-ruler API remains a compatibility adapter for one Plan4 physical ruler; generic ruler creation supports multiple named/UUID objects without duplicating calculation logic.

## API contract

The primary generic routes are:

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

Request schemas reject unknown fields, malformed nested references, invalid units/spaces, duplicate names, missing calibration dependencies, non-finite values, non-positive dimensions, invalid HEX colours and invalid angle/width values. Responses include ID, optional name, kind, visibility, original normalised request, style and service-produced projector draw primitives. JSON output contains no NaN or infinity.

Existing `/metric/ruler`, `/metric/calibration/*`, `/overlay/point` and camera/area APIs retain their established schemas and semantics. Generic endpoints delegate all geometry and state changes to `MultiVisionService`; API code must not open cameras, invoke Pygame or calculate homographies.

The CLI mirrors generic creation and lifecycle operations while remaining a thin HTTP client. The canonical creation form is `overlay <kind> --spec-json '<json>'`; lifecycle commands require exactly one of `--id` or `--name`. JSON specs are the compact way to express mixed coordinate references and styles; any later convenience flags must compile to the same request schema and must not implement geometry locally.

## Deterministic scenario matrix

Automated coverage must include:

- direct projector-pixel circle/rect/grid/line/ruler requests without calibration;
- camera-pixel requests using selected calibrated cameras and rejection for closed, unavailable, stale or wrong cameras;
- canonical physical requests using Plan4 metric calibration and rejection when uncalibrated/stale;
- camera-pixel anchor plus physical radius/width/height and rejection when either dependency is missing;
- projector-pixel anchor plus physical geometry;
- surface-edge point-anchor conversion under skewed perspective, with physical shapes constructed only after canonical conversion;
- circles sampled before transformation, proving physical radius and perspective distortion;
- rotated rects and grids with positive/negative angles and deterministic counter-clockwise convention;
- arbitrary grid origin/phase, exact source spacing, finite explicit extent and automatic finite extent;
- filled and outline-only closed shapes, HEX colours and projector-pixel line widths;
- output-bound clipping with requested dimensions preserved, independently clipped outline segments without synthetic boundary strokes, and entirely off-screen valid geometry producing no calls;
- mixed-space line endpoints and ruler measurement spaces, including required camera identity for camera-pixel measurement;
- named/unnamed UUID4 objects, duplicate names, insertion ordering, show/hide/remove/clear and no replacement;
- camera, metric and output-descriptor invalidation with unrelated-object preservation;
- calibration-pattern and metric-blank suppression, legacy point/ruler coexistence and display main-thread enforcement;
- API/CLI delegation, strict schemas, JSON-safe serialisation and no hardware ownership.

Synthetic and fake-Pygame tests prove software and mathematical behaviour only. They do not prove printer scale, tabletop flatness, projector placement, camera quality or physical dimensions.

## Manual acceptance boundary

The updated smoke procedure must record:

1. a rotated physical grid measured at separated tabletop positions;
2. filled and outline-only physical circles and rects, including a known physical radius/size;
3. physical lines and rulers at multiple locations/orientations;
4. a camera-pixel anchor overlaid with a physical radius, after both relevant calibrations;
5. direct projector-pixel geometry and its intentionally pixel-based semantics;
6. screen-edge clipping without shrinking the requested geometry;
7. named overlay show/hide/remove and coexistence of several objects;
8. explicit invalidation behaviour after camera, metric or projector changes.

No automated response, SVG, screenshot or fake projector output is evidence of physical accuracy.
