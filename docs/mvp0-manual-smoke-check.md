# Plan2 and Plan3 manual hardware smoke check

This is a hardware procedure for the plan2 session-local camera model and the
plan3 edge/available-area diagnostics, not an automated acceptance test. Run it
on the target Mac with the intended cameras and display/projector. Record the
commands and physical observations; fake camera tests and deterministic suite
results are not hardware evidence. The shared requirements are in the [plan2
validation contract](../plans/plan2-validation-contract.md) and [plan3
validation contract](../plans/plan3-validation-contract.md).

## 1. Start one session with the complete startup snapshot

From the project root, connect all devices currently available for the smoke
test before starting the service. Do not use `cameras bind`: plan2 assigns
immutable session slots (`camera-0`, `camera-1`, and so on) from this startup
snapshot. The capture indexes are valid only for this run, and names and
calibrations are not persisted.

Install dependencies if needed, then start the service and leave it running:

```sh
PYTHON="$PWD/.venv/bin/python"
"$PYTHON" -m pip install -r requirements.txt
"$PYTHON" main.py
```

In a second terminal, list the session inventory:

```sh
PYTHON="$PWD/.venv/bin/python"
"$PYTHON" -m multivision.cli cameras list
"$PYTHON" -m multivision.cli status
```

Wait until each initially `OPEN` slot reports `runtime_status: AVAILABLE` and a
positive `frame_counter` in the `cameras list` response. The Pygame window must
show live previews in ascending slot order. Each card shows its slot, current
display name, connection and calibration state, native resolution and any
available calibration metrics; use the list response for frame diagnostics.
Identify each `camera-N` slot from its live preview, match it to the physical
camera, and record:

```text
slot | display name | physical camera description/aim | capture index | resolution | first frame counter
```

The slot is the authority, not the capture index. If more than four devices
were connected, the remaining startup slots are discovered but initially
`CLOSED` because at most four cameras can be active. Record those slots too;
identify each one by opening it during the management check below rather than
assuming its order means a physical identity.

## 2. Manage slots through the CLI without restarting

Use slot IDs for management commands. Choose an initially open slot and record
its frame counter, then rename it through the running service:

```sh
SLOT=camera-0                 # replace with a slot from `cameras list`
NEW_NAME=overhead              # choose a unique name
"$PYTHON" -m multivision.cli cameras rename "$SLOT" "$NEW_NAME"
"$PYTHON" -m multivision.cli cameras list
```

Confirm in the same Pygame window and list response that the slot now has the
new name, its live physical view is unchanged, and its frame counter continues
increasing. Renaming must not move the preview or its session state.

Close an open slot through the CLI and observe that its preview disappears and
its state becomes `CLOSED`:

```sh
"$PYTHON" -m multivision.cli cameras close "$SLOT"
"$PYTHON" -m multivision.cli cameras list
```

Reopen that same slot, still without stopping or restarting `main.py`:

```sh
"$PYTHON" -m multivision.cli cameras open "$SLOT"
"$PYTHON" -m multivision.cli cameras list
```

Confirm that its preview returns, its state becomes `OPEN`/`AVAILABLE`, and a
new frame counter starts increasing. Confirm that the slot is still mapped to
the same physical camera. A close/reopen deliberately clears that slot's
calibration and overlay, so it must be recalibrated before pointing.

For initially closed slots, open them directly through the CLI to view and
identify them, then close and reopen them if desired. Never use a later
discovery or a changed capture index to reinterpret an existing slot.

## 3. Check the uncalibrated-click frame

Before calibrating a camera, choose an `OPEN` and `AVAILABLE` slot whose
calibration state is `UNCALIBRATED`. Click inside that camera's live preview in
the Pygame window. Expected observations:

- the click fails explicitly as an uncalibrated-camera point operation;
- no projector overlay is created;
- that camera's preview is persistently outlined with a red frame while it
  remains uncalibrated;
- another camera's preview is not red-framed; and
- a click outside a preview does not create a red frame.

Record the slot/name, the visible error, and the red-frame observation. This
red frame is debug-camera UI state, not the projector's red-circle overlay.

## 4. Calibrate and point two independently aimed cameras

Choose at least two different physical cameras with independently aimed views
of the known surface. Record which physical camera was used for each slot;
do not treat two names or two capture indexes as evidence of independent
cameras. With the calibration pattern visible on the projector/display, run
these commands against the same service:

```sh
CAMERA_A=camera-0              # replace with the first selected slot
CAMERA_B=camera-1              # replace with the second selected slot
"$PYTHON" -m multivision.cli calibrate --camera "$CAMERA_A"
"$PYTHON" -m multivision.cli calibration verify --camera "$CAMERA_A"
"$PYTHON" -m multivision.cli calibrate --camera "$CAMERA_B"
"$PYTHON" -m multivision.cli calibration verify --camera "$CAMERA_B"
"$PYTHON" -m multivision.cli calibration status
```

Capture each result's unique tag/corner counts, inlier count and ratio,
reprojection errors, coverage and final calibration status. The red frame for
each successfully calibrated camera must clear. If a camera has no retained
frame or cannot detect the pattern, wait for the live frame and retry against
the same running service; do not restart it as a calibration workaround.

Choose a physical point visible in both independently aimed views. Click that
point in each camera's preview and record the slot, current display name,
camera-native click coordinates and returned projector coordinates. The two
projector marks should land at approximately the same physical location.
Optionally confirm the same shared path with native-coordinate CLI requests:

```sh
printf 'Camera A native x y: '
read -r AX AY
printf 'Camera B native x y: '
read -r BX BY
"$PYTHON" -m multivision.cli point --camera "$CAMERA_A" --x "$AX" --y "$AY"
"$PYTHON" -m multivision.cli point --camera "$CAMERA_B" --x "$BX" --y "$BY"
"$PYTHON" -m multivision.cli overlay clear
```

The native coordinates need not be the same numbers for the two views. Record
the physical pointing observations rather than declaring accuracy from the
JSON responses alone.

## 5. Check edge-near tags and available-area diagnostics

During the calibration-pattern presentation in the preceding step, inspect the
projector/display rather than relying on the calibration JSON. Confirm that all
20 tags in the 5×4 layout are visible, that the outer tags reach as close as safely possible to
each configured usable-area edge, and that neither the tags nor their rendered
integer pixels are clipped. Do not record exact boundary contact when raster
rounding leaves a safety gap. Record the projector resolution, usable-area
insets, marker count, observed edge gaps and any clipping. While the pattern is
visible, the diagnostic area outlines and labels must be absent.

Choose two independently calibrated cameras whose useful physical views
actually overlap. Enable their areas through the still-running service:

```sh
"$PYTHON" -m multivision.cli cameras area enable "$CAMERA_A"
"$PYTHON" -m multivision.cli cameras area enable "$CAMERA_B"
"$PYTHON" -m multivision.cli cameras list
```

Confirm on the projector/display and in `cameras list` that each slot reports
`area_enabled: true` with an ordered finite `available_area`, its current
`name`, and an `area_colour`. The polygon is an estimated projection of the
full native camera frame, and the same native-frame footprint is pointable when
its projected point remains inside the projector bounds. The two outlines must
be visibly distinct and retain their colours where they overlap; both current
names must be legible.
The outlines are diagnostic only, are drawn before the existing red point
overlay, and do not change which camera can point. Repeat one of the known
point operations from section 4 and confirm that the red circle remains a
separate, visible overlay over the area outlines. Record each slot, name,
colour, polygon and the physical overlap observation. If the selected views do
not overlap, record that limitation instead of claiming an overlap result.

Disable one area and confirm that only its outline and label disappear. Repeat
the same native point request or preview click used in section 4; its calibration
status, point result and the other camera's outline must be unchanged:

```sh
"$PYTHON" -m multivision.cli cameras area disable "$CAMERA_A"
"$PYTHON" -m multivision.cli cameras list
```

Re-enable it, then rename that same slot without restarting the service:

```sh
RENAMED_AREA=area-camera-a
"$PYTHON" -m multivision.cli cameras area enable "$CAMERA_A"
"$PYTHON" -m multivision.cli cameras rename "$CAMERA_A" "$RENAMED_AREA"
"$PYTHON" -m multivision.cli cameras list
```

Confirm that the slot, enabled state, polygon and colour are preserved and that
only the displayed name and projector label changed. Use the actual new name
when recording the result; the slot remains the authority.

With the renamed area enabled, check every invalidation boundary. Start a new
calibration attempt for that slot and inspect the projector while the pattern
is visible:

```sh
"$PYTHON" -m multivision.cli calibrate --camera "$CAMERA_A"
"$PYTHON" -m multivision.cli cameras list
```

The old outline and label must disappear before/during the pattern presentation.
After a successful recalibration, the slot must remain area-disabled with no
available polygon until explicitly enabled again; after a failed attempt, the
old area must likewise not return. Record the actual calibration outcome rather
than treating either outcome as a pass. Then close and reopen the slot and
confirm that its area is still disabled and its old polygon is absent:

```sh
"$PYTHON" -m multivision.cli cameras close "$CAMERA_A"
"$PYTHON" -m multivision.cli cameras list
"$PYTHON" -m multivision.cli cameras open "$CAMERA_A"
"$PYTHON" -m multivision.cli cameras list
```

Recalibration is required before pointing or area enablement after close/open.
Do not infer restoration from a matching capture index or display name.

## 6. Unplug one startup camera and check fail-closed behaviour

Choose one identified startup camera, preferably one of the calibrated
cameras, and unplug it while `main.py` continues running. Then run:

```sh
UNPLUGGED_SLOT=camera-0        # replace with the unplugged slot
"$PYTHON" -m multivision.cli cameras list
"$PYTHON" -m multivision.cli snapshot "$UNPLUGGED_SLOT"
```

The list must retain the same slot and report `state: UNAVAILABLE`,
`runtime_status: UNAVAILABLE`, an explicit error message, `area_enabled: false`,
`available_area: null` and no usable frame metadata. The snapshot command must
return a non-zero result containing an explicit camera-unavailable error. The
live preview must not be replaced by another camera, the old diagnostic outline
and label must disappear, and the remaining cameras must retain their own slots
and views. Record the slot, error text, last frame counter, area invalidation
and the physical observation.

If the unplugged camera had a calibration, confirm that its calibration is no
longer usable and that a point request fails closed; do not accept a projected
point from it.

## 7. Record the intentional hot-plug boundary

While the same service is still running, connect a camera that was absent from
the startup snapshot. If no spare camera is available, reconnect the camera
just unplugged and check that its old slot does not recover automatically.
Run:

```sh
"$PYTHON" -m multivision.cli cameras list
```

Record the observation explicitly as:

```text
hot-plug-in: intentionally unsupported
new device added to this session: no
restart required for a newly connected device: yes
```

A later device must not silently appear as a new `camera-N`, and a disconnected
slot must not be remapped to it. A newly connected device can be considered
only after a deliberate next startup; do not restart during this check to make
it pass.

## 8. Deterministic automated validation boundary

From the project root, run the deterministic suite separately from the hardware
procedure:

```sh
PYTHON="$PWD/.venv/bin/python"
"$PYTHON" -m pytest
```

The tests may establish pattern placement and integer bounds, polygon geometry,
per-camera state, API/CLI schemas and fake-projector draw order. They must not
be used as evidence that tags are physically edge-near, cameras are physically
identified, projector alignment is correct, or two physical areas overlap.
Record the command and its output separately from the hardware observations; a
passing test suite does not mark any manual check above as passed.

## 9. Result record and acceptance boundary

Attach the `cameras list` and calibration/point command output to a log with the
Mac, date, connected hardware, slot-to-live-preview identifications, management
observations, calibration metrics, edge observations, area names/colours/
polygons, overlap observation, rename result, lifecycle invalidation results,
red-frame observation, unplugged-slot error and hot-plug observation. These are
the required physical observations from the [plan2 validation
contract](../plans/plan2-validation-contract.md) and [plan3 validation
contract](../plans/plan3-validation-contract.md).

This document does not claim that any manual step has passed. Do not claim
physical plan2 or plan3 acceptance until this procedure has been run on the
target Mac with the actual cameras and projector/display and the observations
have been recorded. Deterministic tests and fake-device runs do not substitute
for that record or claim any hardware-only result.

## Plan4 metric hardware procedure

This is a separate physical procedure for Plan4 metric calibration. Run it on
the target Mac with the intended projector/display, tabletop, at least two
identified cameras and a physical ruler. The requirements and evidence boundary
are in the [Plan4 validation contract](../plans/plan4-validation-contract.md).
Do not mix these observations with the Plan2/Plan3 checks above: a passing
camera calibration, generated SVG, API response or synthetic test is not
physical evidence of metric accuracy.

### 1. Generate and verify the printed target

From the project root, generate the target before starting metric calibration:

```sh
PYTHON="$PWD/.venv/bin/python"
TARGET="$PWD/metric-target.svg"
"$PYTHON" -m multivision.cli metric target generate --output "$TARGET"
```

Record the generated path, target format/version and marker family from the
artifact or later metric status response. Print the SVG with **100% / Actual
size** selected. Disable `Fit to page`, printer scaling, browser scaling and
any other automatic resizing. Do not treat the SVG dimensions as proof that the
printer preserved them.

Before continuing, place a physical ruler on the printed reference segment and
record:

```text
expected reference: 100.0 mm
measured printed reference: ______ mm
printing deviation: ______ mm
printer/model and actual-size setting: ____________________
```

If the printed reference is not 100.0 mm within the agreed physical tolerance,
stop and reprint it correctly. Do not calibrate with a scaled sheet or silently
fold its scale error into the projector calibration.

### 2. Place the target and calibrate the shared metric surface

Use the same running service and camera slots identified from the startup
snapshot. Confirm with `cameras list` and `calibration status` that the selected
observation camera is `OPEN`/`AVAILABLE`, has a currently `CALIBRATED`
camera-to-projector calibration, and uses the current projector resolution and
logical output identity. Record the camera slot and its physical description;
the metric transform is shared and is not owned by this camera.

Lay the verified target flat on the tabletop, fully within the intended useful
surface, and leave it there. Do not move, rotate, lift or realign it after this
point. Ensure the target is visible to the selected camera and do not manually
add projector marks to the capture. Start the metric capture:

```sh
CAMERA=camera-0                 # replace with the identified calibrated slot
"$PYTHON" -m multivision.cli metric calibrate --camera "$CAMERA"
"$PYTHON" -m multivision.cli metric status
```

The service should blank the normal projector layers, acknowledge that blank
frame, settle, and accept three consecutive target observations without target
movement or disagreement. Record the result only if it is `CALIBRATED`,
including all of the following from `metric status`:

```text
observation camera slot/generation: ____________________
target format/version and marker family: _______________
unique target markers and correspondence corners: _______
RANSAC inliers and inlier ratio: _______________________
mean fit error (mm) and maximum fit error (mm): ________
spatial coverage: ______________________________________
projector resolution/output identity: _________________
```

A failed or unstable capture is a failed attempt, not evidence of accuracy.
Retry only after correcting the physical setup; do not restore an older metric
record or move the sheet during a capture.

### 3. Project and physically measure rulers

Use surface-mm coordinates accepted by the service. Choose several safe pairs
whose projected endpoints are visible, then deliberately cover separated
positions and orientations: for example, horizontal segments on opposite sides,
a vertical segment, a diagonal segment and segments whose line, ticks and end
markers lie near different intended usable-area edges. The examples below are
only a shape guide; choose coordinates appropriate to the placed target and
projector bounds.

For each ruler, request the line and optionally record the independent physical
measurement in the same command:

```sh
"$PYTHON" -m multivision.cli metric ruler \\
  --from-mm X1,Y1 --to-mm X2,Y2 --unit mm \\
  --observed-length MEASURED --observed-unit mm
```

Use `cm` or `in` for `--unit` on at least one check if those display units are
part of the intended use. Put a physical ruler against each projected line,
measure the distance between its projected start and end markers, and record a
row for every position/orientation/edge case:

```text
ruler id | surface endpoints | display unit | requested length (mm)
         | measured length (mm) | absolute error (mm) | position/orientation/edge notes
```

The requested length comes from the two surface endpoints. The absolute error
must be the independently measured value compared with that request; it must
not be copied from `fit_error_mm`. If a request would put any line, tick or
marker outside the projector bounds, expect an explicit failure and choose a
safe geometry rather than accepting clipped or approximated output.

### 4. Verify another camera consumes the shared transform

Select a second physical camera with its own currently valid camera/projector
calibration. Do not run `metric calibrate` for it and do not create a second
metric transform. Keep the shared ruler visible, use the second camera's live
preview or camera-native point operation to identify the same physical endpoint
or surface location, and confirm that its projected point agrees with the
existing ruler. Then run `metric status` again and record that the same single
metric record, target version and transform remain in use; the observation-camera
field remains provenance for the original calibration, not camera-owned metric
state.

### 5. Clear and deliberately invalidate metric calibration

First check explicit clearing in the same running session:

```sh
"$PYTHON" -m multivision.cli metric clear
"$PYTHON" -m multivision.cli metric status
"$PYTHON" -m multivision.cli metric ruler \\
  --from-mm 100,100 --to-mm 200,100 --unit mm
```

Record `UNCALIBRATED`, the absence of a ruler, and the structured
`METRIC_UNAVAILABLE` failure. Run `metric clear` a second time to check that
clear is idempotent. No old transform or projected ruler may remain usable.

To exercise the separate stale path, re-establish a fresh calibrated metric
record, then, **without restarting the service**, use the supported same-process
projector descriptor control to change the projector resolution or logical
output identity. A restart is not a substitute: it produces a fresh
`UNCALIBRATED` session rather than testing descriptor invalidation. Record that
both camera and metric geometry are invalidated atomically:

```text
metric status: STALE
camera calibration status: STALE / not spatially usable
ruler: removed
```

Attempt both a metric ruler request and a camera-dependent metric recalibration.
Each must fail closed with an explicit stale/calibration error, and no stale
transform may be projected. If the target hardware build does not expose the
same-process descriptor control, record this stale check as **not exercised**
rather than claiming it passed or simulating it with a mock.

### 6. Keep physical evidence separate from synthetic tests

Maintain two separate records. The physical record contains the printer
settings and measured 100-mm reference, target placement, camera identities,
fit/status output, every physical ruler measurement and the clear/invalidation
observations above. The synthetic record may contain only deterministic software
checks, for example:

```sh
"$PYTHON" -m pytest
```

Synthetic tests can establish target geometry, homography behaviour, unit
conversion, state transitions and raster-safe rendering. They cannot establish
that the printer produced a 1:1 target, that the sheet was flat or stationary,
that the projector/table/camera geometry was physically unchanged, or that a
projected ruler matches a physical ruler. Do not claim hardware accuracy from
code, generated SVGs, screenshots, API responses, fake cameras/projectors or
mocks. This procedure records no manual or external check as passed until the
actual observations have been run and attached to the physical record.

## Plan5 generic geometry overlay procedure

This is a separate physical procedure for the coordinate-aware generic overlays.
Run it with the same identified projector, tabletop and camera slots after the
relevant camera and metric calibrations are current. The deterministic suite
and API/CLI responses prove software behaviour only; they do not establish
physical size, alignment or camera identity.

Create named and unnamed overlays through the running service with the canonical
thin CLI form. Use explicit extents and units in every request:

For example, choose coordinates inside the current calibrated surface and
adjust them to the tabletop before measuring:

```sh
GRID_SPEC='{"name":"grid-rotated","origin":{"space":"surface_mm","x":100,"y":100,"unit":"mm"},"geometry_space":"surface_mm","spacing":{"value":25.4,"unit":"mm"},"extent":{"width":{"value":200,"unit":"mm"},"height":{"value":150,"unit":"mm"}},"angle_deg":17}'
CIRCLE_SPEC='{"centre":{"space":"surface_mm","x":250,"y":180,"unit":"mm"},"geometry_space":"surface_mm","radius":{"value":76.2,"unit":"mm"},"style":{"colour":"#00ff00"}}'
RECT_SPEC='{"centre":{"space":"surface_mm","x":450,"y":180,"unit":"mm"},"geometry_space":"surface_mm","width":{"value":100,"unit":"mm"},"height":{"value":50,"unit":"mm"},"angle_deg":-12,"style":{"fill":true,"colour":"#0000ff"}}'
LINE_SPEC='{"start":{"space":"surface_mm","x":100,"y":100,"unit":"mm"},"end":{"space":"surface_mm","x":400,"y":300,"unit":"mm"},"label":"physical line"}'
RULER_SPEC='{"start":{"space":"surface_mm","x":100,"y":100,"unit":"mm"},"end":{"space":"surface_mm","x":200,"y":100,"unit":"mm"},"measurement_space":"surface_mm","unit":"cm"}'
CAMERA_CIRCLE_SPEC='{"centre":{"space":"camera_px","camera":"camera-0","x":812,"y":443},"geometry_space":"surface_mm","radius":{"value":76.2,"unit":"mm"}}'
PROJECTOR_CIRCLE_SPEC='{"centre":{"space":"projector_px","x":500,"y":400},"geometry_space":"projector_px","radius":{"value":100,"unit":"px"}}'
"$PYTHON" -m multivision.cli overlay grid --spec-json "$GRID_SPEC"
"$PYTHON" -m multivision.cli overlay circle --spec-json "$CIRCLE_SPEC"
"$PYTHON" -m multivision.cli overlay rect --spec-json "$RECT_SPEC"
"$PYTHON" -m multivision.cli overlay line --spec-json "$LINE_SPEC"
"$PYTHON" -m multivision.cli overlay ruler --spec-json "$RULER_SPEC"
"$PYTHON" -m multivision.cli overlay circle --spec-json "$CAMERA_CIRCLE_SPEC"
"$PYTHON" -m multivision.cli overlay circle --spec-json "$PROJECTOR_CIRCLE_SPEC"
"$PYTHON" -m multivision.cli overlays list
```

The camera example deliberately uses native camera pixels only as a point
reference; its 76.2-mm radius remains a surface measurement. Use the actual
session slot and live physical camera rather than assuming `camera-0`.

Record the returned IDs, names, normalised requests, projector resolution and
output identity. Confirm that the list response does not contain the internal
materialised primitive collection. Use `overlay show`, `overlay hide` and
`overlay remove` once by ID and once by name, then use `overlays clear`; record
that only the selected overlay changes and that insertion order is preserved.

For physical checks, place a verified flat target or other known-size reference
on the tabletop and record the physical measurement, not just the requested
JSON value. Check all of the following:

1. A rotated `surface_mm` grid with an explicit width and height, measuring its
   spacing at separated tabletop positions.
2. Known-size physical circles and rectangles, including an outline and a fill
   where useful, at separated positions and rotations.
3. Physical lines and rulers at horizontal, vertical and diagonal orientations.
4. A `camera_px` centre used for a physical circle or rectangle after both the
   camera/projector and metric calibrations are current. Record the camera slot,
   live physical camera and native point coordinates.
5. A `projector_px` circle or rectangle beside a physical overlay, confirming
   that its intentionally pixel-based size is different where expected.
6. Geometry crossing each projector edge, confirming that clipping does not
   change the requested physical size and does not add a screen-edge closing
   stroke. Record any entirely off-screen valid geometry and its zero-draw
   observation.
7. Several named and unnamed overlays coexisting while each show/hide/remove
   operation affects only its selected object.
8. Camera invalidation by closing, disconnecting or invalidating one camera;
   camera-dependent generic geometry must disappear while unrelated
   projector-only geometry remains.
9. Metric invalidation by clearing or recalibrating the metric surface;
   physical generic geometry must disappear while projector-only geometry
   remains.
10. A same-process projector resolution or output-identity change; all generic
    materialised geometry must be removed and no old geometry may reappear.

Use `overlays list` and the projector display after every invalidation. Record
one row for every physical measurement in a log like this; leave `result` as
`not exercised` until the observation has actually been made:

```text
check | request/source size | expected | measured | absolute error
      | position/orientation | camera slot and live identity | output identity | result
rotated grid, position A | 25.4-mm spacing, 200 x 150 mm | 25.4 mm | ____ mm | ____ mm
rotated grid, position B | 25.4-mm spacing, 200 x 150 mm | 25.4 mm | ____ mm | ____ mm
circle outline | 152.4-mm diameter | 152.4 mm | ____ mm | ____ mm
circle fill | 152.4-mm diameter | 152.4 mm | ____ mm | ____ mm
rectangle outline/fill | 100 x 50 mm at ____ degrees | 100 x 50 mm | ____ | ____
physical line, horizontal | ____ mm | ____ mm | ____ mm | ____ mm
physical line, vertical | ____ mm | ____ mm | ____ mm | ____ mm
physical ruler, diagonal | ____ mm / ____ | ____ mm | ____ mm | ____ mm
camera anchor + physical shape | ____ mm at camera px (____, ____) | ____ | ____ | ____
projector-pixel shape | ____ px | ____ px | ____ | ____
edge-clipped physical shape | ____ mm, edge ____ | unchanged | ____ | ____
```

For the grid, measure separated cells rather than only adjacent lines. For
circles and rectangles, record outline and fill separately when both are used.
For every line and ruler, include horizontal, vertical and diagonal positions;
measure the endpoints or markers with an independent physical ruler. Record
whether an edge case is partially clipped or entirely off-screen, and for a
valid off-screen-only request record the expected zero draw observation.

Record lifecycle observations separately so a physical measurement cannot hide
an invalidation failure:

```text
transition | dependent geometry | unrelated projector-only geometry | list/display result
show/hide/remove by id and name | ____ | ____ | ____
camera close/disconnect | removed | remains | ____
metric clear/recalibration | removed | remains | ____
projector resolution/output change | all removed | none remains | ____
```

Do not use mock draws, screenshots, synthetic homographies or deterministic
test results as evidence for any physical observation. If a hardware check is
unavailable, record it as `not exercised` rather than passing it by inference.

### Plan5 deterministic and integration record

Run the complete deterministic repository suite separately from the physical
log and attach its output:

```sh
PYTHON="$PWD/.venv/bin/python"
"$PYTHON" -m pytest
```

A passing suite verifies software contracts only — including source-space
construction, projector clipping, API/CLI delegation, renderer ordering and
invalidation. It does not fill any `measured` field above. During integration
review, confirm that the service remains the only camera/metric/projector
geometry authority, that the display consumes projector-native primitives only,
and that no game, AI, scene-graph, DPI or local-scale behaviour has been added.

The remaining physical uncertainty is the error between those mathematical
contracts and the actual installation: printer scaling, target flatness and
placement, projector/table movement, output identity, camera-to-live-device
identification, lens/capture error, raster stroke width and ruler-reading
error. Until the rows above contain observations from the target hardware,
Plan5 physical accuracy remains **not established**; automated tests and API
responses must not be reported as hardware acceptance.

## Plan6 planar tag hardware check

This is a separate physical check for Plan6's read-only tag inspection. The
requirements and response semantics are in the [Plan6 validation
contract](../plans/plan6-validation-contract.md) and the [tag-detection
implementation boundary](tag-detection-implementation.md). Run it against the
same running service and identified camera slots used above. Synthetic tests,
JSON values and fake devices do not establish physical detection or
projector correspondence.

### 1. Inspect tags in several poses and positions

Choose an `OPEN`/`AVAILABLE` session camera and record its slot, current name,
live physical camera and native resolution. For projector-space checks, its
camera-to-projector calibration must also be current. Use a printed
`DICT_5X5_1000` tag, keep it fully visible, and issue the explicit hardware
query:

```sh
multivision tags list --camera <camera> --dictionary DICT_5X5_1000
```

If `multivision` is not on `PATH`, use the repository wrapper without changing
the request:

```sh
"$PWD/bin/multivision" tags list --camera <camera> --dictionary DICT_5X5_1000
```

Repeat the query after placing the tag at separated tabletop positions — for
example, centre, left/right and near the usable-area edges — and after several
physical rotations such as approximately 0°, 45°, 90° and 135°. Allow the live
frame to settle before each individual query, but do not treat a later frame
or a retry as the same observation. Record the returned `frame_counter`,
`captured_at_seconds`, requested camera, resolved `camera_id`, dictionary,
tag ID, camera corners, centre, area and orientation.

`camera.orientation_degrees` is apparent camera-image geometry for the ordered
corner 0→1 edge. It must **not** be interpreted as physical table yaw: do not
use that value to claim the tag's physical rotation, camera pose or 3D pose.
The physical rotation is the separately recorded setup condition.

Where practical, place two physical copies carrying the same marker ID at
different positions or orientations and repeat the query. Confirm from one
response that both valid detections remain visible as separate entries in
`tags`; record the duplicate only if it was actually observed. Do not create a
physical duplicate solely to fill the record, and do not assume marker IDs are
unique.

### 2. Check projector-space correspondence on the calibration plane

With the camera calibration current, place the tag flat on the calibrated table
plane at several of the positions above. For each response, record the
projector `corners`, `centre`, `area_px` and independently calculated
`orientation_degrees`, together with the top-level and tag-level
`projection_status`. Compare the reported projector centre/corners with the
same physical positions established by the existing projected calibration
reference or another already-supported known surface reference. Record the
physical correspondence, not just agreement between numbers in the response.

The projector orientation is the orientation of the transformed 0→1 edge in
projector coordinates; it is not copied from the camera orientation. An empty
or unavailable calibration may still return raw camera geometry but must leave
projector geometry absent with its structured status. Do not treat either case
as a physical correspondence pass.

For a tag deliberately held materially above the calibration plane — for
example on a safe spacer — repeat one query if the hardware setup permits it.
Record that the existing planar homography has no height correction: projected
position, corners, area or orientation may disagree with the elevated tag, and
this is a known limitation rather than evidence of 3D pose support. Do not
claim above-plane accuracy or table-plane correspondence from this check.

Keep a physical record in this form, leaving `result` as `not exercised` until
an observation has actually been made:

```text
camera slot/name | live physical camera | resolution | frame counter | captured_at_seconds
position | physical rotation | tag IDs observed | duplicate IDs | camera centre/orientation
projector centre/corners/orientation | projection status | physical correspondence | result
```

### 3. Keep automated validation and physical evidence separate

Run the deterministic checks separately from the physical record:

```sh
"$PWD/.venv/bin/python" -m pytest
pi-harness validate --plan plans/plan6.md
```

During integration review, confirm that the CLI delegates to the service, the
persistent latest-frame camera owner is used, projector geometry comes from
the shared calibration authority, no overlays or persistent state change, and
no duplicate transform or dictionary logic or unrelated scope has been added.
These commands and the review verify software boundaries only; until the
physical rows above contain actual observations, Plan6 hardware results remain
**not established**.
