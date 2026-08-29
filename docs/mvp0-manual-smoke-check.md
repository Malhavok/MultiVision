# Plan2 manual hardware smoke check

This is a hardware procedure for the plan2 session-local camera model, not an
automated acceptance test. Run it on the target Mac with the intended cameras
and display/projector. Record the commands and physical observations; fake
camera tests and deterministic suite results are not hardware evidence. The
shared requirements are in the [plan2 validation contract](../plans/plan2-validation-contract.md).

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

## 5. Unplug one startup camera and check fail-closed behaviour

Choose one identified startup camera, preferably one of the calibrated
cameras, and unplug it while `main.py` continues running. Then run:

```sh
UNPLUGGED_SLOT=camera-0        # replace with the unplugged slot
"$PYTHON" -m multivision.cli cameras list
"$PYTHON" -m multivision.cli snapshot "$UNPLUGGED_SLOT"
```

The list must retain the same slot and report `state: UNAVAILABLE`,
`runtime_status: UNAVAILABLE`, an explicit error message and no usable frame
metadata. The snapshot command must return a non-zero result containing an
explicit camera-unavailable error. The live preview must not be replaced by
another camera, and the remaining cameras must retain their own slots and
views. Record the slot, error text, last frame counter and the physical
observation.

If the unplugged camera had a calibration, confirm that its calibration is no
longer usable and that a point request fails closed; do not accept a projected
point from it.

## 6. Record the intentional hot-plug boundary

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

## Result record and acceptance boundary

Attach the `cameras list` and calibration/point command output to a log with the
Mac, date, connected hardware, slot-to-live-preview identifications, management
observations, calibration metrics, red-frame observation, unplugged-slot error,
and hot-plug observation. These are the required physical observations from
the [shared validation contract](../plans/plan2-validation-contract.md).

This document does not claim that any manual step has passed. Do not claim
physical plan2 acceptance until this procedure has been run on the target Mac
with the actual cameras and projector/display and the observations have been
recorded. Automated tests and fake-device runs do not substitute for that
record.
