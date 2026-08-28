# MVP0 manual smoke checks

These checks are hardware procedures, not automated acceptance. Run them on the
target Mac with the intended cameras and display/projector; record the command
output and observations without treating fake-camera tests as hardware results.

## One-camera check

From the project root, install the declared dependencies if needed. The
service must have at least one persisted logical-camera binding before the
check. On a clean target Mac, discover device IDs and their stability without
opening cameras:

```sh
PYTHON="$PWD/.venv/bin/python"
"$PYTHON" -m pip install -r requirements.txt
"$PYTHON" -c '
from multivision.discovery import PlatformDeviceDiscovery
for device in PlatformDeviceDiscovery().discover_devices():
    print(device.device_id, device.name, f'stable={device.is_stable_id}')
'
```

If the configuration has no binding, start the service in one terminal:

```sh
"$PYTHON" main.py
```

In a second terminal, choose an entry marked `stable=True`, bind its ID, then
close and restart the first service because binding changes take effect on
restart. Do not bind an entry marked `stable=False`.

```sh
PYTHON="$PWD/.venv/bin/python"
printf 'Logical camera name: '
read -r CAMERA
printf 'Stable device ID: '
read -r DEVICE_ID
"$PYTHON" -m multivision.cli cameras bind "$CAMERA" "$DEVICE_ID"
# Close the UI in the first terminal, then start `main.py` there again.
```

Start the integrated service and UI (with the existing binding, or after the
restart above):

```sh
"$PYTHON" main.py
```

In a second terminal, inspect available devices and configured logical
cameras:

```sh
PYTHON="$PWD/.venv/bin/python"
"$PYTHON" -m multivision.cli cameras list
"$PYTHON" -m multivision.cli status
```

`cameras list` reports the currently discoverable devices and stable-ID
metadata; `status` reports the configured logical names. Choose a configured
logical camera from `status` (do not assume a particular name or number of
cameras), then enter it when prompted. The UI shows live previews but not a
frame counter, so wait until the service reports a positive `frame_counter`
before requesting the first snapshot:

```sh
printf 'Logical camera name from the list: '
read -r CAMERA
"$PYTHON" - "$CAMERA" <<'PY'
import json
import sys
import time
import urllib.request

camera_name = sys.argv[1]
deadline = time.monotonic() + 30
while True:
    with urllib.request.urlopen(
        'http://127.0.0.1:8000/cameras/' + camera_name + '/status',
        timeout=5,
    ) as response:
        status = json.load(response)
    print(status)
    if status['runtime_status'] == 'AVAILABLE' and status['frame_counter'] > 0:
        break
    if time.monotonic() >= deadline:
        raise SystemExit('No first frame within 30 seconds')
    time.sleep(1)
PY
"$PYTHON" -m multivision.cli snapshot "$CAMERA" --output /tmp/multivision-"$CAMERA".jpg
```
If the first request reports a frame-unavailable error, wait for the counter and
retry this command against the same running service; it must not reopen the
camera.

Before calibration, prepare the known surface: the service renders the
deterministic AprilTag pattern during calibration/verification. Alternatively,
place a flat printed copy of that pattern on the observed surface with its
projector-native scale and coordinates known. Then run:

```sh
"$PYTHON" -m multivision.cli calibrate --camera "$CAMERA"
"$PYTHON" -m multivision.cli calibration verify --camera "$CAMERA"
```

The UI should show the camera's live preview and an explicit
connection/calibration state. The service's `cameras list` or per-camera status
response should report a frame counter that continues increasing. The
calibration response should include tag/corner, inlier, reprojection-error and
coverage metrics; verification returns the resulting status and the separate
`calibration status` command exposes the persisted metrics. A missing camera
should remain `UNAVAILABLE` with an explicit error rather than being replaced
by another device. If calibration or verification races the first retained
frame and returns a frame-unavailable error, wait for the frame counter to
become positive and retry against the same running service; it must not reopen
the camera.

Pick a visible point whose camera-native coordinates can be read from the
preview/captured image. First click that point in the corresponding live UI
preview and confirm that a red circle appears at the mapped projector
location. Then enter the same measured native coordinates and run the CLI
request:

```sh
printf 'Camera-native x y: '
read -r CAMERA_NATIVE_X CAMERA_NATIVE_Y
"$PYTHON" -m multivision.cli point --camera "$CAMERA" --x "$CAMERA_NATIVE_X" --y "$CAMERA_NATIVE_Y"
```

Expected observation: the UI preview click and the CLI point request use the
same camera-to-projector path, and the red circle remains until replaced or
cleared. Capture the reported projected coordinates and all calibration
metrics. Then clear it:

```sh
"$PYTHON" -m multivision.cli overlay clear
```

The circle should disappear. Close the Pygame window (or press Escape) and
observe that the process exits without hanging. `Ctrl-C` in the service
terminal is an equivalent shutdown check. After restarting, run `cameras list`
and `calibration status` to check that bindings recover and persisted
calibrations are `UNVERIFIED` until verification succeeds.

## Cross-camera check

Use the same running service and enumerate every currently configured camera
without hard-coding a camera count:

```sh
PYTHON="$PWD/.venv/bin/python"
BASE_URL=http://127.0.0.1:8000
"$PYTHON" - "$BASE_URL" <<'PY'
import json
import sys
import time
import urllib.request

base_url = sys.argv[1]
deadline = time.monotonic() + 30
while True:
    with urllib.request.urlopen(f'{base_url}/cameras', timeout=5) as response:
        cameras = json.load(response)
    pending = [
        camera['camera']
        for camera in cameras
        if camera['runtime_status'] == 'STARTING'
        or (
            camera['runtime_status'] == 'AVAILABLE'
            and camera['frame_counter'] <= 0
        )
    ]
    if len(pending) == 0:
        print('All currently available cameras have a retained frame:', cameras)
        break
    if time.monotonic() >= deadline:
        raise SystemExit(f'No first frame within 30 seconds: {pending}')
    time.sleep(1)
PY
CAMERA_DATA=$(curl --fail --silent --show-error "$BASE_URL/cameras")
printf '%s\n' "$CAMERA_DATA" | "$PYTHON" -c '
import json
import sys
for camera in json.load(sys.stdin):
    print(camera["camera"], camera["runtime_status"])
'
CAMERAS=$(printf '%s\n' "$CAMERA_DATA" | "$PYTHON" -c '
import json
import sys
for camera in json.load(sys.stdin):
    if camera["runtime_status"] == "AVAILABLE":
        print(camera["camera"])
')
UNAVAILABLE_CAMERAS=$(printf '%s\n' "$CAMERA_DATA" | "$PYTHON" -c '
import json
import sys
for camera in json.load(sys.stdin):
    if camera["runtime_status"] != "AVAILABLE":
        print('{}: {} – {}'.format(camera["camera"], camera["runtime_status"], camera["error_message"]))
')
printf 'Unavailable configured cameras (recorded and skipped):\n%s\n' "${UNAVAILABLE_CAMERAS:-none}"
```

Calibrate and verify each available configured camera. The status output above
records unavailable configured cameras and the loop skips them, not silently
substituting another device:

```sh
while IFS= read -r CAMERA; do
    [ -n "$CAMERA" ] || continue
    "$PYTHON" -m multivision.cli calibrate --camera "$CAMERA"
    "$PYTHON" -m multivision.cli calibration verify --camera "$CAMERA"
done <<EOF
$CAMERAS
EOF
"$PYTHON" -m multivision.cli cameras list
```

For several physical points distributed across the usable surface, use each
camera preview that can see that point and click the same point. Record the
logical camera, camera-native click coordinates, returned projector coordinates,
calibration status, inlier ratio, reprojection errors and coverage for every
attempt. The red circles should land at approximately the same physical
location for cameras that share visibility; do not compare cameras that cannot
see that point. Also exercise one point near each useful calibrated-region edge
and confirm unsupported clicks fail explicitly rather than extrapolating.

These commands provide the procedure only. No hardware observation is claimed
until they have been run on the target Mac.

## Remaining hardware-only acceptance work

The following acceptance work remains to be run and recorded on the target Mac;
this repository does not claim any of it passed:

- confirm stable device IDs remain attached to the same physical cameras across
  reconnect and restart, and that a changed capture index cannot select another
  binding;
- confirm at least three intended cameras stay open simultaneously and their
  frame counters continue increasing without snapshot-triggered reopen;
- calibrate and verify each usable camera against the projector, recording
  tag/corner counts, inlier ratio, reprojection errors and coverage;
- measure click-to-projection accuracy at several shared physical points across
  cameras, including useful region edges, and record the empirical tolerance;
- exercise restart recovery, missing-camera failure and stale-calibration
  refusal with the actual devices.

Run the procedures above on the target hardware and attach the observed output
and measurements before declaring MVP0 physically accepted. The automated suite
and fake-device tests are not substitutes for these checks.
