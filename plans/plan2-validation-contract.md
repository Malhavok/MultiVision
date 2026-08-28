# Plan2 validation contract

This sidecar defines the shared observables for `plans/plan2.md`. It is implementation guidance for plan2 and does not amend ADR-0001. Where the documents differ, the Driver's current plan2 requirements govern this plan.

## Implementation-facing scope and compatibility boundary

Plan2 changes camera identity and camera-state compatibility for the running session without editing ADR-0001. Implementations must treat the following as one boundary:

- **Session identity.** Startup enumeration creates immutable, deterministic `camera-N` slots. The initial capture-index snapshot is authoritative only for that `main.py` process; indexes must not be persisted or reused as identity after restart.
- **Session state.** Display names, calibration and overlay associations belong to the in-memory slot. Camera names and calibration transforms are not persisted or authoritative across sessions.
- **Capacity.** More than four devices may be discovered, but no more than four cameras may be open and displayed simultaneously.
- **Management surface.** Camera management is CLI-only from the operator's UI perspective – the CLI calls the running service, while Pygame is not a camera-management authority. Management must not require restarting `main.py`.
- **Discovery boundary.** The startup snapshot is fixed for the session. Devices connected after startup are not added, and a disconnected device is never silently replaced by another capture index.
- **ADR compatibility.** This boundary supersedes ADR-0001 only where its persisted stable camera bindings and persisted camera calibration conflict with plan2. The service remains the sole owner of capture handles, and shared coordinate, pointing, failure and test-seam contracts still apply.

## Session identity and discovery

- At service startup, enumerate the cameras that are connected at that time and create deterministic session slots `camera-0`, `camera-1`, and so on.
- A session slot is the authoritative camera identity for the current `main.py` process. Capture indexes are valid only for that session and must never be persisted or reused as identity after restart. The initial enumeration snapshot is the only source of slot-to-index mapping for that session; close/open operations do not rediscover or remap slots.
- The session may discover more than four devices, but no more than four may be simultaneously open and displayed. Closed slots remain visible in the inventory and can be opened later.
- Default display names equal their slot names. A rename changes only the display name; it does not move the slot, frame, calibration or overlay to another camera.
- Devices connected after startup are outside scope. A device disconnected after startup becomes unavailable and is never silently replaced by another device.

## Camera lifecycle

- `OPEN` means the runtime owns one persistent capture handle and retains the latest usable frame.
- `CLOSED` means the handle has been released and the slot is intentionally not acquiring frames.
- `UNAVAILABLE` means the expected current-session device cannot be acquired or has disconnected. It is an explicit failure state, not an invitation to try another index.
- Rename, close and open are service operations and must take effect without restarting `main.py`.
- Closing a camera releases its handle, removes its usable preview, clears any overlay owned by it and invalidates its calibration. Reopening the slot starts without calibration and requires a new calibration.
- A disconnect follows the same calibration invalidation and fail-closed spatial behaviour as close, but the disconnected slot is not automatically repopulated by a later hot-plug.

## Calibration and pointing

- Calibration is held in memory and belongs to a session slot. No camera binding, capture index, session name or calibration transform is authoritative across process restarts.
- A rename preserves the slot's calibration and overlay association. Close/reopen does not.
- A click inside a current camera preview always resolves to that slot's current preview transform and calibration. It must never borrow another slot's transform.
- An available, uncalibrated camera accepts the click as a diagnostic interaction but does not create a projector overlay; its Pygame preview receives a persistent red frame while it remains uncalibrated.
- The red frame is camera-preview UI state, independent of the projector overlay. It is cleared by successful calibration or close/reopen, and is not created by clicks outside the preview, unavailable-camera clicks or another camera's state.
- A calibrated camera retains the existing fail-closed checks for stale/invalid calibration, calibrated-region bounds, finite projection and projector bounds.

## CLI/API surface

The exact URL layout may follow existing conventions, but the capabilities must exist on the running service:

```text
GET    /cameras
POST   /cameras/{slot}/rename       {"name": "overhead"}
POST   /cameras/{slot}/open
POST   /cameras/{slot}/close
POST   /overlay/point
DELETE  /overlay
```

The CLI must expose equivalent operations, using session slots as unambiguous control identifiers. Management commands address slots; pointing may address either a slot or a unique current display name:

```text
multivision cameras list
multivision cameras rename camera-0 overhead
multivision cameras close camera-1
multivision cameras open camera-1
```

Pointing may accept the current display name or slot if the resolver is explicit and unambiguous; responses should include both slot and current name. Unknown slots, duplicate names, invalid transitions and four-active-camera overflow must return structured non-success responses without changing state.

## Rendering

- All active cameras up to the four-camera limit are rendered at once in deterministic slot order.
- Cards show the current display name, session slot, runtime state, calibration state and latest available diagnostics.
- Layout state is rebuilt or updated after rename, close, open and disconnect without stale frames or stale click bounds.
- The projector surface remains separate from the debug camera UI. Camera-management operations do not require Pygame teardown or process restart.

## Persistence and compatibility

- Existing persisted camera bindings and persisted calibration records are not loaded as session camera identity or geometry.
- Existing non-camera configuration that remains explicitly supported, such as projector resolution and calibration thresholds, may continue to load.
- A clean session with no applicable legacy camera state creates fresh `camera-N` slots. Legacy data must not cause a new slot to inherit another camera's old transform.

## Deterministic validation matrix

Tests must cover:

- zero, one, four and more-than-four discovered devices;
- deterministic slot creation and default names;
- rename success, duplicate-name rejection and rename preservation of calibration/overlay;
- close/open handle release, reopening without restart and calibration reset;
- open overflow, unknown slots, malformed requests and disconnect failure;
- no camera opened twice and no closed/unavailable camera read;
- dynamic preview layout and click bounds after every management operation;
- calibrated click selecting the correct per-slot transform;
- uncalibrated click producing a red preview frame without projector output;
- legacy persisted bindings/calibrations being ignored or safely invalidated;
- no hot-plug-in assumption and explicit restart requirement for newly connected devices.

## Manual acceptance boundary

Automated tests may validate session state, ownership, geometry and UI control flow, but not physical camera identity or projected accuracy. On the target Mac, record:

1. every startup slot's live view and assigned name;
2. close/open operations without restarting `main.py`;
3. independent calibration and pointing for at least two cameras;
4. persistent red framing before calibration and correct projector pointing after calibration;
5. a disconnect becoming unavailable without substituting another camera;
6. a newly connected camera remaining absent until the next startup.
