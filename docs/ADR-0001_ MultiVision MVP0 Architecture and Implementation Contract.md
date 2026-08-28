# ADR-0001: MultiVision MVP0 Architecture and Implementation Contract

**Status:** Accepted for MVP0 implementation  
**System:** MultiVision  
**Scope:** MVP0  
**Decision type:** Architecture / runtime / calibration / implementation contract

---

# 1. Purpose

MultiVision is intended to become a spatial interaction layer between software agents and a physical tabletop.

It connects:

- multiple cameras observing a physical surface;
- one display/projector illuminating that surface;
- software capable of understanding camera images;
- software capable of drawing at physical locations visible to those cameras.

MVP0 deliberately does **not** attempt to understand games, miniatures, cards, tokens or other semantic objects.

MVP0 exists to prove one fundamental capability:

> A point selected in the image of any calibrated camera can be mapped reliably to the same physical point on the projector surface.

The user-facing acceptance test is:

1. a camera shows a physical point on the table;
2. the user clicks that point in the camera preview;
3. MultiVision transforms the click from camera coordinates to projector coordinates;
4. the projector draws a red circle;
5. the circle appears on the same physical location that was clicked.

If several cameras can see the same point, clicking that point through each camera should result in approximately the same projected location.

---

# 2. Initial hardware

The initial development environment contains:

- one iPhone exposed to macOS through Continuity Camera;
- two Sandberg cameras;
- the MacBook built-in camera;
- all camera feeds expected to support approximately 1920×1080;
- one external display/projector.

The cameras are heterogeneous.

They noticeably differ in:

- automatic white balance;
- exposure;
- colour response;
- field of view;
- lens distortion;
- latency.

MultiVision must not assume visually identical camera output.

Geometric calibration must be independent from colour matching.

---

# 3. Implementation constraints

MVP0 should favour a small, boring implementation over architectural generality.

Use:

- Python 3.12 or newer;
- OpenCV;
- Pygame;
- FastAPI;
- a small CLI client;
- standard library functionality where practical.

Prefer:

- simple modules;
- explicit state;
- threads over multiprocessing unless a concrete blocker is demonstrated;
- testable pure geometry functions;
- dependency injection only at hardware boundaries where it materially improves testing.

Do not introduce:

- a database;
- SQLAlchemy;
- message queues;
- event buses;
- plugin systems;
- MCP;
- object recognition;
- AI integration;
- game-specific concepts;
- 3D reconstruction;
- SLAM;
- camera intrinsic calibration unless measurement demonstrates it is necessary;
- asynchronous camera pipelines merely for architectural elegance.

Do not expand the scope without a measured MVP0 requirement.

Hardware-dependent functionality must have enough abstraction that geometry, calibration logic and API behaviour can be tested without physical cameras or a projector.

---

# 4. Runtime model

MultiVision is a persistent service.

The service exists primarily because camera devices must remain open.

Repeatedly opening cameras is undesirable because it:

- adds latency;
- may trigger Continuity Camera reconnection;
- resets camera state;
- restarts automatic exposure and white-balance adaptation;
- makes interactive use less predictable.

Therefore:

> The running MultiVision service is the sole owner of camera handles.

CLI commands and API requests must never independently open a configured camera.

---

# 5. Main-thread ownership

On macOS, assume SDL/Pygame window management and event processing need to remain on the process main thread.

Therefore the preferred process model is:

```text
MultiVision process
│
├── main thread
│   ├── Pygame event loop
│   ├── debug UI
│   └── projector renderer
│
├── camera workers
│   ├── Continuity Camera
│   ├── Sandberg A
│   ├── Sandberg B
│   └── MacBook Camera
│
├── calibration / OpenCV work
│
└── FastAPI worker
```

FastAPI must not dictate the overall process lifecycle.

If Pygame limitations make a clean multi-display implementation impractical in one process, splitting the projector renderer into a cooperating process is acceptable.

That change must remain internal.

It must not change:

- camera ownership semantics;
- API behaviour;
- CLI behaviour;
- calibration mathematics.

Do not redesign the system around this possibility pre-emptively.

---

# 6. Coordinate spaces

MultiVision explicitly distinguishes coordinate spaces.

## 6.1 Camera-native space

Each camera has its own native coordinates:

```text
camera_x = 0 .. width-1
camera_y = 0 .. height-1
```

All geometric calibration is based on native camera coordinates.

Preview size and UI layout must never become part of calibration.

---

## 6.2 Projector-native space

The projector/display framebuffer has coordinates:

```text
projector_x = 0 .. width-1
projector_y = 0 .. height-1
```

For MVP0, projector space is the canonical physical surface coordinate system.

---

## 6.3 Normalised surface space

The implementation should additionally support:

```text
x = 0.0 .. 1.0
y = 0.0 .. 1.0
```

This allows later APIs to remain independent from projector resolution.

Normalised coordinates are an interface convenience.

They do not replace native coordinates inside calibration calculations.

---

# 7. Camera model

Each camera has:

```text
stable identity
logical name
capture backend
native resolution
runtime status
latest frame
calibration status
camera → projector transform
projector → camera transform
calibration quality metrics
```

Logical names may include:

```text
overhead
side-left
side-right
macbook
```

These names are configuration.

They are not inferred from volatile camera indexes.

---

# 8. Camera discovery

Never persist assignments such as:

```text
camera 0 = overhead
camera 1 = side-left
```

Numeric capture indexes may change between boots or reconnects.

Provide:

```text
multivision cameras list
```

which exposes available device information.

Provide:

```text
multivision cameras bind overhead <stable-device-id>
```

Bindings should persist.

If a stable identifier is available from the platform, use it.

If OpenCV cannot reliably enumerate or identify macOS devices, use AVFoundation or another macOS-native discovery mechanism behind the camera abstraction.

Higher layers must not care which mechanism performs discovery.

A missing bound camera must become unavailable.

It must never silently inherit another device merely because that device received the same numeric index.

---

# 9. Camera lifecycle

After startup:

1. discover configured devices;
2. open each available bound camera;
3. begin continuously acquiring frames;
4. retain the latest usable frame;
5. allow automatic exposure and white balance to settle.

Exact frame synchronisation is not required in MVP0.

Each camera may have different latency.

MVP0 operates on static or slowly changing tabletop scenes.

---

# 10. Fiducial calibration pattern

The projector displays a known calibration pattern.

Use OpenCV-supported fiducial detection.

For MVP0, prefer:

```text
cv2.aruco
```

using an AprilTag-family dictionary such as:

```text
DICT_APRILTAG_36h11
```

This avoids introducing a separate detection stack unless testing demonstrates a reason to do so.

The projected pattern should contain approximately 9–12 uniquely identified tags distributed across the usable projection area.

Example:

```text
T01 ------ T02 ------ T03 ------ T04

T12                              T05


T11                              T06

T10 ------ T09 ------ T08 ------ T07
```

Exact tag count and spacing may be adjusted experimentally.

The key requirement is spatial coverage.

---

# 11. Partial camera coverage

A camera does not need to see the entire projector surface.

This is intentional.

A side camera may see only part of the table.

Calibration may succeed when only a subset of tags is visible, provided the visible tags give a geometrically useful distribution.

The implementation must not equate:

```text
enough points to mathematically solve a homography
```

with:

```text
a trustworthy calibration
```

A cluster of markers occupying a tiny portion of the image is weak calibration even if the matrix can technically be calculated.

---

# 12. Calibration correspondences

Every detected marker provides four corners.

For each visible marker corner, the system knows:

```text
projector coordinate
↔
camera coordinate
```

With 12 visible tags this gives up to 48 correspondence points.

Calibration should use all valid correspondences rather than reducing every tag to its centre.

---

# 13. Calibration algorithm

For each camera:

1. display the calibration pattern;
2. allow a short settling interval;
3. capture one or more frames;
4. detect known tag IDs;
5. obtain sub-image marker corner coordinates where practical;
6. associate detected camera-space corners with known projector-space corners;
7. estimate projector-to-camera homography using OpenCV;
8. use RANSAC to reject outliers;
9. calculate inverse camera-to-projector mapping;
10. calculate reprojection statistics;
11. calculate calibration coverage / geometry quality;
12. validate the result;
13. store both transform directions if valid.

Conceptually:

```text
H_pc:
projector → camera

H_cp = inverse(H_pc):
camera → projector
```

Both transforms should be available explicitly.

Do not repeatedly invert the matrix for every point operation.

---

# 14. Calibration quality

Calibration quality must be measurable.

At minimum record:

- number of unique tags detected;
- number of correspondence corners;
- RANSAC inlier count;
- inlier ratio;
- mean or median reprojection error;
- maximum reprojection error;
- spatial coverage metric.

Thresholds should be configuration rather than scattered constants.

Initial values may be conservative defaults and should be tuned from real hardware measurements.

MVP0 should record the metrics even if the exact final thresholds remain experimental.

---

# 15. Valid calibration region

Homography extrapolation outside the calibrated region is unsafe.

Therefore every calibration must store a valid region corresponding to the useful spatial support of detected markers.

A camera click outside that region must not silently produce a projected point.

The system should instead return a clear error such as:

```text
POINT_OUTSIDE_CALIBRATED_REGION
```

A small configurable margin may be allowed.

The implementation must prefer refusing a point over confidently projecting it into an unreliable location.

---

# 16. Lens distortion

A planar homography assumes approximately perspective-linear camera geometry.

Consumer cameras may introduce lens distortion, particularly near image edges.

MVP0 must not automatically add camera intrinsic calibration.

First:

1. perform normal homography calibration;
2. measure reprojection errors;
3. test click accuracy across the useful field;
4. identify whether error systematically increases near edges.

Only introduce intrinsic lens calibration / undistortion if measurement demonstrates that the simpler model is insufficient.

This is an explicit anti-overengineering decision.

---

# 17. White balance and exposure

Colour mismatch between cameras is expected.

Colour calibration is outside MVP0.

Automatic white balance and exposure are allowed to stabilise after cameras open.

Calibration code must not depend on cameras having matching colour response.

If auto-exposure or auto-WB visibly harms fiducial detection, the camera abstraction may later support:

```text
AUTO
LOCK_CURRENT
MANUAL
```

No manual colour-normalisation pipeline should be built for MVP0.

---

# 18. Pygame output modes

Pygame must support two conceptual outputs.

## 18.1 Debug camera UI

Display live previews for all active cameras.

Each preview must show:

- logical camera name;
- connection state;
- calibration state;
- native resolution;
- basic calibration quality when available.

The user must be able to click inside a camera preview.

---

## 18.2 Projector surface

The projector surface renders:

- calibration tags during calibration;
- overlays during normal operation.

MVP0 requires only one overlay primitive:

```text
red circle
```

The overlay system should nevertheless model overlays as objects rather than hardcoding the circle directly into calibration code.

Do not implement additional overlay types until required.

---

# 19. Preview coordinate mapping

Camera previews may be scaled or letterboxed.

A mouse click must therefore follow the complete conversion path:

```text
Pygame window coordinate
↓
preview-local coordinate
↓
camera-native coordinate
↓
camera → projector homography
↓
projector-native coordinate
↓
red circle
```

Clicks inside letterbox/padding areas must be rejected.

The transform from preview-local coordinates to camera-native coordinates must be tested independently.

Changing UI layout or preview size must never alter geometric calibration.

---

# 20. Click behaviour

When the user clicks inside a calibrated camera preview:

1. identify the clicked camera;
2. map UI coordinates to native camera coordinates;
3. validate that the point lies inside that camera's calibrated region;
4. transform the point into projector coordinates;
5. validate that the projected result is finite and within projector bounds;
6. draw a red circle centred on that projector coordinate.

The red circle should remain visible until:

- another point replaces it;
- the user clears the overlay;
- an API/CLI command clears it.

The same transformation path must be used for:

- GUI clicks;
- API point requests;
- CLI point requests.

There must not be separate geometry implementations for each interface.

---

# 21. Cross-camera validation

Cross-camera validation is a core MVP0 test.

Procedure:

1. choose a physical point visible in at least two cameras;
2. click that physical point in Camera A preview;
3. note the projected circle location;
4. click the same point in Camera B preview;
5. compare the resulting physical projection.

Repeat at several locations across the table.

This test reveals:

- calibration quality;
- lens-distortion problems;
- incorrect preview coordinate scaling;
- poor marker coverage;
- bad camera transforms.

Do not judge the system from a single point near the centre.

---

# 22. Persistent calibration

Calibration data may be saved between runs.

Saved calibration must include enough metadata to determine whether it is still applicable, including:

- stable camera ID;
- camera resolution;
- projector resolution;
- calibration version;
- transformation matrices;
- calibration metrics;
- timestamp.

However:

> Persisted calibration is not automatically trusted after restart.

A saved calibration should be considered:

```text
UNVERIFIED
```

until a lightweight verification succeeds.

Physical movement of either camera or projector invalidates geometry even if every device ID remains unchanged.

---

# 23. Calibration verification

Calibration verification is separate from full calibration.

Verification:

1. displays known fiducials;
2. captures camera frames;
3. detects marker positions;
4. predicts their positions using stored calibration;
5. measures current reprojection error.

If error remains below configured thresholds:

```text
CALIBRATED
```

Otherwise:

```text
STALE
```

and spatial operations from that camera must fail until calibration is renewed.

---

# 24. Service API

The MultiVision service exposes a local HTTP API.

Initial capabilities should remain small.

Example operations:

```text
GET  /health

GET  /cameras
GET  /cameras/{camera}/status
GET  /cameras/{camera}/snapshot

POST /calibration
POST /calibration/verify
GET  /calibration/status

POST /overlay/point
DELETE /overlay
```

The exact URL layout is not architectural.

The important rule is:

> External callers interact with MultiVision through stable capabilities, never by manipulating camera or renderer internals directly.

---

# 25. CLI

The CLI is the primary automation interface for MVP0.

It talks to the running MultiVision service.

Example:

```text
multivision status

multivision cameras list

multivision cameras bind overhead <device-id>

multivision calibrate

multivision calibration verify

multivision snapshot overhead

multivision point \
  --camera overhead \
  --x 842 \
  --y 517

multivision overlay clear
```

The CLI must not:

- open cameras;
- calculate independent homographies;
- initialise Pygame;
- duplicate service logic.

It is a thin client.

This allows later use from:

- shell scripts;
- tests;
- Codex;
- Luna;
- custom harnesses;
- future MCP adapters.

---

# 26. Failure philosophy

Spatial systems should fail explicitly rather than produce plausible but incorrect output.

The following conditions must reject the operation:

```text
camera unavailable
camera uncalibrated
calibration stale
point outside calibrated region
invalid homography
projected point outside output bounds
camera resolution changed
projector resolution changed
```

A different camera must never silently inherit another camera's calibration.

---

# 27. Implementation sequence

Implement MVP0 incrementally.

Do not build the complete system in one pass.

## Phase 1 — camera discovery

Implement:

- enumeration;
- stable identity where possible;
- logical binding;
- persistent configuration.

### Done when

The target Mac can enumerate the intended cameras and display stable identifying information across normal reconnect/restart scenarios.

Hardware confirmation is manual.

---

## Phase 2 — persistent camera capture

Implement persistent open handles and latest-frame acquisition.

### Done when

Configured cameras remain open simultaneously for an extended smoke test without being reopened per snapshot.

A suggested manual test is at least 10 minutes.

The program must expose frame counters or timestamps so continued capture can be observed.

---

## Phase 3 — Pygame camera previews

Render live previews from every open camera.

Correctly handle:

- scaling;
- aspect ratio;
- letterboxing;
- click-to-native-coordinate conversion.

### Done when

A synthetic/native pixel target can be clicked through a scaled preview and maps back to the expected native camera coordinate.

This mapping must have automated tests.

---

## Phase 4 — projector calibration pattern

Render the fiducial pattern on the selected output surface.

### Done when

Known projector coordinates for every marker corner are deterministically available to the calibration subsystem.

---

## Phase 5 — marker detection

Detect fiducials independently for each camera.

### Done when

Recorded or synthetic test frames can identify marker IDs and corner positions.

Hardware smoke tests should confirm that each real camera detects a useful subset of projected markers.

---

## Phase 6 — homography

Calculate calibration from detected corners.

### Done when

Automated synthetic tests using known transforms recover a homography within expected numerical tolerance.

Tests must cover:

- perspective transformation;
- noisy correspondences;
- RANSAC outliers;
- insufficient spatial coverage;
- points outside calibrated regions.

---

## Phase 7 — GUI click → projector circle

Connect:

```text
preview click
→ native camera point
→ homography
→ projector point
→ red circle
```

### Done when

The target Mac demonstrates successful physical click-to-projection for at least one camera.

This criterion cannot be considered passed using mocks alone.

---

## Phase 8 — multi-camera calibration

Calibrate multiple physical cameras independently.

### Done when

At least three available cameras can calibrate against the same projector surface and can independently point to shared physical locations.

---

## Phase 9 — minimal API and CLI

Expose the already-working capabilities externally.

### Done when

The following can be performed from the CLI without touching the GUI internals:

```text
status
camera list
calibrate
verify calibration
point using a camera-space coordinate
clear overlay
```

---

## Phase 10 — persistence and verification

Persist bindings and calibration.

Require verification before trusting geometry after restart.

---

# 28. Automated test requirements

Automated tests must not require physical cameras.

Provide fakes or fixtures for:

- camera frames;
- device discovery;
- projector dimensions;
- calibration correspondences.

At minimum test:

- preview-to-native coordinate conversion;
- homography estimation;
- homography inversion;
- projection round trip;
- rejection of degenerate transforms;
- rejection of out-of-region points;
- RANSAC outlier handling;
- calibration status transitions;
- projector-resolution invalidation;
- camera-resolution invalidation.

Do not attempt to mock hardware so thoroughly that mocks are mistaken for hardware acceptance.

---

# 29. Manual hardware test boundary

Luna or another coding agent may implement the complete hardware path but must never claim that physical acceptance criteria passed unless the commands were executed on the target Mac with the actual hardware.

When reaching a hardware-dependent checkpoint, provide:

1. the exact command to run;
2. the expected visible behaviour;
3. relevant metrics/output to capture;
4. likely failure modes.

Then continue with work that does not depend on the unknown physical result where possible.

Do not fabricate successful hardware results.

---

# 30. MVP0 acceptance criteria

MVP0 is complete only when all of the following are demonstrated on the target hardware.

## Cameras

At least three intended cameras can remain open simultaneously.

Repeated snapshots do not reinitialise the devices.

---

## Preview

Pygame provides usable live previews of all open cameras.

---

## Calibration

At least three cameras independently calibrate against the same projector/display.

A camera may see only part of the projected surface.

Calibration quality metrics are visible.

---

## Spatial pointing

For each calibrated camera:

1. click a visible surface location in its preview;
2. transform that click into projector space;
3. render a red circle;
4. observe the circle at approximately the physical point clicked.

---

## Cross-camera consistency

Several physical test points are selected across the usable surface.

For points visible in multiple cameras, different cameras should produce approximately the same physical output location.

The system must record measured error rather than merely report success.

The final acceptable physical tolerance is determined empirically during MVP0.

Do not hardcode an assumed millimetre tolerance before measurement.

---

## Recovery

Camera bindings survive restart.

Persisted calibration is verified before use.

Missing cameras or stale calibration fail explicitly.

---

# 31. Non-goals

MVP0 does not include:

- miniature recognition;
- token recognition;
- card recognition;
- board recognition;
- game state;
- tracking;
- optical-flow pipelines;
- frame synchronisation;
- camera colour matching;
- projector colour calibration;
- 3D geometry;
- object-height correction;
- board-cell coordinates;
- semantic areas;
- automated camera selection;
- agents;
- LLMs;
- MCP.

---

# 32. Adversarial risks

## 32.1 Numerically valid but bad homography

A homography can exist despite poor marker distribution.

Mitigation:

- track spatial coverage;
- track reprojection error;
- reject weak calibration;
- test physical points throughout the usable region.

---

## 32.2 Lens distortion

Edge error may remain despite low average reprojection error.

Mitigation:

- measure local physical error;
- do not introduce intrinsic calibration until required.

---

## 32.3 Auto-exposure reacting to projected calibration

Bright markers may alter camera exposure while calibration is captured.

Mitigation:

- wait briefly after displaying the pattern;
- optionally analyse several frames;
- later add temporary exposure lock only if necessary.

---

## 32.4 Preview scaling bug masquerading as bad calibration

An incorrect UI-to-camera conversion could produce systematic projection errors even when homography is perfect.

Mitigation:

- isolate and unit-test preview coordinate conversion.

---

## 32.5 Extrapolation outside observed geometry

A camera may see regions beyond the fiducial support area.

Mitigation:

- explicitly maintain calibrated region;
- refuse unsupported clicks.

---

## 32.6 Projector or camera movement

A mechanically moved device makes persisted calibration incorrect.

Mitigation:

- calibration verification;
- explicit STALE state;
- fail closed.

---

## 32.7 Continuity Camera behaves differently from USB cameras

Device lifecycle and stable identity may differ.

Mitigation:

- camera backend abstraction;
- platform-native discovery if required;
- keep differences below the camera interface.

---

## 32.8 Pygame multi-display limitations

SDL/Pygame behaviour on macOS may complicate simultaneous debug UI and projector rendering.

Mitigation:

- keep Pygame on the main thread;
- if necessary isolate projector rendering in a cooperating process;
- do not let this alter external architecture or camera ownership.

---

# 33. Understanding invariant

Any implementation claiming to satisfy this ADR must preserve the following conceptual model:

```text
PHYSICAL SURFACE
       ↑
       │ projector
       │
PROJECTOR SPACE
       ↑
       │ per-camera homography
       │
CAMERA-NATIVE SPACE
       ↑
       │ preview transformation only
       │
DEBUG UI
```

The debug UI is not spatial truth.

Camera indexes are not identity.

Colour is not geometry.

The API is not the implementation.

The CLI is not the service.

Mocks are not hardware validation.

Most importantly:

> The value of MVP0 is not that cameras can display video or that OpenCV can calculate a homography.

> MVP0 succeeds only when a user can click the same physical point through different camera views and MultiVision reliably points back to that same physical location.

---

# 34. Expected next milestone

Only after MVP0 is physically validated should MVP1 introduce a canonical tabletop coordinate space and higher-level primitives such as:

```text
point_at(x, y)
capture_region(...)
show_region(...)
best_view(...)
```

Game-space coordinates, board understanding and AI integration come later.