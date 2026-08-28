# MVP0 validation contract

## Authority and scope

`docs/ADR-0001_ MultiVision MVP0 Architecture and Implementation Contract.md` is the accepted authority. This sidecar makes its implementation contracts explicit for the harness; it does not add MVP0 features.

Required outcome: a user can click a physical point in a calibrated camera preview and see a red circle at the corresponding projector location. Camera count is configuration-driven: the implementation supports one or more cameras without hard-coded minimum, maximum or assumptions about exactly three devices. Multiple configured cameras must independently map shared physical points consistently when available. The initial physical smoke test may use one camera and a screen or flat printed calibration sheet as the known surface; that validates the mapping path but not cross-camera consistency. Physical acceptance is manual; automated tests must never imply hardware validation.

Explicit non-goals: semantic/object recognition, tracking, frame synchronisation, colour matching/calibration, 3D or object-height correction, board/game concepts, agents, LLM/MCP integration and intrinsic lens calibration before measured homography error requires it.

## Configuration boundary

Configuration is a single, standard-library-readable file with one overridable path for tests; a simple platform-appropriate user configuration location and JSON representation are provisional implementation choices. It contains logical bindings and calibration thresholds/metadata, not a database or a second runtime state authority.

## Ownership and lifecycle

- The running service is the sole owner of camera handles. CLI and API code request capabilities through the service and never open cameras or calculate transforms.
- Discovery, binding, capture, calibration, API and rendering are separate small boundaries. Platform-specific macOS discovery stays behind the discovery boundary.
- Camera workers may run in threads and retain the latest usable frame plus an observable counter or timestamp. Pygame window creation, event handling and projector rendering stay on the process main thread.
- A different device must never inherit a logical camera's binding or calibration because it received the same numeric capture index.
- Startup opens every available configured camera and marks missing ones unavailable; adding or removing configured cameras must not require code changes. Shutdown releases every capture handle and stops workers/API work cleanly.

## Coordinate and point contracts

- Camera-native coordinates use each device's native width and height. Calibration never uses preview dimensions.
- Projector-native coordinates are the canonical physical surface coordinates. Normalised `(x, y)` values in `[0.0, 1.0]` may be an API convenience and must convert through projector resolution.
- A preview click converts window coordinate → preview-local coordinate → camera-native coordinate. Scaled previews preserve aspect ratio with explicit letterbox/padding; clicks in padding are rejected.
- The shared point path validates camera availability, `CALIBRATED` status, calibrated-region membership, finite homography output and projector bounds before creating an overlay. Errors are explicit, including `POINT_OUTSIDE_CALIBRATED_REGION`; no plausible extrapolation is accepted.
- GUI, API and CLI point operations reuse the same geometry and point service. There is one authoritative transform per camera and both directions are stored (`projector → camera` and `camera → projector`).

## Calibration contract

- The pattern uses approximately 9–12 uniquely identified OpenCV-supported AprilTag-family markers, preferably `DICT_APRILTAG_36h11`, distributed across the usable projection area. Exact count/spacing remain tunable, but coverage must be recorded.
- Every valid detected marker contributes its four corners. Correspondences retain marker identity and corner ordering and are independent per camera.
- Calibration uses OpenCV homography estimation with RANSAC, stores the inverse explicitly, and records unique tag count, correspondence-corner count, RANSAC inlier count/ratio, mean or median reprojection error, maximum reprojection error and a spatial coverage metric.
- Quality thresholds are configuration, not scattered constants. A mathematically solvable but tightly clustered marker set is rejected as weak calibration. The valid region represents useful marker support, with only a small configurable margin.
- Persisted calibration stores stable camera ID, camera resolution, projector resolution, calibration version, both matrices, metrics and timestamp. Loaded data starts `UNVERIFIED`; verification against known fiducials is required before use. Verification below configured thresholds gives `CALIBRATED`; failure gives `STALE`.
- Camera/projector resolution changes, invalid matrices, insufficient points, failed detection and failed verification prevent spatial operations. Lens undistortion is deferred unless measured useful-field error proves the homography insufficient.

## Interface contracts

The local service exposes the stable capabilities represented by these operations, regardless of exact URL spelling:

- health;
- camera list, per-camera status and snapshot;
- full calibration, calibration verification and calibration status;
- point using a camera and camera-space coordinate;
- clear/replace red-circle overlay.

The CLI is a thin HTTP client for those capabilities: `status`, `cameras list`, `cameras bind`, `calibrate`, `calibration verify`, `snapshot`, `point` and `overlay clear`. It must report service/error responses and must not initialise Pygame or duplicate business logic.

The only required overlay primitive is a red circle. It remains until replaced or explicitly cleared. Calibration code does not hardcode rendering, and rendering does not become geometric truth.

## Automated and manual validation

Automated tests use fake discovery/capture, recorded or synthetic frames, synthetic projector dimensions and known transforms. They cover preview conversion, homography/inversion/round-trip, noise and RANSAC outliers, degenerate/weak/out-of-region cases, status transitions, persistence round-trips and resolution invalidation, endpoint/CLI failure paths and capture-handle ownership.

Tests must be deterministic, run without physical devices, and use the repository's `.venv/bin/python` from the project root. Hardware-dependent tests are separate smoke checks.

Before claiming physical progress, provide the exact command, expected visible behaviour, metrics/output to capture and likely failure modes. The first manual smoke test requires only one camera and either a display/projector screen or a flat printed calibration sheet with known coordinates; it checks marker detection/calibration and click-to-point mapping. The current hardware check should exercise all three configured cameras when available, while the implementation and procedure must work unchanged with fewer or more cameras; shared-point comparisons require only the cameras that can see each chosen point. Restart binding recovery and verification/stale failure behaviour remain required. No agent may claim any of these observations without running them on the target Mac and hardware.
