# ADR-0004: Realtime Overlay Runtime and Spatial Anchors

**Status:** Proposed — ready for planning  
**System:** MultiVision  
**Scope:** Realtime overlay updates, spatial anchors, fiducial-aware projection and display decoupling  
**Decision type:** Runtime / vision / renderer / API contract

---

# 1. Purpose

ADR-0003 introduced generic physical geometry overlays defined in projector or metric surface coordinates.

The current implementation is adequate for relatively static diagnostic geometry, but it is not yet suitable for interactive tabletop use where overlays must react continuously to physical objects and external clients.

Observed problems include:

- repeated overlay mutations are far too slow for interactive use;
- invoking separate create/update operations for many overlays causes unnecessary overhead and intermediate partially-updated states;
- no arrow primitive exists;
- overlay geometry is effectively materialised too early for objects whose position changes continuously;
- fiducial markers can be visually contaminated by projector output, causing the vision system to lose the very marker that an overlay depends on;
- camera preview rendering may consume significant CPU/GPU time even though preview is primarily a diagnostic aid;
- fiducial identity cannot be treated as a globally unique numeric marker ID because different marker groups may reuse IDs or use different dictionaries/physical marker sizes.

ADR-0004 turns overlays into a **realtime spatial presentation subsystem** while preserving the generic, game-agnostic boundary from ADR-0003.

The user-facing capabilities enabled by this ADR include:

- update projected geometry interactively at at least 30 state changes per second;
- create or replace many overlays atomically in one batch;
- draw arrows as a first-class primitive;
- attach an overlay to a detected fiducial marker;
- draw a line or arrow between two fiducial markers;
- draw an arrow between a fixed table position and a moving fiducial, e.g. continuously indicating where a tagged card should be placed;
- attach an overlay with a marker-local offset/orientation so it moves and rotates with the physical tagged object;
- dim projected content and keep projector overlays from destroying active fiducial detection;
- keep camera capture/tracking active while camera preview runs at an independent lower rate or is disabled.

Board-shape or rectangle detection is explicitly outside ADR-0004.

---

# 2. Core decision

MultiVision separates an overlay's **declarative specification** from its **current frame materialisation**.

Conceptually:

```text
client/API
    ↓
overlay specification
    ↓
overlay registry
    ↓
resolve against latest SpatialState
    ↓
projector-native primitives
    ↓
projector renderer
```

Static overlays may resolve to identical primitives across many frames.

Dynamic overlays, such as an arrow whose endpoint is attached to a fiducial, are resolved against the newest usable spatial observation without requiring the client to recreate the overlay every frame.

The renderer must not perform camera capture or fiducial detection.

The vision/tracking path publishes immutable or snapshot-like spatial state which the overlay resolver consumes without waiting for a new camera frame.

---

# 3. Realtime performance contract

The realtime requirement applies to the **running service**, not to repeatedly cold-starting the command-line process.

The service must support at least:

```text
30 accepted and published overlay state mutations / second
```

under normal runtime conditions with:

- camera capture active;
- normal fiducial tracking active;
- projector rendering active;
- camera preview enabled at its normal diagnostic refresh rate.

The implementation target is interactive behaviour without visible multi-frame stalls during ordinary overlay updates.

A benchmark must distinguish:

```text
CLI process startup latency
HTTP transport latency
request validation latency
overlay mutation latency
spatial resolution/materialisation latency
projector presentation latency
```

The implementation must not claim failure of the realtime requirement merely because spawning 30 independent Python CLI processes per second is slow.

Likewise, it must not claim success solely from an in-process registry microbenchmark while the running projector/camera system visibly stalls.

No arbitrary sleeps, debounce delays or reduced camera operation may be introduced merely to make the benchmark pass.

---

# 4. Batch overlay mutations

The service must support atomic batch mutation of overlays.

A batch may contain many create/update/remove operations, or a replacement set where that API shape is simpler and consistent with the existing registry.

The important semantics are:

```text
validate complete batch
↓
resolve required static dependencies
↓
commit complete new registry state
```

If any operation in the batch is invalid, the batch fails without publishing a partially modified overlay state.

The projector must never intentionally expose an intermediate sequence such as:

```text
frame N:     overlays A B C
frame N + 1: overlays A B C D
frame N + 2: overlays A B C D E
frame N + 3: overlays A B C D E F
```

when the caller requested `D E F` as one logical batch.

Instead the observable transition should be:

```text
old committed state
↓
new committed state
```

Batch limits must be explicit and bounded to prevent an untrusted or accidental request from generating unbounded geometry work.

The CLI should expose batch submission conveniently, preferably through a structured JSON document and/or stdin. Exact syntax is an implementation detail.

---

# 5. Arrow primitive

ADR-0004 adds a first-class `arrow` overlay request.

At minimum an arrow contains:

```text
start anchor
end anchor
style
head length / default
head width / default
optional label
```

An arrow is conceptually geometry composed from existing low-level projector primitives:

```text
shaft = segment
head = triangle/polygon
```

The projector renderer does not need a specialised raster path if the existing segment/polygon renderer can represent the result cleanly.

Arrowhead geometry must remain stable as endpoints move and must fail explicitly for degenerate zero-length arrows rather than producing NaN/invalid projector coordinates.

Physical dimensions such as a metric arrowhead size must follow the coordinate/units model from ADR-0003 rather than assuming constant projector pixel scale.

---

# 6. Spatial anchors

Dynamic overlay endpoints are represented by **spatial anchors**.

An anchor identifies a spatial reference whose current projector position and, where relevant, orientation can be resolved by MultiVision.

ADR-0004 requires at least the following anchor forms.

## 6.1 Fixed surface anchor

A fixed point in the shared metric tabletop coordinate system:

```text
surface_mm(x, y)
```

This anchor remains fixed until metric calibration becomes unavailable/stale or the overlay specification changes.

## 6.2 Projector-space anchor

Where already supported and useful for diagnostics, a fixed projector-native point may remain available.

It is not a substitute for a metric surface anchor when physical meaning is required.

## 6.3 Fiducial anchor

A point derived from the current observation of a named fiducial:

```text
fiducial(group, id)
```

The default point may be marker centre, with optional explicit marker-local offsets.

## 6.4 Marker-local anchor

A fiducial anchor may additionally define a local offset and orientation relationship:

```text
fiducial(group="cards", id=17)
local_offset_mm=(20, 0)
follow_rotation=true
```

This allows an overlay to move and rotate with a physical tagged object rather than only follow its centre.

Marker-local geometry must use the detected marker pose/orientation on the calibrated table. It must not infer orientation from projector pixels independently of the vision/calibration model.

---

# 7. Namespaced fiducial identity

A numeric marker ID alone is not a valid global identity.

All fiducial-backed anchors use the compound identity:

```text
(group, id)
```

For example:

```text
("cards", 12)
("units", 12)
```

are distinct fiducials.

A fiducial group is a configured namespace and may define properties required to interpret its members, such as:

```text
name
dictionary / marker family
physical marker size
tracking defaults
```

The exact configuration schema may follow existing MultiVision configuration conventions.

The group is semantically significant, not a UI label.

Marker lookup, tracking state, protected projector regions and anchor resolution must all use the compound `(group, id)` identity.

Do not silently fall back to `id`-only matching when the requested group is unavailable.

---

# 8. Two-anchor geometry

Line and arrow overlays may resolve their two endpoints independently.

Supported combinations include at least:

```text
surface → surface
surface → fiducial
fiducial → surface
fiducial → fiducial
```

This deliberately enables the destination-guide use case:

```text
fixed destination on table
        ↓
      arrow
        ↓
currently detected tagged card
```

As the card moves, only the fiducial endpoint moves. The caller does not continuously resend coordinates.

Two fiducial endpoints may belong to different groups.

The implementation must not assume that both endpoints originate from the same camera. The spatial state/resolution layer owns any multi-camera observation choice required to provide one authoritative tabletop position.

---

# 9. SpatialState

ADR-0004 introduces a service-owned current spatial snapshot, named `SpatialState` conceptually in this ADR.

Exact class naming is not normative.

It contains the latest spatial information required to resolve dynamic overlays, including at minimum:

```text
current metric/projector calibration authority
usable fiducial observations keyed by (group, id)
observation timestamps / freshness metadata
marker tabletop position
marker tabletop orientation where available
```

SpatialState should be published as an immutable snapshot, copy-on-write structure or equivalent thread-safe read model.

Projector rendering and overlay resolution must be able to obtain the latest complete state without blocking on camera capture.

The vision path may update at a different frequency from projector rendering.

Projector rendering uses the latest usable state; it does not wait synchronously for the next camera frame.

---

# 10. Fiducial tracking and freshness

Dynamic anchors need explicit behaviour when a fiducial disappears or becomes stale.

A marker observation must carry enough freshness information to prevent ancient positions from being treated as current indefinitely.

The first implementation should prefer a simple, configurable freshness policy rather than prediction or motion modelling.

For each dynamic overlay, the system must have deterministic fail/visibility semantics when an anchor cannot be resolved.

Acceptable initial behaviour is:

```text
fresh marker       → render overlay
marker briefly lost → optionally retain last observation for a small bounded grace period
marker stale/lost  → hide/suppress the dependent dynamic overlay
marker returns     → overlay resolves again automatically
```

The grace period, if used, must be bounded and observable/configurable.

Do not extrapolate marker movement or build a Kalman/prediction subsystem as part of ADR-0004.

Do not delete the overlay specification merely because its marker is temporarily unavailable.

---

# 11. Projector / vision coexistence

Projection must not make active vision targets unreadable.

Observed projector lines or shapes can contaminate a fiducial pattern enough to prevent reliable detection.

ADR-0004 therefore treats projector/vision coexistence as a correctness requirement, not merely a styling preference.

Two mechanisms are required.

## 11.1 Overlay intensity

The presentation layer supports a bounded global overlay intensity and per-overlay intensity/opacity where supported reliably by the renderer.

Conceptually:

```text
global_overlay_intensity = 0.0 .. 1.0
overlay.style.intensity  = 0.0 .. 1.0
```

Naming should avoid implying physical projector lamp control. `intensity` or `overlay intensity` is preferred over ambiguous hardware `brightness` unless hardware brightness is genuinely controlled.

Changing intensity must not alter physical geometry.

## 11.2 Protected fiducial regions

Active usable fiducials produce projector-space protected regions over which ordinary overlays are suppressed/clipped.

Conceptually:

```text
visible tracked fiducial
↓
marker footprint transformed into projector space
↓
expand by configured safety margin
↓
protected region
↓
normal overlays clipped/suppressed inside region
```

This protection is stronger than relying on low intensity alone.

Protection must use `(group, id)` identity and the current calibrated spatial observation.

The safety margin should be specified in physical units where metric calibration is available, so protection is not accidentally tiny on one part of a perspective surface and excessive on another.

Calibration patterns and other deliberate vision/calibration modes remain governed by their existing exclusive rendering semantics.

---

# 12. Protection policy

The default ordinary-operation policy is:

> Generic projector overlays must not draw across protected regions of currently tracked fiducials required by the spatial/vision subsystem.

The implementation may protect all configured/tracked fiducials or only an explicit active set, provided the rule is deterministic and does not create circular behaviour where a marker must first be lost before protection is enabled.

Protection must not cause an overlay's geometry model to change.

For example, a six-inch line remains a six-inch requested segment; only the pixels intersecting the protected marker region are omitted from presentation.

A line may therefore appear visually discontinuous while crossing a marker. That is preferable to destroying marker readability.

Any future opt-out should be explicit and should not become the default merely for visual neatness.

---

# 13. Camera capture, tracking and preview separation

Camera hardware capture is not the same concern as camera preview rendering.

The current runtime already owns camera capture workers; ADR-0004 must preserve that independent capture model rather than reintroducing synchronous camera reads in the display loop.

The runtime should conceptually allow independent frequencies such as:

```text
camera capture:      camera/native or configured rate
fiducial tracking:   configured rate, e.g. 15–30 Hz
projector rendering: 30–60 Hz
camera preview:      diagnostic rate, e.g. 5–15 Hz
HTTP/API handling:   independent of display cadence
```

Exact defaults require profiling on target hardware.

The important architectural rule is that camera preview must not set the cadence of projector rendering, API mutation or tracking.

---

# 14. Camera preview modes

Camera preview is diagnostically important because the user needs to see where cameras are aimed.

ADR-0004 therefore does **not** remove preview merely to save CPU.

Instead preview must be independently controllable.

At minimum support behaviour equivalent to:

```text
active / normal diagnostic preview
low-rate / idle preview
off
```

Turning preview off must not stop:

- camera capture;
- retained latest-frame publication;
- fiducial tracking;
- calibration validity;
- dynamic overlay anchor resolution.

Preview frame conversion/scaling/blitting may be rate-limited independently of projector presentation.

If profiling proves Pygame's main-thread requirements prevent safe independent windows/loops, preserve main-thread SDL ownership but schedule preview work at a lower cadence rather than forcing every projector frame to rebuild every preview.

---

# 15. Threading and no-GIL stance

ADR-0004 does not require Python no-GIL as its primary performance strategy.

Before considering interpreter-level concurrency changes, the implementation must profile and separate:

```text
camera device I/O
fiducial detection
frame colour/format conversion
preview scaling/blitting
overlay validation/materialisation
registry locking
HTTP/CLI overhead
projector texture upload/presentation
```

Camera capture already benefits from worker threads and native OpenCV operations may release the GIL where appropriate.

The preferred first architecture is independent workers/snapshots plus reduced coupling, not a global migration to an experimental/runtime-specific no-GIL deployment.

A no-GIL build may be investigated later only if profiling identifies Python lock contention as a material remaining bottleneck.

---

# 16. Locking and publication semantics

Realtime overlay updates must not hold a broad camera-management lock across expensive geometry generation, fiducial detection or rendering unless correctness demonstrably requires it.

The implementation plan must inspect the current shared-lock boundaries and minimise critical sections.

Preferred model:

```text
validate/build candidate state outside publication lock where safe
↓
short atomic registry/state swap
↓
renderer reads committed snapshot
```

Similarly, publishing a new SpatialState should not require projector rendering to wait for the vision worker to finish its next detection cycle.

Locking must preserve calibration invalidation semantics from earlier ADRs.

Do not trade correctness for lock-free complexity; short immutable snapshot swaps are preferred over a custom lock-free data structure.

---

# 17. Overlay lifecycle with dynamic anchors

The overlay registry stores the overlay **specification**, not only one historical projector materialisation.

A dynamic overlay remains independently addressable through the lifecycle from ADR-0003:

```text
create
inspect/list
show/hide
update/replace
remove
clear
```

Listing an overlay should make it possible to distinguish:

```text
static vs dynamic
anchor definitions
currently resolvable vs unresolved
current visibility
last resolution/freshness status where useful
```

Do not mutate the stored requested geometry every frame to the currently observed coordinates; that would erase the declarative anchor relationship.

---

# 18. API and CLI

All ADR-0004 capabilities must be available through the running service API and thin CLI where practical.

Conceptual API operations include:

```text
POST /overlays/batch
POST /overlays/arrow
PATCH/PUT /overlays/{id}
GET /overlays

GET/PUT overlay intensity controls
GET configured fiducial groups / tracking state where useful
```

Exact routes may follow existing API conventions.

Dynamic anchor JSON should be explicit rather than overloaded positional tuples.

Conceptual examples:

```json
{
  "kind": "arrow",
  "start": {
    "type": "surface",
    "x": 420,
    "y": 180,
    "unit": "mm"
  },
  "end": {
    "type": "fiducial",
    "group": "cards",
    "id": 17
  }
}
```

and:

```json
{
  "type": "fiducial",
  "group": "cards",
  "id": 17,
  "local_offset": {"x": 20, "y": 0, "unit": "mm"},
  "follow_rotation": true
}
```

The CLI delegates all state changes to the running service.

The CLI must not implement its own tracking loop or repeatedly calculate camera/projector transforms.

A long-lived external client may use HTTP directly when it needs high-rate updates; repeatedly spawning the CLI process is not the normative realtime transport.

---

# 19. Failure philosophy

Realtime behaviour must fail predictably rather than draw plausible but spatially wrong output.

Reject or suppress as appropriate when:

```text
unknown fiducial group
invalid fiducial ID
required calibration unavailable/stale
marker observation too old
non-finite coordinates
invalid marker-local offsets
zero-length arrow
batch exceeds configured limits
batch contains any invalid operation
resolved projector geometry is invalid
```

Distinguish:

```text
invalid overlay specification
```

from:

```text
valid dynamic overlay temporarily unresolved because its marker is absent
```

The former is an API error.

The latter is runtime state and should normally suppress rendering while retaining the overlay definition.

---

# 20. Non-goals

ADR-0004 does not include:

- detecting all rectangles or board cells from camera imagery;
- generic contour/object recognition;
- game rules;
- identifying miniatures without explicit fiducials;
- path planning;
- motion prediction;
- Kalman filtering;
- persistent overlay scenes across service restart;
- arbitrary scene graphs;
- hardware projector lamp brightness control unless already trivially available;
- a no-GIL Python migration;
- AI/LLM logic.

Those may be considered separately when concrete use cases require them.

---

# 21. Automated tests

At minimum test:

- compound `(group, id)` fiducial identity and collisions across groups;
- unknown-group and wrong-group failure semantics;
- group-specific marker dictionary/size configuration parsing where introduced;
- fixed surface anchor resolution;
- fiducial centre anchor resolution;
- marker-local offset and rotation resolution;
- line/arrow with surface → fiducial anchors;
- line/arrow with fiducial → fiducial anchors;
- two endpoints belonging to different groups;
- arrowhead geometry for arbitrary direction;
- degenerate zero-length arrow rejection/suppression;
- temporary missing-marker behaviour;
- stale observation expiry;
- overlay automatic recovery when a marker returns;
- batch all-or-nothing validation and commit;
- batch ordering/determinism;
- configured batch/geometry limits;
- global/per-overlay intensity bounds;
- protected fiducial-region generation;
- safety-margin transformation;
- clipping overlay primitives against protected fiducial regions without changing requested physical geometry;
- dynamic overlay specification remains unchanged across frame resolutions;
- renderer consumes a complete SpatialState snapshot rather than partially updated marker data;
- camera preview disable/low-rate mode does not stop camera capture or tracking;
- overlay update path remains functional with preview disabled;
- calibration invalidation correctly makes dependent anchors unusable;
- lock/publication tests demonstrating no partially committed batch state is exposed.

Synthetic geometry tests should use perspective-skewed transforms so projector-pixel shortcuts cannot accidentally pass.

---

# 22. Performance tests

Performance validation must include a running-service benchmark, not only unit-level microbenchmarks.

At minimum measure:

1. repeated single-overlay state updates through the service boundary;
2. atomic batches containing representative counts such as 10, 50 and 100 simple overlays, within configured limits;
3. dynamic anchor resolution while fiducial state changes;
4. projector render cadence while overlay mutations occur;
5. the same workload with camera preview enabled and disabled;
6. CPU utilisation split sufficiently to identify preview/frame-conversion/tracking bottlenecks.

Acceptance requirement:

> On the target development Mac, the running MultiVision service sustains at least 30 overlay state mutations per second during normal camera capture, tracking and projector operation without multi-frame projector stalls caused by the mutation path.

If the exact target workload makes 30 independent HTTP round trips unnecessarily pessimistic, a batch/replacement workload may additionally demonstrate much higher effective object-update throughput, but this does not remove the 30 state-mutation/s service target.

Report CLI cold-start separately.

---

# 23. Manual hardware acceptance

Manual acceptance requires the physical projector/camera setup.

Minimum demonstrations:

1. create and move/update a visible overlay interactively at a rate that is visually smooth and measure at least 30 service mutations/s;
2. submit a batch of many overlays and confirm they appear as one coherent state transition rather than progressively over many frames;
3. render an arrow between two fixed physical points;
4. attach an overlay to a fiducial and physically move/rotate the tagged object; confirm the overlay follows;
5. draw a line/arrow between two fiducials and move each independently;
6. draw an arrow from a fixed table destination to a tagged card and confirm it continuously indicates the current card position;
7. demonstrate two identical numeric IDs in different configured groups are addressed independently;
8. project geometry that would otherwise cross a tracked marker and confirm the protected region prevents projector content from corrupting the marker sufficiently to break normal tracking;
9. change overlay intensity and confirm geometry remains unchanged;
10. run with camera preview enabled, reduced-rate and disabled; confirm camera capture/tracking and dynamic overlays continue to function in all three modes;
11. compare CPU utilisation with preview enabled vs disabled/reduced to establish whether preview is a material bottleneck.

A coding agent must not claim physical marker-readability or projector/camera coexistence passed from synthetic tests alone.

---

# 24. Mythra / adversarial constraints

The implementation plan derived from this ADR must explicitly defend against these failure modes:

1. **Optimising CLI cold start instead of the service.** The realtime contract concerns the long-running runtime; CLI startup is measured separately.
2. **Fake batching.** Looping over N ordinary HTTP/service mutations is not an atomic batch and still exposes intermediate states.
3. **Early materialisation.** Storing only projector coordinates loses the anchor relationship and forces clients to resend moving geometry.
4. **Renderer-owned vision.** Projector rendering must never wait for or perform fiducial detection.
5. **ID-only fiducials.** Marker identity is `(group, id)` everywhere; silent cross-group matches are correctness bugs.
6. **Projection destroys tracking.** Dimming alone is insufficient; protected fiducial regions are a correctness mechanism.
7. **Circular protection activation.** Protection must not require losing a marker first before the system decides it should protect that marker.
8. **Stale-marker ghosts.** Last-known marker positions must not remain authoritative indefinitely.
9. **Preview removal masquerading as optimisation.** The user still needs preview for camera aiming. Decouple/rate-limit it rather than deleting it.
10. **Preview cadence controls everything.** Projector/API/tracking cadence must remain independent of preview refresh.
11. **Premature no-GIL migration.** Profile real bottlenecks before changing interpreter/runtime assumptions.
12. **Broad lock contention.** Expensive materialisation/detection under a shared camera-management lock can defeat the realtime target even with multiple threads.
13. **Dynamic overlay deletion on temporary loss.** Missing markers suppress resolution; they do not erase caller intent.
14. **Visual clipping changes geometry.** Fiducial protection may hide part of a line, but it must not shorten or reroute the requested physical line.
15. **Overengineering tracking.** No prediction/filtering framework is needed merely to follow a marker at tabletop speed.
16. **ADR-0005 creep.** Rectangle/board-feature detection is not part of ADR-0004 even if convenient vision utilities are nearby.

These constraints must appear in planning acceptance criteria and tests, not only implementation comments.

---

# 25. Consequences

After ADR-0004, MultiVision's projector output becomes reactive rather than merely static.

A caller can define relationships such as:

```text
fixed tabletop destination → moving tagged card

tagged unit A → tagged unit B

overlay local to tagged card orientation
```

and leave the running service to maintain the projected geometry from current spatial observations.

Batch mutations make larger scenes practical without partial visual updates, while the 30 Hz service target establishes an explicit interactive-performance contract.

Camera preview remains available for setup but no longer needs to dictate the cadence of projector rendering or vision.

Most importantly, projection and vision become cooperative subsystems: overlays can be dimmed and are prevented from drawing across protected fiducial regions, so the projector does not casually destroy the tracking signal required to drive dynamic overlays.

This ADR intentionally stops at realtime overlays, anchors and vision-safe presentation. Board/rectangle detection belongs to a later ADR.