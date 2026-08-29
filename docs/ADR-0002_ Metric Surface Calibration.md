# ADR-0002: Metric Surface Calibration

**Status:** Proposed — ready for planning  
**System:** MultiVision  
**Scope:** Physical surface metric calibration  
**Decision type:** Geometry / calibration / validation contract

---

# 1. Purpose

MultiVision already knows how to map between camera-native coordinates and projector-native coordinates. That answers:

> Which projector pixel corresponds to this physical point seen by a camera?

The next requirement is to know physical distance on the tabletop. MultiVision must be able to answer:

> How many millimetres separate these two physical points?

and:

> Which projector pixels represent a physical length of exactly 100 mm?

This ADR introduces a second calibration layer: **metric surface calibration**.

The two calibration layers are intentionally separate:

```text
camera calibration
camera-native ↔ projector-native

metric surface calibration
projector-native ↔ surface-mm
```

The first establishes spatial correspondence. The second establishes physical scale and orientation on the same planar tabletop.

The user-facing acceptance test is deliberately physical:

1. generate and print the MultiVision A4 metric calibration target at actual size;
2. verify its printed reference dimensions with a physical ruler;
3. place it flat on the tabletop;
4. calibrate the metric surface using a camera that already has valid camera/projector calibration;
5. ask MultiVision to project a ruler of a known length between chosen points;
6. place a physical ruler against the projected ruler;
7. compare the requested and measured physical length.

MultiVision must report calibration error in physical units rather than only pixels.

---

# 2. Core decision

For a flat tabletop, metric calibration is a property of the **projected physical surface**, not of an individual camera.

Therefore:

> There is one projector-native ↔ surface-mm transform for the active physical tabletop geometry.

A calibrated camera is a sensor used to observe the printed target and derive that transform. The resulting metric calibration must not become owned by that camera.

After successful calibration, any camera with a valid camera/projector transform can participate in metric operations through the shared surface transform.

Do not store independent `mm_per_pixel` values per camera.

Do not assume one scalar `mm_per_pixel` for the projector. Perspective means physical scale may vary across projector-native coordinates.

Use a planar projective transform between projector-native coordinates and a metric 2D surface coordinate system.

Conceptually:

```text
H_ps:
projector-native → surface-mm

H_sp:
surface-mm → projector-native
```

Both directions must be stored explicitly.

---

# 3. Assumptions and boundaries

ADR-0002 assumes:

- the useful tabletop is approximately planar;
- the printed calibration target lies flat on that same plane;
- the projector geometry does not change after metric calibration;
- the camera used for calibration has a currently valid camera/projector calibration;
- the printed target has been verified to be physically the expected size.

ADR-0002 does **not** introduce:

- 3D reconstruction;
- depth measurement;
- SLAM;
- automatic tabletop plane discovery;
- game concepts;
- miniature or token tracking;
- camera intrinsic calibration unless existing measured error demonstrates it is necessary;
- a per-camera physical scale.

If the tabletop is materially non-planar, the homography model is no longer sufficient. Do not attempt to hide that failure with local scale corrections.

---

# 4. Surface metric coordinate space

Introduce an explicit metric surface coordinate space:

```text
surface_mm_x
surface_mm_y
```

Internal physical units are **millimetres**.

User-facing APIs and CLI may accept:

- `mm`;
- `cm`;
- `in`.

All values are normalised to millimetres before geometry calculations.

The printed calibration target defines a deterministic metric coordinate frame. Its fiducial IDs and known physical locations determine origin, axis direction and orientation; the user does not need to align the sheet with projector axes.

It is valid for the projected surface corners to map to negative or otherwise non-intuitive metric coordinates relative to the target origin. The metric frame is mathematical; callers should not depend on the target being placed at a specific physical location.

---

# 5. Printable A4 calibration target

MultiVision must generate a printable A4 metric calibration target.

The target must be deterministic and reproducible from code. It must not be a hand-maintained binary asset whose physical dimensions are implicit.

The generated target must contain:

- multiple uniquely identified OpenCV-supported fiducials;
- fiducials distributed across as much of the printable A4 area as practical;
- known metric coordinates for every fiducial corner used by calibration;
- an unambiguous orientation;
- human-readable instructions to print at **100% / Actual size**;
- at least one labelled physical reference segment, preferably 100 mm or longer, for verification with a ruler;
- a warning that `Fit to page`, printer scaling or browser scaling invalidates the nominal geometry;
- target format/version metadata sufficient to reject incompatible target definitions.

The implementation may choose the exact printable representation, but it must preserve known physical dimensions when printed correctly. A vector format with explicit physical dimensions is preferred. If a raster representation is emitted, its intended physical size/DPI must be explicit and tested.

The generation capability must be available from the CLI, for example conceptually:

```text
multivision metric target generate --output metric-target.svg
```

Exact command spelling is not architectural.

---

# 6. Printed-size verification

A4 paper dimensions alone are not trusted as evidence that the printer produced a 1:1 target.

Before calibration, the user must be able to verify at least one known reference segment with a physical ruler.

The workflow should explicitly instruct:

```text
Expected reference: 100.0 mm
Measure before continuing.
```

An incorrectly scaled print must not silently produce a calibration that appears precise while being systematically wrong.

For MVP implementation, user confirmation of correct physical printing is sufficient. Automatic inference of printer scale is not required.

A later implementation may optionally allow the user to enter an actually measured reference length and compensate for uniform print scaling, but this is not required by ADR-0002 and must not replace correct printing as the default path.

---

# 7. Calibration correspondences

Metric calibration uses the already-calibrated camera as an observation device.

For every detected printed-target fiducial corner, the system knows:

```text
known surface-mm coordinate
↔
detected camera-native coordinate
```

The existing camera → projector transform provides:

```text
detected camera-native coordinate
→
projector-native coordinate
```

Therefore calibration obtains correspondences:

```text
projector-native coordinate
↔
known surface-mm coordinate
```

Use all valid fiducial corners rather than reducing each marker to its centre.

Use multiple markers spread over the A4 sheet. Four non-collinear points are mathematically sufficient for a homography but are not sufficient evidence of trustworthy physical calibration.

---

# 8. Calibration algorithm

The metric calibration process is:

1. require a currently valid camera/projector calibration for the selected observation camera;
2. instruct the user to place the verified A4 target flat on the physical tabletop and not move it;
3. capture one or more stable frames;
4. detect target fiducials and their native camera-space corners;
5. associate marker IDs/corners with known target surface-mm coordinates;
6. map detected camera coordinates into projector-native coordinates using the existing camera calibration;
7. estimate `projector-native → surface-mm` homography using all valid correspondences;
8. use RANSAC or equivalent robust fitting to reject outliers;
9. calculate and store the inverse `surface-mm → projector-native` transform;
10. calculate physical error metrics in millimetres;
11. validate geometric coverage and transform quality;
12. make metric operations available only if validation succeeds.

Metric calibration must not modify the camera/projector calibration that was used to observe the target.

---

# 9. Calibration quality and error

Metric calibration exists partly so MultiVision can express accuracy in useful physical terms.

At minimum record:

- number of unique target fiducials detected;
- number of correspondence corners;
- RANSAC inlier count;
- inlier ratio;
- mean or median residual error in mm;
- maximum residual error in mm;
- spatial coverage across the printed target;
- target format/version;
- projector resolution;
- observation camera slot and calibration generation used to derive the result, for diagnostics only;
- timestamp/session metadata useful for diagnostics.

Residual error from points used in the fit is **not** sufficient evidence of whole-surface physical accuracy. The A4 target covers only part of a potentially larger projected surface and extrapolation error can be larger elsewhere.

Therefore distinguish:

```text
fit_error_mm
```

from:

```text
physical_validation_error_mm
```

The latter comes from independent physical ruler checks and must not be fabricated from the calibration fit.

Do not claim sub-millimetre or whole-surface accuracy merely because homography residuals on the target are small.

---

# 10. Projected ruler validation

MultiVision must provide a calibration/validation ruler capability.

The user supplies two physical-surface points or a start point, direction and requested physical length. MultiVision projects a ruler between the corresponding projector-native points and labels its expected length.

For example conceptually:

```text
multivision metric ruler \
  --from-mm 100,100 \
  --to-mm 300,100
```

or an equivalent command/API operation.

The projected ruler should include:

- a clear start marker;
- a clear end marker;
- a line between them;
- tick marks where practical;
- an explicit length label, e.g. `200 mm / 20.0 cm`.

The purpose is not decorative rendering. It is to allow a physical ruler to be placed directly on the tabletop and verify that projected metric geometry matches reality.

Validation should be performed at several locations and orientations, not only near the calibration target or projector centre. Suggested checks include:

- horizontal-ish segment near one side;
- horizontal-ish segment near the opposite side;
- vertical-ish segment;
- diagonal segment;
- at least one segment near the edge of the intended usable area.

The system should make it possible to record requested length, physically observed length and absolute error.

---

# 11. Metric calibration state

Metric operations require an explicit state, for example:

```text
UNCALIBRATED
CALIBRATED
STALE
```

A metric transform must be considered invalid if known projector-native geometry changes, including at minimum:

- projector/display resolution changes;
- selected projector/display output changes;
- the user explicitly resets or recalibrates the metric surface.

Physical movement of the projector relative to the tabletop also invalidates metric calibration even if software cannot automatically detect that movement.

Recalibrating or moving an observation camera does **not** inherently invalidate a previously established projector ↔ surface-mm transform, because metric calibration belongs to the surface/projector geometry rather than the camera. However, a camera must itself have valid camera/projector calibration before it can be used for a new metric calibration or metric observation.

Do not invent automatic persistence trust. If metric calibration is persisted, it must be treated as unverified after restart unless a physical verification mechanism demonstrates that projector/table geometry is unchanged.

Session-local calibration is acceptable for the first implementation.

---

# 12. API and CLI capabilities

Expose metric calibration as stable service capabilities rather than UI-only behaviour.

Conceptual operations:

```text
POST /metric/calibration
GET  /metric/calibration/status
DELETE /metric/calibration

POST /metric/ruler
DELETE /metric/ruler
```

Conceptual CLI:

```text
multivision metric target generate
multivision metric calibrate --camera <camera>
multivision metric status
multivision metric ruler ...
multivision metric clear
```

Exact URLs and command spellings may follow existing repository conventions.

The CLI remains a thin client. It must not independently open cameras, detect fiducials, calculate homographies or render the projector.

---

# 13. Shared geometry authority

The metric transform must become shared geometry authority for later physical overlays.

Later features must use:

```text
surface-mm
→ shared metric transform
→ projector-native
→ renderer
```

They must not calculate their own `pixels_per_cm`, camera-specific scale, display-size approximation or ad-hoc conversion.

In particular, later grids, movement radii, line-of-sight helpers and piece positions must consume this shared metric coordinate system.

---

# 14. Failure philosophy

Metric operations fail closed.

Reject or clearly fail when:

```text
no valid observation-camera calibration
printed target cannot be identified reliably
insufficient target coverage
metric homography is degenerate
metric fit error exceeds configured threshold
metric calibration is unavailable or stale
requested metric geometry maps outside projector bounds
projector configuration no longer matches calibration
```

Never substitute an approximate DPI or nominal projector size when metric calibration is unavailable.

A caller asking for `6 in` of physical distance must either receive geometry derived from a valid metric transform or an explicit failure.

---

# 15. Automated tests

Automated tests must cover at minimum:

- deterministic generation of target geometry;
- known physical dimensions of generated target coordinates;
- marker IDs and orientation;
- synthetic projector ↔ surface-mm homography recovery;
- composition of camera → projector and projector → surface mappings;
- inversion / metric round trips;
- perspective transforms where scale changes across projector pixels;
- noisy correspondences;
- outlier rejection;
- degenerate target geometry;
- invalid/missing camera calibration;
- projector-resolution invalidation;
- unit conversion between mm, cm and inches;
- ruler length calculations in metric space;
- rejection of geometry mapping outside valid projector bounds.

Automated tests may prove mathematical correctness. They cannot prove that a printer produced the target at 1:1 scale or that a physical projected ruler is the requested length.

---

# 16. Manual hardware acceptance

Hardware acceptance requires the actual projector, tabletop, at least one calibrated camera, printed target and physical ruler.

Minimum manual evidence:

1. generate the target from MultiVision;
2. print at actual size;
3. verify the printed reference segment with a ruler;
4. calibrate metric surface successfully;
5. record fit metrics;
6. project multiple known-length rulers at different positions/orientations;
7. measure them physically;
8. record requested length, measured length and absolute error;
9. verify that a deliberately invalidated metric calibration refuses metric operations.

A coding agent must not claim physical metric accuracy from mocks, screenshots or synthetic tests.

---

# 17. Consequences

After ADR-0002, MultiVision gains a physical coordinate system independent from camera resolution and projector pixel density.

The system can express geometry in real units:

```text
25.4 mm
100 mm
6 in
```

and render it onto the table using a single shared transform.

This becomes the foundation for ADR-0003 physical geometry overlays and later piece interaction.

The cost is an additional explicit calibration step whenever projector/table geometry changes. That cost is preferred over pretending that pixel distance is a physical measurement.

---

# 18. Explicit non-goals for the next plan

The implementation plan derived from ADR-0002 must stop after metric calibration and ruler validation work.

Do not include:

- gameplay grids except what is minimally useful to validate metric geometry;
- movement ranges;
- line of sight;
- miniature registration;
- occupancy detection;
- background subtraction;
- AI behaviour;
- multiplayer/networking;
- game-specific rules.

Those belong to later ADRs.
