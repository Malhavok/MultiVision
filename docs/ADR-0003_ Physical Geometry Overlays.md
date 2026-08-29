# ADR-0003: Physical Geometry Overlays

**Status:** Proposed — ready for planning  
**System:** MultiVision  
**Scope:** Metric tabletop overlays  
**Decision type:** Geometry / renderer / API contract

---

# 1. Purpose

ADR-0002 gives MultiVision a shared physical coordinate system expressed in millimetres.

ADR-0003 uses that metric surface to render useful physical geometry onto the tabletop.

The goal is not to understand a game. The goal is to provide reusable spatial primitives such as:

- square grids;
- distance rulers;
- circles / movement-radius helpers;
- straight lines / line-of-sight helpers;
- simple highlighted areas.

These primitives must be callable from the local UI, CLI and HTTP API through the same underlying implementation.

The user-facing examples include:

- project a 1-inch square grid under 32 mm miniatures;
- show a 6-inch movement radius around a selected physical position;
- draw a line between two physical points to inspect line of sight;
- display a temporary physical ruler;
- leave a grid visible while other overlays are added and removed.

---

# 2. Core decision

All physical overlays are defined in **surface-mm space** and transformed through the shared metric calibration from ADR-0002.

Conceptually:

```text
caller
↓
physical geometry in surface-mm
↓
shared surface-mm → projector-native transform
↓
projector renderer
```

No overlay type may calculate its own projector scale, `pixels_per_cm`, camera-specific scale or approximate DPI.

If metric calibration is unavailable or stale, metric overlays fail explicitly.

---

# 3. Architecture boundary

ADR-0003 is a generic spatial rendering layer.

It must not introduce game semantics such as:

```text
Piece
Unit
Player
Army
Turn
MovementStat
AttackRange
Weapon
LineOfSightRule
Terrain
```

For example, MultiVision may understand:

```text
circle(center=(420 mm, 310 mm), radius=152.4 mm)
```

but it must not understand:

```text
SpaceMarine.move_range = 6 inches
```

A future game/client/AI decides that a unit has a six-inch movement allowance and asks MultiVision to render the corresponding geometry.

The same rule applies to line of sight: ADR-0003 draws a physical segment between supplied points. It does not decide whether terrain blocks sight, whether a shot is legal or which game rule applies.

---

# 4. Coordinate and unit model

Internal geometry uses millimetres.

External callers may specify physical dimensions using:

- millimetres;
- centimetres;
- inches.

Values are normalised to millimetres before geometry calculations.

Positions are expressed in the metric surface coordinate system established by ADR-0002.

Examples:

```text
(350 mm, 420 mm)

1 in grid spacing = 25.4 mm

6 in radius = 152.4 mm
```

Unit conversion is deterministic geometry logic and must be independent of rendering.

---

# 5. Overlay primitives

The first implementation must remain deliberately small.

Required primitives are:

## 5.1 Grid

A square grid with configurable physical spacing.

At minimum:

```text
spacing
origin/phase
extent or clipping region
style
```

Typical use:

```text
spacing = 25.4 mm
```

for a one-inch tabletop grid.

Grid alignment must be stable in surface-mm coordinates. Re-rendering the same grid must not shift because of projector pixel rounding.

The grid is square in physical space even when the projector-native representation is perspective-distorted.

---

## 5.2 Circle / radius

A physical circle defined by:

```text
center
radius
style
```

Typical use:

```text
center = current miniature position
radius = 152.4 mm
```

for a six-inch helper.

The circle is defined as a true circle in metric surface space. Its projector representation may be an ellipse or another perspective-transformed curve in projector-native coordinates.

Do not draw a naïve projector-pixel circle and label it as a physical radius.

Implementation may approximate the transformed curve with sufficiently dense line segments. Approximation tolerance must be expressed and tested.

---

## 5.3 Line segment

A straight physical segment between two surface points.

At minimum:

```text
start
end
style
optional label
```

Typical uses:

- line-of-sight helper;
- measured distance indication;
- direction indicator.

The geometric distance between start and end should be available to callers in physical units.

---

## 5.4 Ruler

ADR-0002 requires a minimal projected ruler for calibration validation.

ADR-0003 generalises that capability into an ordinary overlay primitive.

A ruler may include:

- start/end markers;
- line;
- major/minor physical tick marks;
- distance label;
- selectable display units.

The implementation should reuse one physical-distance calculation path rather than maintaining separate calibration-ruler and general-ruler mathematics.

ADR-0002 validation requirements remain authoritative for calibration acceptance.

---

## 5.5 Highlighted region

Provide a simple polygonal or rectangular physical-region overlay if the existing Plan3 region/available-area rendering can be cleanly generalised without creating duplicate geometry paths.

This is useful for later movement destinations and setup instructions.

It is secondary to grid, circle, line and ruler. Do not delay the core ADR-0003 implementation merely to build a general-purpose shape framework.

---

# 6. Rendering physical curves correctly

The metric surface → projector transform is projective.

Straight lines remain straight under a homography, but circles and regular grids in metric space must not be implemented by assuming constant projector pixel scale.

The renderer must preserve physical geometry first and derive projector geometry second.

For curves such as circles:

1. generate sample points in surface-mm space;
2. transform those points through the shared metric transform;
3. render the resulting projector-space polyline/polygon;
4. use a bounded approximation error/tessellation strategy.

The first implementation does not need a symbolic conic renderer.

A simple deterministic sampling strategy is preferred if it produces sufficiently accurate physical results.

---

# 7. Clipping and valid physical area

Overlays may extend beyond the visible projector or beyond areas useful to cameras.

ADR-0003 distinguishes these concepts:

- **projectable physical area**: surface region reached by the calibrated projector;
- **camera available area**: region a particular camera can observe reliably;
- **overlay geometry**: requested shape, which may be larger than either.

Rendering should clip safely to projector output bounds.

Do not silently reinterpret the requested physical size merely to make it fit.

For example, a requested 6-inch circle that extends off-screen remains a 6-inch circle with the invisible portion clipped; it must not be shrunk.

Camera visibility is not required merely to render an overlay. A caller may intentionally illuminate a projectable region not currently useful to one specific camera.

---

# 8. Overlay lifecycle

Multiple useful overlays need to coexist.

The existing single-point overlay behaviour must not force ADR-0003 into a global singleton overlay model.

The service should support independently addressable overlay instances with stable session-local IDs.

Conceptually:

```text
overlay-id
kind
geometry
style
visible/state
```

Required lifecycle capabilities:

```text
create/show
inspect/list
replace/update where useful
remove one
clear a group or all
```

Do not introduce an event bus or elaborate scene graph.

A small in-memory overlay registry owned by the running service is sufficient.

Persistent overlays across process restart are not required.

---

# 9. Overlay ordering

Rendering order must be deterministic.

Calibration patterns retain exclusive priority while calibration is active and must not be contaminated by normal overlays.

During ordinary operation, an explicit simple ordering is sufficient. For example:

```text
background / grid
regions / ranges
lines / rulers
points / emphasis markers
labels
```

The exact layer names may follow the existing renderer.

The important requirements are:

- a grid must not obscure a highlighted movement/range overlay;
- labels remain readable where practical;
- the existing pointing overlay remains usable;
- adding a new overlay does not unpredictably reorder unrelated existing overlays.

Do not implement a fully general z-index system unless the concrete renderer requires it.

---

# 10. Style model

ADR-0003 needs enough style control for overlays to remain distinguishable, but styling must remain bounded.

Useful style attributes include:

```text
colour
line width
fill / outline mode
alpha where renderer support is reliable
label visibility/text
```

Do not build a CSS-like styling system.

Default styles should make common overlays immediately usable without requiring every caller to specify presentation details.

Geometry and style must remain separate enough that changing colour or line width cannot change physical dimensions.

---

# 11. Grid behaviour

Grid generation deserves explicit rules because small ambiguities become obvious on a physical table.

A grid must have:

- exact physical spacing in surface-mm space;
- deterministic phase/origin;
- a bounded extent;
- predictable clipping;
- stable line placement across frames.

The caller may choose an origin/anchor so that a grid can be aligned to a physical board or scenario.

If no origin is supplied, use a deterministic surface-space default rather than aligning to projector pixels.

One-inch grid means exactly:

```text
25.4 mm × 25.4 mm
```

not a rounded centimetre approximation.

A future hex grid is plausible but is explicitly outside the minimum ADR-0003 implementation unless it falls out nearly for free from the chosen geometry model.

---

# 12. Range behaviour

ADR-0003 exposes physical radii/areas, not movement rules.

The caller provides the centre and requested radius.

Example conceptual command:

```text
multivision overlay circle \
  --center-mm 420,310 \
  --radius 6in
```

A later piece subsystem may use this primitive when a physical miniature is picked up.

ADR-0003 does not decide:

- whether the base edge or base centre is the measurement origin;
- whether terrain modifies movement;
- whether movement follows a path rather than radial distance;
- whether another miniature blocks movement.

Those are client/game decisions.

---

# 13. Line-of-sight behaviour

A line helper is intentionally literal.

Given two surface-mm points, MultiVision draws the segment and may label its physical length.

A future caller can use it as a line-of-sight indicator.

ADR-0003 does not inspect the camera image for obstructions and does not determine legal visibility.

This avoids coupling generic physical rendering to miniature recognition or game rules.

---

# 14. API and CLI

All new overlay capabilities must be available through the service API and thin CLI.

Conceptual API operations:

```text
POST   /overlays/grid
POST   /overlays/circle
POST   /overlays/line
POST   /overlays/ruler
GET    /overlays
DELETE /overlays/{id}
DELETE /overlays
```

Exact routes may follow existing conventions.

Conceptual CLI:

```text
multivision overlay grid --spacing 1in

multivision overlay circle \
  --center-mm 420,310 \
  --radius 6in

multivision overlay line \
  --from-mm 100,100 \
  --to-mm 500,350

multivision overlay ruler \
  --from-mm 100,100 \
  --to-mm 300,100 \
  --unit cm

multivision overlays list
multivision overlay remove <id>
multivision overlays clear
```

CLI commands must delegate geometry/state changes to the running service. They must not initialise Pygame or calculate an independent metric transform.

This API/CLI boundary is intentionally suitable for later AI clients.

---

# 15. AI/client boundary

ADR-0003 should make later AI control easy without introducing AI itself.

An AI client should eventually be able to request:

```text
highlight this physical area
show this 6-inch radius
show a line from A to B
show a destination marker here
```

through the same API used by any other client.

Do not create a privileged AI-only geometry path.

Do not implement LLM calls, prompting, agent state or game reasoning in ADR-0003.

---

# 16. Interaction with current Plan3 camera areas

Current MultiVision can derive and display per-camera available-area polygons as diagnostics.

ADR-0003 must not reinterpret these camera diagnostic areas as the global physical surface or as gameplay zones.

They are useful existing geometry and renderer infrastructure, but they have different semantics:

```text
camera available area
= where this camera has useful calibrated support

physical overlay
= geometry requested on the metric projected surface
```

Reuse pure polygon/transform/rendering code where appropriate. Do not merge the domain concepts merely because both are drawn as shapes.

---

# 17. Failure philosophy

Physical geometry should fail explicitly rather than render plausible but dimensionally wrong output.

Reject operations when:

```text
metric calibration unavailable
metric calibration stale
non-finite coordinates
invalid or non-positive physical sizes
unsupported units
requested geometry cannot be transformed safely
resulting projector coordinates are invalid
```

Geometry extending partially beyond the projector is normally clipped rather than rejected, provided the visible part can be rendered without changing requested physical dimensions.

A caller must never receive an apparent `6 in` overlay generated from an uncalibrated pixel approximation.

---

# 18. Automated tests

At minimum test:

- unit conversion (`1 in == 25.4 mm` exactly within numeric representation);
- deterministic grid line generation;
- grid physical spacing before projection;
- stable grid phase/origin;
- perspective transformation of grid geometry;
- physical circle sampling before projection;
- bounded circle tessellation error;
- line length calculations;
- ruler tick/label geometry where applicable;
- clipping without resizing physical geometry;
- invalid/non-positive dimensions;
- unavailable/stale metric calibration;
- overlay registry create/list/remove/clear semantics;
- deterministic render ordering;
- API/CLI delegation to the same service geometry path.

Synthetic tests should include a deliberately skewed projector ↔ surface transform so an implementation that incorrectly assumes constant pixels-per-mm cannot accidentally pass.

---

# 19. Manual hardware acceptance

Hardware acceptance requires a metric-calibrated physical table.

Minimum manual demonstrations:

1. display a 1-inch square grid;
2. physically verify spacing at several positions using a ruler;
3. display at least one known-radius circle, such as 3 or 6 inches;
4. physically verify radius/diameter at multiple directions;
5. display a line/ruler between selected surface points and verify its physical length;
6. show multiple overlay types simultaneously and verify deterministic visibility/order;
7. remove one overlay without disturbing unrelated overlays;
8. invalidate metric calibration and confirm metric overlays refuse to render as valid physical geometry.

For the grid, measurements should include separated cells away from the calibration target so cumulative/projective error is visible.

A coding agent must not claim these physical checks passed from automated tests alone.

---

# 20. Mythra / adversarial constraints

The implementation plan derived from this ADR must explicitly defend against the following likely failure modes:

1. **Constant pixel scale assumption.** A projector pixel is not a fixed physical length across a perspective-projected surface.
2. **Game-engine creep.** Movement, LOS and ranges are geometry requests; MultiVision must not own game rules.
3. **Over-generalised overlay framework.** Do not build a scene graph, plugin renderer or generic vector graphics engine before concrete primitives need it.
4. **False physical accuracy.** A mathematically correct transform is not evidence that physical output is accurate; retain hardware/ruler acceptance.
5. **Camera-area confusion.** Per-camera valid regions are sensor diagnostics, not the metric world model.
6. **Rounding drift.** Projector-pixel rounding must not alter metric grid phase or physical dimensions across frames.
7. **Curve distortion.** A projector-native circle is generally not a physical circle; define curves in metric space first.
8. **Singleton overlay state.** A grid, range and line need to coexist; avoid replacing all overlays whenever one changes.

These constraints should be reflected in tests and plan acceptance criteria, not merely mentioned in code comments.

---

# 21. Consequences

After ADR-0003, MultiVision becomes a useful physical geometry API even before miniature tracking exists.

A human, script or future AI can ask the table to display exact physical constructs such as:

```text
1-inch grid
6-inch radius
350-mm line
20-cm ruler
```

without knowing projector pixels or camera geometry.

This creates a clean foundation for the next layer: physical pieces and occupancy transitions.

A later piece subsystem can remain small because it can reuse ADR-0003 primitives for selection, movement ranges, valid destinations, invalid-placement feedback and AI-directed movement prompts.

---

# 22. Explicit non-goals for the next plan

The implementation plan derived from ADR-0003 must stop at generic metric overlays.

Do not include:

- piece registration;
- background baking/subtraction;
- occupancy detection;
- pickup/drop detection;
- miniature identity recognition;
- movement legality rules;
- line-of-sight obstruction rules;
- AI/LLM integration;
- multiplayer/networking;
- persistent game state.

Those are subsequent concerns built on top of this geometry layer.
