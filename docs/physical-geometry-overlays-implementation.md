# Plan5 implementation boundary

- Point references are `camera_px`, `projector_px` or `surface_mm`.
- Geometry and measurement spaces are only `projector_px` or `surface_mm`.
- Camera pixels may locate geometry, but never define shape size or ruler distance.
- Projector-native geometry remains the final rendering authority.
- No game, AI or second scale/transform authority belongs in this layer.

## Projector-footprint grids

Normal `GridRequest` values continue to require an explicit finite origin and
extent. For a flat projected surface, the service also supports a named
projector-footprint grid capability:

```text
POST /overlays/grid/projector-footprint
multivision overlay grid --fill-projector --spacing 35mm
```

This capability inverse-projects the four projector-output corners through the
current metric homography, derives a finite surface-mm bounding box, builds an
ordinary physical `GridRequest`, and relies on the existing projector-native
clipping path. It preserves the current grid spacing in physical millimetres
while covering the complete projector output. The mode intentionally assumes
that the calibrated A4 sheet and the full projected surface share one flat
plane, as requested by the operator.

See the [shared validation contract](../plans/plan5-validation-contract.md).
