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
current metric homography, derives a finite surface-mm footprint, and uses the
local projector X/Y directions at the projector centre as the grid anchor and
orientation. It then builds an ordinary physical `GridRequest` and relies on
the existing projector-native clipping path. It preserves the current grid
spacing in physical millimetres while covering the complete projector output.
The mode intentionally assumes
that the calibrated A4 sheet and the full projected surface share one flat
plane, as requested by the operator.

## Text labels

Rectangles accept an optional `label`, `label_angle_deg` and `label_scale`; the
label is anchored at the rectangle centre. Independent floating labels use
`POST /overlays/text` or `multivision overlay text --spec-json`, with a
`position`, `text`, `angle_deg` and `scale`. Positions use the same
`projector_px`, `camera_px` and `surface_mm` point references as other overlays
and are resolved through the existing transform chain before rendering.

Labels use the existing default Pygame font. Rotation and scale are applied in
projector-native pixels at render time; font selection is intentionally not part
of this capability. Scale is bounded to `0.1`–`32.0` to keep rasterisation
bounded.

See the [shared validation contract](../plans/plan5-validation-contract.md).
