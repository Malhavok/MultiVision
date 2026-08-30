# Plan5 implementation boundary

- Point references are `camera_px`, `projector_px` or `surface_mm`.
- Geometry and measurement spaces are only `projector_px` or `surface_mm`.
- Camera pixels may locate geometry, but never define shape size or ruler distance.
- Projector-native geometry remains the final rendering authority.
- No game, AI or second scale/transform authority belongs in this layer.

See the [shared validation contract](../plans/plan5-validation-contract.md).
