# Calibration JSON contract

`calibration.json` is a portable snapshot of the two session calibration authorities.
It is deliberately not loaded automatically – physical geometry is trusted only
when the Driver confirms that the cameras, projector output and surface have not
moved.

The following is a schematic of the stable envelope and record fields. The
placeholder numeric values are illustrative; real homographies and metrics must
satisfy the service validators.

```json
{
  "format": "multivision-calibration",
  "version": 1,
  "camera": {
    "calibrations": {
      "camera-1": {
        "camera_id": "camera-1",
        "camera_resolution": {"width": 1920, "height": 1080},
        "projector_resolution": {"width": 1280, "height": 720},
        "projector_output_descriptor": {
          "projector_resolution": {"width": 1280, "height": 720},
          "output_identity": "default"
        },
        "version": 1,
        "projector_to_camera": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "camera_to_projector": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "metrics": {},
        "timestamp": 0,
        "valid_region": [],
        "calibration_scope": "global"
      }
    }
  },
  "metric": {
    "calibration": {
      "state": "CALIBRATED",
      "projector_to_surface": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
      "surface_to_projector": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
      "projector_output_descriptor": {},
      "target_format": "multivision-metric-target",
      "target_version": 2,
      "marker_family": "DICT_APRILTAG_36h11",
      "metrics": {},
      "validation_records": []
    }
  }
}
```

The `camera` and `metric` values are the corresponding public status response
objects. The service validates their matrices, versions, resolutions, output
descriptor and metric source-camera provenance before accepting a load.

Export and load through the thin client:

```sh
bin/multivision calibration save --output calibration.json
bin/multivision calibration load --input calibration.json
```

The lower-level HTTP equivalents are `POST /calibration/status` and
`POST /metric/calibration/status`. The CLI uses `POST /calibration/load` so the
camera and metric records commit as one operation. Loading skips capture and verification by
design; it is valid only after the operator confirms that the physical setup is
unchanged.
