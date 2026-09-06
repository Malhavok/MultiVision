---
name: multivision-calibration
description: Operate the running MultiVision camera/projector and metric calibration workflow without repeating camera discovery. Use when starting the service, calibrating cameras, recalibrating the metric surface, or retrying a failed hardware capture.
---

# MultiVision calibration

Use the already-running MultiVision service as the authority for camera slots,
projector output, calibration state and capture timing. This skill is an
operator runbook, not a second implementation of calibration.

## Operating rules

- Do not probe camera indexes or rediscover cameras as part of the normal
  workflow. The service owns its session inventory, and the CLI talks to that
  inventory through HTTP.
- Do not run `cameras list` unless the health check fails, a named camera is
  unavailable, or the Driver asks for an inventory report.
- Do not open cameras, detect tags, calculate homographies or render patterns
  outside the service.
- Keep the service running after a successful operation.
- A camera/projector calibration and metric surface calibration are separate
  operations. Never substitute one for the other.
- Runtime tracking processes only cameras with a current usable calibration and
  runs at the default 10 Hz rate.

## Start or reuse the service

1. Check the service once:

   ```sh
   bin/multivision status
   ```

2. If the check fails, start the GUI-owning service once and wait for health:

   ```sh
   nohup .venv/bin/python -m multivision.main >/tmp/multivision-session.log 2>&1 &
   bin/multivision status
   ```

3. If health succeeds, reuse the existing process. Do not restart it merely to
   run another calibration.

4. If `calibration.json` exists, ask the Driver whether to load the saved
   calibration before starting a new capture. Never load it silently. If the
   Driver agrees, run:

   ```sh
   bin/multivision calibration load --input calibration.json
   ```

   The load assumes the cameras, projector output, resolutions and physical
   geometry are unchanged; the service validates those authorities and rejects
   incompatible records without recapturing anything. If the Driver declines,
   leave the current session calibration state unchanged.

## Camera/projector calibration

Use one service-owned full run instead of calibrating cameras one by one:

```sh
bin/multivision full-calibration
```

The command presents one projected pattern, captures all currently available
cameras, verifies accepted transforms with fresh frames, and hides the pattern
when it finishes. Treat a partial report as a useful result: one camera may be
accepted while another has no stable capture window.

## Metric surface calibration

The metric target must be printed at 100% / Actual size and lie flat on the
projected surface. Select a camera that already has a current `CALIBRATED`
camera/projector transform, then run:

```sh
bin/multivision metric calibrate --camera <calibrated-camera-slot>
```

The operation temporarily presents a blank projector frame while it captures
the printed target. Moving the target is a reason to rerun this command; it is
not a reason to rerun camera/projector calibration unless the projector or its
geometry moved.

## Retry and verification

- Retry the same calibration command after correcting the physical setup.
- Do not rediscover cameras before every retry.
- After completion, inspect only the relevant status endpoint unless diagnosis
  is needed:

  ```sh
  bin/multivision calibration status
  bin/multivision metric status
  ```

- A successful camera calibration must report `CALIBRATED`; a successful metric
  calibration must report `state: CALIBRATED`.
- Never infer physical accuracy from fit residuals alone. Validate the projected
  ruler against a physical ruler when accuracy matters.

## Current rig note

The last successful full camera run selected `camera-2` as the master and the
last metric calibration used `camera-2`. Reuse that slot while the current
service session remains unchanged; otherwise choose a currently calibrated
slot from the service status rather than probing the hardware again.
