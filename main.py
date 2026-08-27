import json
import subprocess
import sys
from pathlib import Path

import cv2


_MAX_DEVICE_INDEX = 10


def _system_camera_names() -> list[str]:
    """Return camera names reported by macOS, or an empty list on failure."""
    if sys.platform != "darwin":
        return []

    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        profile = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []

    entries = profile.get("SPCameraDataType", [])
    if not isinstance(entries, list):
        return []

    names: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            name = entry.get("_name")
            if isinstance(name, str) and name:
                names.append(name)
    return names


def enumerate_devices() -> dict[str, int]:
    """
    List available cameras by human-readable name and OpenCV index.

    OpenCV does not provide a portable API for listing camera devices. On
    macOS, ``system_profiler`` supplies the names while AVFoundation probes
    determine which corresponding indices can actually be opened. The
    fallback name is useful when macOS metadata is unavailable.
    """
    names = _system_camera_names()
    devices: dict[str, int] = {}

    for index in range(_MAX_DEVICE_INDEX):
        capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        try:
            if not capture.isOpened():
                continue

            name = names[index] if index < len(names) else f"Camera {index}"
            original_name = name
            suffix = 2
            while name in devices:
                name = f"{original_name} ({suffix})"
                suffix += 1
            devices[name] = index
        finally:
            capture.release()

    return devices


def save_screen(name: str, index: int) -> None:
    """Capture one frame from ``index`` and save it as ``<name>.png``."""
    capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera {name!r} (index {index})")

        success, frame = capture.read()
        if not success or frame is None:
            raise RuntimeError(f"Could not read a frame from camera {name!r}")

        output_path = Path(f"{name}.png")
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Could not save screenshot to {output_path}")
    finally:
        capture.release()


def main() -> None:
    devices = enumerate_devices()
    for device, idx in devices.items():
        save_screen(device, idx)


if __name__ == '__main__':
    main()
