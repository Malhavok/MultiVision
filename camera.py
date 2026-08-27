import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import pygame


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

    for index in range(len(names)):
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


def _capture_frame(index: int) -> Any:
    capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera at index {index}")

        success, frame = capture.read()
        if not success or frame is None:
            raise RuntimeError(f"Could not read a frame from camera at index {index}")
        return frame
    finally:
        capture.release()


def grab_screen(index: int) -> pygame.Surface:
    """Capture camera ``index`` as a pygame-compatible surface."""
    frame = cv2.cvtColor(_capture_frame(index), cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]
    return pygame.image.frombuffer(frame.tobytes(), (width, height), "RGB")


def save_screen(index: int) -> None:
    """Capture camera ``index`` and save it as ``screenshot-<index>.png``."""
    output_path = Path(f"screenshot-{index}.png")
    pygame.image.save(grab_screen(index), str(output_path))
