"""Immutable fiducial tracking state and deterministic spatial selection."""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, NamedTuple

from multivision.fiducials import (
    FiducialIdentity,
    FiducialObservation,
    GroupFiducialDetectionResult,
)
from multivision.geometry import (
    Point2D,
    TagGeometry,
    build_protected_projector_regions,
    build_tag_geometry,
    invert_homography,
    project_tag_geometry,
    validate_homography,
)
from multivision.types import (
    Frame,
    is_finite_real,
)


Clock = Callable[[], float]
ObservationKey = tuple[FiducialIdentity, str]
_UNSET = object()


class CameraGeneration(NamedTuple):
    """The camera authority against which an observation was accepted."""

    lifecycle_generation: int = 0
    calibration_generation: int = 0
    is_available: bool = True
    is_open: bool = True
    is_calibrated: bool = True


class SpatialState(NamedTuple):
    """One complete immutable spatial publication."""

    generation: int
    metric_calibration: object | None
    selected_observations: Mapping[FiducialIdentity, FiducialObservation]
    last_seen_monotonic_seconds: Mapping[FiducialIdentity, float]
    stability_scores: Mapping[FiducialIdentity, float]
    projector_footprints: Mapping[FiducialIdentity, tuple[Point2D, ...]]
    protection_regions: Mapping[FiducialIdentity, tuple[Point2D, ...]]
    detector_failures: tuple[object, ...]
    camera_generations: Mapping[str, CameraGeneration]

    @classmethod
    def empty(cls) -> 'SpatialState':
        empty_mapping: Mapping[Any, Any] = MappingProxyType({})
        return cls(
            0,
            None,
            empty_mapping,
            empty_mapping,
            empty_mapping,
            empty_mapping,
            empty_mapping,
            (),
            empty_mapping,
        )

    @property
    def observations(self) -> Mapping[FiducialIdentity, FiducialObservation]:
        return self.selected_observations

    @property
    def markers(self) -> Mapping[FiducialIdentity, FiducialObservation]:
        return self.selected_observations

    @property
    def selected(self) -> Mapping[FiducialIdentity, FiducialObservation]:
        return self.selected_observations

    @property
    def selected_markers(self) -> Mapping[FiducialIdentity, FiducialObservation]:
        return self.selected_observations

    @property
    def freshness(self) -> Mapping[FiducialIdentity, float]:
        return self.last_seen_monotonic_seconds

    @property
    def last_seen(self) -> Mapping[FiducialIdentity, float]:
        return self.last_seen_monotonic_seconds

    @property
    def footprints(self) -> Mapping[FiducialIdentity, tuple[Point2D, ...]]:
        return self.projector_footprints

    @property
    def protected_regions(self) -> Mapping[FiducialIdentity, tuple[Point2D, ...]]:
        return self.protection_regions

    @property
    def diagnostic_detector_failures(self) -> tuple[object, ...]:
        return self.detector_failures

    def get_observation(
        self,
        group: str,
        marker_id: int,
    ) -> FiducialObservation | None:
        return self.selected_observations.get(FiducialIdentity(group, marker_id))

    def is_resolved(self, group: str, marker_id: int) -> bool:
        return self.get_observation(group, marker_id) is not None


@dataclass(frozen=True)
class _Candidate:
    observation: FiducialObservation
    history: tuple[FiducialObservation, ...]


class SpatialTracker:
    """Own bounded histories and publish complete spatial snapshots."""

    def __init__(
        self,
        history_length: int = 8,
        grace_period_seconds: float = 5.0,
        metric_calibration: object | None = None,
        protection_margin_mm: float = 5.0,
        update_deadband_mm: float = 5.0,
        update_deadband_degrees: float = 10.0,
        clock: Clock | None = None,
    ) -> None:
        _validate_positive_integer(history_length, 'history_length')
        if history_length > 32:
            raise ValueError('history_length must not exceed 32')
        _validate_positive_number(grace_period_seconds, 'grace_period_seconds')
        _validate_non_negative_number(
            protection_margin_mm,
            'protection_margin_mm',
        )
        _validate_non_negative_number(update_deadband_mm, 'update_deadband_mm')
        _validate_non_negative_number(
            update_deadband_degrees,
            'update_deadband_degrees',
        )
        if clock is not None and not callable(clock):
            raise TypeError('clock must be callable')

        self._history_length = history_length
        self._grace_period_seconds = float(grace_period_seconds)
        self._protection_margin_mm = float(protection_margin_mm)
        self._update_deadband_mm = float(update_deadband_mm)
        self._update_deadband_degrees = float(update_deadband_degrees)
        self._clock = clock if clock is not None else _monotonic_seconds
        self._lock = threading.RLock()
        self._histories: dict[ObservationKey, deque[FiducialObservation]] = {}
        self._current_candidates: dict[ObservationKey, FiducialObservation] = {}
        self._last_selected: dict[FiducialIdentity, FiducialObservation] = {}
        self._last_published: dict[FiducialIdentity, FiducialObservation] = {}
        self._camera_generations: dict[str, CameraGeneration] = {}
        self._metric_calibration = metric_calibration
        self._metric_signature = _calibration_signature(metric_calibration)
        self._detector_failures: tuple[object, ...] = ()
        self._generation = 0
        self._snapshot = SpatialState.empty()
        self._publish_locked()

    @property
    def history_length(self) -> int:
        return self._history_length

    @property
    def grace_period_seconds(self) -> float:
        return self._grace_period_seconds

    @property
    def snapshot(self) -> SpatialState:
        """Return the latest complete snapshot, expiring grace state if needed."""
        with self._lock:
            self._publish_locked(force=False)
            return self._snapshot

    def get_snapshot(self) -> SpatialState:
        return self.snapshot

    def publish(self) -> SpatialState:
        return self.snapshot

    @property
    def state(self) -> SpatialState:
        return self.snapshot

    @property
    def histories(self) -> Mapping[ObservationKey, tuple[FiducialObservation, ...]]:
        with self._lock:
            return MappingProxyType({
                key: tuple(history)
                for key, history in self._histories.items()
            })

    def get_history(
        self,
        group: str,
        marker_id: int,
        camera_slot: str,
    ) -> tuple[FiducialObservation, ...]:
        with self._lock:
            return tuple(
                self._histories.get(
                    (FiducialIdentity(group, marker_id), camera_slot),
                    (),
                )
            )

    def update(
        self,
        observations: Iterable[FiducialObservation] | GroupFiducialDetectionResult = (),
        detector_failures: Iterable[object] = (),
        metric_calibration: object = _UNSET,
        camera_states: Mapping[str, object] | None = None,
        frames_by_camera: Mapping[str, Frame] | None = None,
        *,
        candidates: Iterable[FiducialObservation] | GroupFiducialDetectionResult | None = None,
    ) -> SpatialState:
        """Consume one complete detector cycle and atomically publish its result."""
        if candidates is not None:
            if observations != ():
                raise ValueError('Specify observations or candidates, not both')
            observations = candidates
        checked_observations, checked_failures = _split_detection_result(
            observations,
            detector_failures,
        )
        if camera_states is not None:
            if not isinstance(camera_states, Mapping):
                raise ValueError('camera_states must be a mapping')
            checked_camera_states = {
                camera_slot: _normalise_camera_generation(camera_slot, value)
                for camera_slot, value in camera_states.items()
            }
        else:
            checked_camera_states = None
        if frames_by_camera is not None:
            _validate_frames_by_camera(frames_by_camera)

        with self._lock:
            self._sync_metric_calibration_locked()
            if checked_camera_states is not None:
                self._apply_camera_states_locked(checked_camera_states)
            if metric_calibration is not _UNSET:
                self._set_metric_calibration_locked(metric_calibration)
            self._detector_failures = tuple(checked_failures)
            self._current_candidates = {}
            for observation in checked_observations:
                if frames_by_camera is not None and not _matches_frame(
                    observation,
                    frames_by_camera,
                ):
                    continue
                normalised_observation = self._normalise_usable_observation_locked(
                    observation,
                )
                if normalised_observation is None:
                    continue
                if not self._is_camera_authority_valid_locked(normalised_observation):
                    continue
                key = (
                    normalised_observation.identity,
                    normalised_observation.camera_slot,
                )
                previous_observation = self._current_candidates.get(key)
                if previous_observation is None or _observation_order_key(
                    normalised_observation,
                ) > _observation_order_key(previous_observation):
                    self._current_candidates[key] = normalised_observation

            # A detector may return duplicate evidence for one marker. It is
            # valid evidence, but it is not two temporal samples for stability.
            for key, observation in self._current_candidates.items():
                self._record_history_observation_locked(key, observation)
            return self._publish_locked()

    def consume(
        self,
        observations: Iterable[FiducialObservation] | GroupFiducialDetectionResult = (),
        detector_failures: Iterable[object] = (),
        metric_calibration: object = _UNSET,
        camera_states: Mapping[str, object] | None = None,
        frames_by_camera: Mapping[str, Frame] | None = None,
    ) -> SpatialState:
        return self.update(
            observations,
            detector_failures,
            metric_calibration,
            camera_states,
            frames_by_camera,
        )

    def consume_frames(
        self,
        frames_by_camera: Mapping[str, Frame],
        observations: Iterable[FiducialObservation] | GroupFiducialDetectionResult = (),
        detector_failures: Iterable[object] = (),
        metric_calibration: object = _UNSET,
        camera_states: Mapping[str, object] | None = None,
    ) -> SpatialState:
        return self.update(
            observations,
            detector_failures,
            metric_calibration,
            camera_states,
            frames_by_camera,
        )

    def set_metric_calibration(self, metric_calibration: object | None) -> SpatialState:
        with self._lock:
            self._set_metric_calibration_locked(metric_calibration)
            return self._publish_locked()

    def reset(self, metric_calibration: object | None = None) -> SpatialState:
        """Discard all temporal evidence while retaining an optional authority."""
        with self._lock:
            self._histories.clear()
            self._current_candidates.clear()
            self._last_selected.clear()
            self._last_published.clear()
            self._camera_generations.clear()
            self._detector_failures = ()
            self._set_metric_calibration_locked(metric_calibration)
            return self._publish_locked()

    invalidate = reset

    def update_camera_generation(
        self,
        camera_slot: str,
        lifecycle_generation: int,
        calibration_generation: int = 0,
        is_available: bool = True,
        is_open: bool = True,
        is_calibrated: bool = True,
    ) -> SpatialState:
        generation = CameraGeneration(
            lifecycle_generation,
            calibration_generation,
            is_available,
            is_open,
            is_calibrated,
        )
        _validate_camera_generation(camera_slot, generation)
        with self._lock:
            previous_generation = self._camera_generations.get(camera_slot)
            self._camera_generations[camera_slot] = generation
            if previous_generation != generation:
                self._discard_camera_history_locked(camera_slot)
            return self._publish_locked()

    def set_camera_generation(
        self,
        camera_slot: str,
        lifecycle_generation: int,
        calibration_generation: int = 0,
        is_available: bool = True,
        is_open: bool = True,
        is_calibrated: bool = True,
    ) -> SpatialState:
        return self.update_camera_generation(
            camera_slot,
            lifecycle_generation,
            calibration_generation,
            is_available,
            is_open,
            is_calibrated,
        )

    def invalidate_camera(
        self,
        camera_slot: str,
        lifecycle_generation: int | None = None,
        calibration_generation: int = 0,
    ) -> SpatialState:
        with self._lock:
            previous_generation = self._camera_generations.get(
                camera_slot,
                CameraGeneration(),
            )
            next_lifecycle_generation = (
                previous_generation.lifecycle_generation + 1
                if lifecycle_generation is None
                else lifecycle_generation
            )
            generation = CameraGeneration(
                next_lifecycle_generation,
                calibration_generation,
                False,
                False,
                False,
            )
            _validate_camera_generation(camera_slot, generation)
            self._camera_generations[camera_slot] = generation
            self._discard_camera_history_locked(camera_slot)
            return self._publish_locked()

    def _apply_camera_states_locked(
        self,
        camera_states: Mapping[str, CameraGeneration],
    ) -> None:
        for camera_slot, generation in camera_states.items():
            previous_generation = self._camera_generations.get(camera_slot)
            self._camera_generations[camera_slot] = generation
            if previous_generation != generation:
                self._discard_camera_history_locked(camera_slot)

    def _set_metric_calibration_locked(self, metric_calibration: object | None) -> None:
        self._metric_calibration = metric_calibration
        self._sync_metric_calibration_locked()

    def _sync_metric_calibration_locked(self) -> None:
        new_signature = _calibration_signature(self._metric_calibration)
        if new_signature == self._metric_signature:
            return
        self._metric_signature = new_signature
        self._histories.clear()
        self._current_candidates.clear()
        self._last_selected.clear()

    def _record_history_observation_locked(
        self,
        key: ObservationKey,
        observation: FiducialObservation,
    ) -> None:
        history_values = list(self._histories.get(key, ()))
        sample_key = _history_sample_key(observation)
        history_values = [
            previous_observation
            for previous_observation in history_values
            if _history_sample_key(previous_observation) != sample_key
        ]
        history_values.append(observation)
        history_values.sort(key=_observation_order_key)
        self._histories[key] = deque(
            history_values[-self._history_length:],
            maxlen=self._history_length,
        )

    def _normalise_usable_observation_locked(
        self,
        observation: FiducialObservation,
    ) -> FiducialObservation | None:
        if not _is_valid_observation_record(observation):
            return None
        if not _is_metric_calibration_usable(self._metric_calibration):
            return None

        camera_geometry = _normalise_geometry(observation.camera)
        if camera_geometry is None:
            return None
        projector_geometry = _normalise_geometry(observation.projector)
        if projector_geometry is None:
            return None
        surface_geometry = None
        if observation.surface is not None:
            surface_geometry = _normalise_geometry(observation.surface)
            if surface_geometry is None:
                return None
        try:
            metric_matrix = _get_metric_matrix(
                self._metric_calibration,
                'projector_to_surface',
            )
        except Exception:  # noqa: (A changing calibration authority must fail closed.)
            return None
        if metric_matrix is not None:
            try:
                surface_geometry = project_tag_geometry(projector_geometry, metric_matrix)
            except (TypeError, ValueError):
                return None
        if surface_geometry is None:
            return None
        try:
            return observation._replace(
                camera=camera_geometry,
                projector=projector_geometry,
                surface=surface_geometry,
            )
        except (TypeError, ValueError):
            return None

    def _is_camera_authority_valid_locked(
        self,
        observation: FiducialObservation,
    ) -> bool:
        generation = self._camera_generations.get(observation.camera_slot)
        if generation is None:
            return True
        return (
            generation.is_available
            and generation.is_open
            and generation.is_calibrated
            and generation.lifecycle_generation
            == observation.camera_lifecycle_generation
            and generation.calibration_generation
            == observation.camera_calibration_generation
        )

    def _discard_camera_history_locked(self, camera_slot: str) -> None:
        keys_to_discard = tuple(
            key for key in self._histories if key[1] == camera_slot
        )
        for key in keys_to_discard:
            self._histories.pop(key, None)
        current_keys = tuple(
            key for key in self._current_candidates if key[1] == camera_slot
        )
        for key in current_keys:
            self._current_candidates.pop(key, None)
        selected_keys = tuple(
            identity
            for identity, observation in self._last_selected.items()
            if observation.camera_slot == camera_slot
        )
        for identity in selected_keys:
            self._last_selected.pop(identity, None)
            self._last_published.pop(identity, None)

    def _publish_locked(
        self,
        force: bool = True,
    ) -> SpatialState:
        self._sync_metric_calibration_locked()
        now_seconds = _read_clock(self._clock)
        if not _is_metric_calibration_usable(self._metric_calibration):
            self._histories.clear()
            self._current_candidates.clear()
            self._last_selected.clear()
            selected_candidates: dict[FiducialIdentity, _Candidate] = {}
        else:
            self._prune_expired_state_locked(now_seconds)
            selected_candidates = self._select_candidates_locked(now_seconds)

        selected_observations = {
            identity: candidate.observation
            for identity, candidate in selected_candidates.items()
        }
        last_seen = {
            identity: observation.received_monotonic_seconds
            for identity, observation in selected_observations.items()
        }
        stability_scores = {
            identity: calculate_stability_score(candidate.history)
            for identity, candidate in selected_candidates.items()
        }
        footprints = {
            identity: candidate.observation.projector.corners
            for identity, candidate in selected_candidates.items()
            if candidate.observation.projector is not None
        }
        protection_regions: dict[FiducialIdentity, tuple[Point2D, ...]] = {}
        for identity, candidate in selected_candidates.items():
            projector_geometry = candidate.observation.projector
            if projector_geometry is None:
                continue
            regions = build_protected_projector_regions(
                {identity: projector_geometry.corners},
                self._protection_margin_mm,
                self._metric_calibration,
            )
            if len(regions) > 0:
                protection_regions[identity] = regions[0]
        if not force and _snapshot_contents_match(
            self._snapshot,
            self._metric_calibration,
            selected_observations,
            last_seen,
            stability_scores,
            footprints,
            protection_regions,
            self._detector_failures,
            self._camera_generations,
        ):
            return self._snapshot

        self._generation += 1
        self._snapshot = SpatialState(
            self._generation,
            self._metric_calibration,
            MappingProxyType(selected_observations),
            MappingProxyType(last_seen),
            MappingProxyType(stability_scores),
            MappingProxyType(footprints),
            MappingProxyType(protection_regions),
            self._detector_failures,
            MappingProxyType(dict(self._camera_generations)),
        )
        return self._snapshot

    def _prune_expired_state_locked(self, now_seconds: float) -> None:
        expired_history_keys = tuple(
            key
            for key, history in self._histories.items()
            if len(history) == 0
            or _observation_age_seconds(history[-1], now_seconds)
            >= self._grace_period_seconds
        )
        for key in expired_history_keys:
            self._histories.pop(key, None)

        expired_candidate_keys = tuple(
            key
            for key, observation in self._current_candidates.items()
            if (
                _observation_age_seconds(observation, now_seconds)
                >= self._grace_period_seconds
                or not self._is_camera_authority_valid_locked(observation)
            )
        )
        for key in expired_candidate_keys:
            self._current_candidates.pop(key, None)

        expired_identities = tuple(
            identity
            for identity, observation in self._last_selected.items()
            if (
                _observation_age_seconds(observation, now_seconds)
                >= self._grace_period_seconds
                or not self._is_camera_authority_valid_locked(observation)
            )
        )
        for identity in expired_identities:
            self._last_selected.pop(identity, None)
            self._last_published.pop(identity, None)

    def _select_candidates_locked(
        self,
        now_seconds: float,
    ) -> dict[FiducialIdentity, _Candidate]:
        candidates_by_identity: dict[FiducialIdentity, list[_Candidate]] = {}
        for key, observation in self._current_candidates.items():
            if _observation_age_seconds(observation, now_seconds) >= self._grace_period_seconds:
                continue
            history = tuple(self._histories.get(key, ()))
            candidates_by_identity.setdefault(observation.identity, []).append(
                _Candidate(observation, history),
            )

        selected: dict[FiducialIdentity, _Candidate] = {}
        identities = set(candidates_by_identity) | set(self._last_selected)
        for identity in sorted(identities):
            current_candidates = candidates_by_identity.get(identity, [])
            if len(current_candidates) > 0:
                warmed_candidates = [
                    candidate
                    for candidate in current_candidates
                    if len(candidate.history) >= 2
                ]
                selected_candidate = min(
                    warmed_candidates
                    if len(warmed_candidates) > 0
                    else current_candidates,
                    key=(
                        _warmed_candidate_key
                        if len(warmed_candidates) > 0
                        else _unwarmed_candidate_key
                    ),
                )
                candidate_observation = selected_candidate.observation
                previous_observation = self._last_published.get(identity)
                published_observation = candidate_observation
                if previous_observation is not None and not _pose_exceeds_deadband(
                    previous_observation,
                    candidate_observation,
                    self._update_deadband_mm,
                    self._update_deadband_degrees,
                ):
                    published_observation = _refresh_observation_metadata(
                        previous_observation,
                        candidate_observation,
                    )
                selected[identity] = _Candidate(
                    published_observation,
                    selected_candidate.history,
                )
                self._last_selected[identity] = published_observation
                self._last_published[identity] = published_observation
                continue

            retained_observation = self._last_selected.get(identity)
            if retained_observation is None:
                continue
            if (
                _observation_age_seconds(retained_observation, now_seconds)
                < self._grace_period_seconds
                and self._is_camera_authority_valid_locked(retained_observation)
            ):
                key = (identity, retained_observation.camera_slot)
                selected[identity] = _Candidate(
                    retained_observation,
                    tuple(self._histories.get(key, ())),
                )
        return selected


def _pose_exceeds_deadband(
    previous: FiducialObservation,
    current: FiducialObservation,
    deadband_mm: float,
    deadband_degrees: float,
) -> bool:
    if (
        previous.camera_slot != current.camera_slot
        or previous.camera_lifecycle_generation != current.camera_lifecycle_generation
        or previous.camera_calibration_generation != current.camera_calibration_generation
    ):
        return True
    previous_surface = previous.surface
    current_surface = current.surface
    if previous_surface is None or current_surface is None:
        return True
    displacement_mm = math.hypot(
        current_surface.centre.x - previous_surface.centre.x,
        current_surface.centre.y - previous_surface.centre.y,
    )
    angle_delta = (
        current_surface.orientation_degrees
        - previous_surface.orientation_degrees
    )
    rotation_degrees = abs((angle_delta + 180.0) % 360.0 - 180.0)
    return displacement_mm > deadband_mm or rotation_degrees > deadband_degrees


def _refresh_observation_metadata(
    previous: FiducialObservation,
    current: FiducialObservation,
) -> FiducialObservation:
    return previous._replace(
        marker_size_mm=current.marker_size_mm,
        camera_slot=current.camera_slot,
        camera_lifecycle_generation=current.camera_lifecycle_generation,
        frame_counter=current.frame_counter,
        received_monotonic_seconds=current.received_monotonic_seconds,
        camera_calibration_generation=current.camera_calibration_generation,
    )


def calculate_stability_score(
    observations: Iterable[FiducialObservation | Point2D | tuple[float, float]],
) -> float:
    """Return mean consecutive Euclidean displacement in surface millimetres."""
    values = tuple(observations)
    if len(values) < 2:
        return math.inf
    centres = tuple(
        value.surface.centre
        if isinstance(value, FiducialObservation) and value.surface is not None
        else _coerce_score_point(value)
        for value in values
    )
    if any(point is None for point in centres):
        return math.inf
    checked_centres = tuple(point for point in centres if point is not None)
    if len(checked_centres) != len(centres):
        return math.inf
    displacement_values = tuple(
        math.hypot(
            checked_centres[idx].x - checked_centres[idx - 1].x,
            checked_centres[idx].y - checked_centres[idx - 1].y,
        )
        for idx in range(1, len(centres))
    )
    if any(not math.isfinite(value) for value in displacement_values):
        return math.inf
    return sum(displacement_values) / len(displacement_values)


calculate_surface_displacement_score = calculate_stability_score
SpatialStateTracker = SpatialTracker
Tracker = SpatialTracker


def _snapshot_contents_match(
    snapshot: SpatialState,
    metric_calibration: object | None,
    selected_observations: Mapping[FiducialIdentity, FiducialObservation],
    last_seen: Mapping[FiducialIdentity, float],
    stability_scores: Mapping[FiducialIdentity, float],
    footprints: Mapping[FiducialIdentity, tuple[Point2D, ...]],
    protection_regions: Mapping[FiducialIdentity, tuple[Point2D, ...]],
    detector_failures: tuple[object, ...],
    camera_generations: Mapping[str, CameraGeneration],
) -> bool:
    return (
        snapshot.metric_calibration is metric_calibration
        and snapshot.selected_observations == selected_observations
        and snapshot.last_seen_monotonic_seconds == last_seen
        and snapshot.stability_scores == stability_scores
        and snapshot.projector_footprints == footprints
        and snapshot.protection_regions == protection_regions
        and snapshot.detector_failures == detector_failures
        and snapshot.camera_generations == camera_generations
    )


def _split_detection_result(
    observations: Iterable[FiducialObservation] | GroupFiducialDetectionResult,
    detector_failures: Iterable[object],
) -> tuple[tuple[FiducialObservation, ...], tuple[object, ...]]:
    if isinstance(observations, GroupFiducialDetectionResult):
        return observations.observations, tuple(observations.errors) + tuple(detector_failures)
    try:
        checked_observations = tuple(observations)
        checked_failures = tuple(detector_failures)
    except TypeError as ex:
        raise ValueError('observations and detector_failures must be iterable') from ex
    return checked_observations, checked_failures


def _normalise_camera_generation(
    camera_slot: object,
    value: object,
) -> CameraGeneration:
    if not isinstance(camera_slot, str) or len(camera_slot) == 0:
        raise ValueError('camera slot must be a non-empty string')
    if isinstance(value, CameraGeneration):
        generation = value
    else:
        lifecycle_generation = _get_authority_field(
            value,
            'lifecycle_generation',
            0,
        )
        calibration_generation = _get_authority_field(
            value,
            'calibration_generation',
            _get_authority_field(value, 'calibration_version', 0),
        )
        state_value = _get_authority_field(value, 'state', _UNSET)
        runtime_value = _get_authority_field(value, 'runtime_status', _UNSET)
        calibration_status = _get_authority_field(
            value,
            'calibration_status',
            _UNSET,
        )
        state_text = _authority_text(state_value)
        runtime_text = _authority_text(runtime_value)
        calibration_text = _authority_text(calibration_status)
        is_open_default = _derive_open_state(state_text, runtime_text)
        is_calibrated_default = (
            calibration_status is _UNSET
            or calibration_text == 'CALIBRATED'
        )
        generation = CameraGeneration(
            lifecycle_generation,
            calibration_generation,
            _get_authority_field(value, 'is_available', is_open_default),
            _get_authority_field(value, 'is_open', is_open_default),
            _get_authority_field(value, 'is_calibrated', is_calibrated_default),
        )
    _validate_camera_generation(camera_slot, generation)
    return generation


def _get_authority_field(
    value: object,
    field_name: str,
    default: object,
) -> object:
    if isinstance(value, Mapping):
        return value.get(field_name, default)
    return getattr(value, field_name, default)


def _authority_text(value: object) -> str | None:
    if value is _UNSET:
        return None
    text = getattr(value, 'value', value)
    return text if isinstance(text, str) else None


def _derive_open_state(
    state_text: str | None,
    runtime_text: str | None,
) -> bool:
    if state_text is not None:
        return state_text == 'OPEN'
    if runtime_text is not None:
        return runtime_text == 'AVAILABLE'
    return True


def _validate_camera_generation(
    camera_slot: object,
    generation: CameraGeneration,
) -> None:
    if not isinstance(camera_slot, str) or len(camera_slot) == 0:
        raise ValueError('camera slot must be a non-empty string')
    for value, field_name in (
        (generation.lifecycle_generation, 'lifecycle_generation'),
        (generation.calibration_generation, 'calibration_generation'),
    ):
        _validate_non_negative_integer(value, field_name)
    if not all(isinstance(value, bool) for value in generation[2:]):
        raise ValueError('camera generation flags must be booleans')


def _validate_frames_by_camera(frames_by_camera: Mapping[str, Frame]) -> None:
    for camera_slot, frame in frames_by_camera.items():
        if not isinstance(camera_slot, str) or not isinstance(frame, Frame):
            raise ValueError('frames_by_camera must map slots to Frame values')


def _matches_frame(
    observation: FiducialObservation,
    frames_by_camera: Mapping[str, Frame],
) -> bool:
    frame = frames_by_camera.get(observation.camera_slot)
    return frame is not None and frame.frame_counter == observation.frame_counter


def _is_valid_observation_record(observation: object) -> bool:
    if not isinstance(observation, FiducialObservation):
        return False
    if (
        not isinstance(observation.group, str)
        or len(observation.group.strip()) == 0
        or not isinstance(observation.id, int)
        or isinstance(observation.id, bool)
        or observation.id < 0
        or not is_finite_real(observation.marker_size_mm)
        or observation.marker_size_mm <= 0
        or not isinstance(observation.camera_slot, str)
        or len(observation.camera_slot) == 0
        or not _is_non_negative_integer(observation.camera_lifecycle_generation)
        or not _is_non_negative_integer(observation.camera_calibration_generation)
        or not _is_non_negative_integer(observation.frame_counter)
        or not is_finite_real(observation.received_monotonic_seconds)
    ):
        return False
    return _normalise_geometry(observation.camera) is not None


def _normalise_geometry(geometry: TagGeometry | None) -> TagGeometry | None:
    if not isinstance(geometry, TagGeometry):
        return None
    try:
        checked_geometry = build_tag_geometry(geometry.corners, geometry.centre)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not is_finite_real(geometry.orientation_degrees)
        or not is_finite_real(geometry.area_px)
        or geometry.area_px <= 0
    ):
        return None
    return checked_geometry


def _is_metric_calibration_usable(metric_calibration: object | None) -> bool:
    if metric_calibration is None:
        return False
    try:
        is_usable = getattr(metric_calibration, 'is_usable', None)
        state = getattr(metric_calibration, 'state', None)
        get_record = getattr(metric_calibration, 'get_record', None)
    except Exception:  # noqa: (Malformed calibration must fail closed.)
        return False
    try:
        record = get_record() if callable(get_record) else metric_calibration
        if record is None:
            return False
        record_state = getattr(record, 'state', None)
        metric_matrix = _get_metric_matrix(
            metric_calibration,
            'projector_to_surface',
        )
        has_metric_matrix = metric_matrix is not None
        if has_metric_matrix:
            validate_homography(metric_matrix)
    except Exception:  # noqa: (Malformed calibration must fail closed.)
        return False
    has_authority_marker = (
        is_usable is not None
        or state is not None
        or record_state is not None
        or callable(get_record)
        or has_metric_matrix
    )
    if not has_metric_matrix:
        return False
    if not has_authority_marker:
        return False
    if callable(is_usable):
        try:
            if not bool(is_usable()):
                return False
        except Exception:  # noqa: (A calibration registry is an external authority.)
            return False
    elif is_usable is not None:
        if not isinstance(is_usable, bool) or not is_usable:
            return False
    state_value = getattr(state, 'value', state)
    record_state_value = getattr(record_state, 'value', record_state)
    if (
        isinstance(state_value, str)
        and state_value in {'STALE', 'UNCALIBRATED', 'UNVERIFIED'}
    ):
        return False
    if (
        isinstance(record_state_value, str)
        and record_state_value in {'STALE', 'UNCALIBRATED', 'UNVERIFIED'}
    ):
        return False
    return True


def _get_metric_matrix(
    metric_calibration: object | None,
    direction: str,
) -> object | None:
    if metric_calibration is None:
        return None
    record = metric_calibration
    get_record = getattr(metric_calibration, 'get_record', None)
    if callable(get_record):
        record = get_record()
    if record is None:
        return None
    matrix = getattr(record, direction, None)
    if matrix is not None:
        return matrix
    homography = getattr(record, 'homography', None)
    if homography is not None:
        return getattr(homography, direction, homography)
    if isinstance(record, Sequence) and not isinstance(
        record,
        (str, bytes, bytearray),
    ):
        matrix = validate_homography(record)
        return (
            matrix
            if direction == 'projector_to_surface'
            else invert_homography(matrix)
        )
    if direction == 'surface_to_projector':
        projector_to_surface = getattr(record, 'projector_to_surface', None)
        if projector_to_surface is not None:
            return invert_homography(validate_homography(projector_to_surface))
    return None


def _observation_age_seconds(
    observation: FiducialObservation,
    now_seconds: float,
) -> float:
    return max(0.0, now_seconds - observation.received_monotonic_seconds)


def _history_sample_key(
    observation: FiducialObservation,
) -> tuple[int, float]:
    return observation.frame_counter, observation.received_monotonic_seconds


def _observation_order_key(
    observation: FiducialObservation,
) -> tuple[float, int, tuple[Point2D, ...]]:
    return (
        observation.received_monotonic_seconds,
        observation.frame_counter,
        observation.camera.corners,
    )


def _warmed_candidate_key(
    candidate: _Candidate,
) -> tuple[float, float, tuple[int, int | str], int]:
    return (
        calculate_stability_score(candidate.history),
        -candidate.observation.received_monotonic_seconds,
        _camera_slot_order_key(candidate.observation.camera_slot),
        -candidate.observation.frame_counter,
    )


def _unwarmed_candidate_key(
    candidate: _Candidate,
) -> tuple[float, tuple[int, int | str], int]:
    return (
        -candidate.observation.received_monotonic_seconds,
        _camera_slot_order_key(candidate.observation.camera_slot),
        -candidate.observation.frame_counter,
    )


def _camera_slot_order_key(camera_slot: str) -> tuple[int, int | str]:
    prefix, separator, index = camera_slot.rpartition('-')
    if prefix == 'camera' and separator != '' and index.isdigit():
        return (0, int(index))
    return (1, camera_slot)


def _calibration_signature(metric_calibration: object | None) -> object:
    if metric_calibration is None:
        return None
    try:
        record = metric_calibration
        get_record = getattr(metric_calibration, 'get_record', None)
        if callable(get_record):
            record = get_record()
        if record is None:
            return (id(metric_calibration), None)
        return (
            id(metric_calibration),
            id(record),
            _signature_value(getattr(record, 'state', None)),
            _signature_value(getattr(record, 'timestamp', None)),
            _signature_value(getattr(record, 'is_usable', None)),
            _matrix_signature(
                _get_metric_matrix(metric_calibration, 'projector_to_surface'),
            ),
            _matrix_signature(
                _get_metric_matrix(metric_calibration, 'surface_to_projector'),
            ),
        )
    except Exception:  # noqa: (A changing calibration authority must fail closed.)
        return (id(metric_calibration), 'invalid')


def _matrix_signature(matrix: object | None) -> object:
    if matrix is None:
        return None
    try:
        return tuple(
            tuple(float(value) for value in row)
            for row in matrix  # type: ignore[union-attr]
        )
    except (OverflowError, TypeError, ValueError):
        return _signature_value(matrix)


def _signature_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, Mapping):
        return (
            'mapping',
            tuple(
                sorted(
                    (
                        _signature_value(key),
                        _signature_value(item),
                    )
                    for key, item in value.items()
                )
            ),
        )
    if isinstance(value, Sequence):
        return ('sequence', tuple(_signature_value(item) for item in value))
    return ('object', id(value), repr(value))


def _read_clock(clock: Clock) -> float:
    value = clock()
    if not is_finite_real(value):
        raise ValueError('clock must return a finite number')
    return float(value)


def _monotonic_seconds() -> float:
    import time

    return time.monotonic()


def _coerce_score_point(value: object) -> Point2D | None:
    if isinstance(value, Point2D):
        return value if _is_finite_point(value) else None
    try:
        x_pos, y_pos = value  # type: ignore[misc]
    except (TypeError, ValueError):
        return None
    point = Point2D(x_pos, y_pos)
    return point if _is_finite_point(point) else None


def _is_finite_point(point: object) -> bool:
    return (
        isinstance(point, Point2D)
        and is_finite_real(point.x)
        and is_finite_real(point.y)
    )


def _is_non_negative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_positive_integer(value: object, field_name: str) -> None:
    if not _is_non_negative_integer(value) or value == 0:
        raise ValueError(f'{field_name} must be a positive integer')


def _validate_non_negative_integer(value: object, field_name: str) -> None:
    if not _is_non_negative_integer(value):
        raise ValueError(f'{field_name} must be a non-negative integer')


def _validate_positive_number(value: object, field_name: str) -> None:
    if not is_finite_real(value) or float(value) <= 0:
        raise ValueError(f'{field_name} must be a finite positive number')


def _validate_non_negative_number(value: object, field_name: str) -> None:
    if not is_finite_real(value) or float(value) < 0:
        raise ValueError(f'{field_name} must be a finite non-negative number')


__all__ = [
    'CameraGeneration',
    'SpatialState',
    'SpatialStateTracker',
    'SpatialTracker',
    'Tracker',
    'calculate_stability_score',
    'calculate_surface_displacement_score',
]
