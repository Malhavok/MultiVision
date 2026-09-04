import math
import unittest
from types import SimpleNamespace

from multivision.fiducials import (
    DetectedMarker,
    FiducialObservation,
    GroupTrackingError,
    build_fiducial_observation,
)
from multivision.geometry import Point2D
from multivision.metric import MetricHomographyPair
from multivision.spatial import (
    CameraGeneration,
    SpatialTracker,
    calculate_stability_score,
)
from multivision.types import Frame


class FakeClock:
    def __init__(self, seconds: float = 0.0) -> None:
        self.seconds = seconds

    def __call__(self) -> float:
        return self.seconds


class SpatialTrackerTest(unittest.TestCase):
    def test_selects_the_stable_camera_for_compound_identities(self) -> None:
        clock = FakeClock()
        tracker = SpatialTracker(
            history_length=3,
            metric_calibration=_metric_calibration(),
            clock=clock,
        )
        for frame_counter in range(1, 4):
            clock.seconds = float(frame_counter - 1)
            state = tracker.update(
                (
                    _observation('cards', 7, 'camera-0', frame_counter, float(frame_counter - 1), 100),
                    _observation('cards', 7, 'camera-1', frame_counter, float(frame_counter - 1), 100 * frame_counter),
                    _observation('tokens', 7, 'camera-1', frame_counter, float(frame_counter - 1), 300),
                ),
            )

        assert state.observations[('cards', 7)].camera_slot == 'camera-0', f'{state=}'
        assert state.observations[('tokens', 7)].camera_slot == 'camera-1', f'{state=}'
        assert tracker.get_history('cards', 7, 'camera-0')[-1].surface.centre == Point2D(100, 100)
        assert len(tracker.get_history('cards', 7, 'camera-0')) == 3
        assert len(tracker.get_history('cards', 7, 'camera-1')) == 3

    def test_deadband_holds_rendered_pose_then_accumulates_motion(self) -> None:
        clock = FakeClock()
        tracker = SpatialTracker(
            metric_calibration=_metric_calibration(),
            update_deadband_mm=5.0,
            update_deadband_degrees=10.0,
            clock=clock,
        )
        first_state = tracker.update(
            (_observation('cards', 1, 'camera-0', 1, 0.0, 100),),
        )
        held_state = tracker.update(
            (_observation('cards', 1, 'camera-0', 2, 1.0, 104),),
        )
        moved_state = tracker.update(
            (_observation('cards', 1, 'camera-0', 3, 2.0, 106),),
        )

        assert first_state.observations[('cards', 1)].surface.centre.x == 100, (
            f'{first_state=}'
        )
        assert held_state.observations[('cards', 1)].surface.centre.x == 100, (
            f'{held_state=}'
        )
        assert held_state.observations[('cards', 1)].frame_counter == 2, (
            f'{held_state=}'
        )
        assert moved_state.observations[('cards', 1)].surface.centre.x == 106, (
            f'{moved_state=}'
        )

    def test_deadband_updates_when_rotation_exceeds_threshold(self) -> None:
        clock = FakeClock()
        tracker = SpatialTracker(
            metric_calibration=_metric_calibration(),
            update_deadband_mm=5.0,
            update_deadband_degrees=10.0,
            clock=clock,
        )
        tracker.update(
            (_observation('cards', 1, 'camera-0', 1, 0.0, 100),),
        )
        held_state = tracker.update(
            (_observation('cards', 1, 'camera-0', 2, 1.0, 100, 9.0),),
        )
        rotated_state = tracker.update(
            (_observation('cards', 1, 'camera-0', 3, 2.0, 100, 11.0),),
        )

        assert held_state.observations[('cards', 1)].surface.orientation_degrees == 0, (
            f'{held_state=}'
        )
        assert abs(
            rotated_state.observations[('cards', 1)].surface.orientation_degrees - 11,
        ) < 1e-9, f'{rotated_state=}'

    def test_numeric_session_slot_ties_use_the_lowest_slot_id(self) -> None:
        clock = FakeClock(10)
        tracker = SpatialTracker(metric_calibration=_metric_calibration(), clock=clock)
        first_state = tracker.update(
            (
                _observation('cards', 1, 'camera-2', 1, 10, 5),
                _observation('cards', 1, 'camera-10', 1, 10, 5),
            ),
        )
        assert first_state.observations[('cards', 1)].camera_slot == 'camera-2', (
            f'{first_state=}'
        )

        clock.seconds = 11
        second_state = tracker.update(
            (
                _observation('cards', 1, 'camera-2', 2, 11, 5),
                _observation('cards', 1, 'camera-10', 2, 11, 5),
            ),
        )
        assert second_state.observations[('cards', 1)].camera_slot == 'camera-2', (
            f'{second_state=}'
        )

    def test_same_camera_candidates_at_one_time_use_the_highest_frame_counter(self) -> None:
        tracker = SpatialTracker(metric_calibration=_metric_calibration(), clock=FakeClock())
        state = tracker.update(
            (
                _observation('cards', 1, 'camera-0', 3, 0.0, 30),
                _observation('cards', 1, 'camera-0', 4, 0.0, 40),
            ),
        )

        selected_observation = state.observations[('cards', 1)]
        assert selected_observation.frame_counter == 4, f'{state=}'
        assert selected_observation.surface.centre == Point2D(40, 100), f'{state=}'

    def test_unwarmed_and_warmed_ties_use_the_specified_order(self) -> None:
        clock = FakeClock(10)
        tracker = SpatialTracker(metric_calibration=_metric_calibration(), clock=clock)
        first_state = tracker.update(
            (
                _observation('cards', 1, 'camera-1', 4, 10, 5),
                _observation('cards', 1, 'camera-0', 8, 10, 6),
            ),
        )
        assert first_state.observations[('cards', 1)].camera_slot == 'camera-0', f'{first_state=}'

        clock.seconds = 11
        second_state = tracker.update(
            (
                _observation('cards', 1, 'camera-1', 5, 11, 5),
                _observation('cards', 1, 'camera-0', 9, 11, 20),
            ),
        )
        assert second_state.observations[('cards', 1)].camera_slot == 'camera-1', f'{second_state=}'
        assert second_state.stability_scores[('cards', 1)] == 0.0, f'{second_state=}'

    def test_candidate_ties_prefer_newer_received_time_before_slot(self) -> None:
        clock = FakeClock()
        tracker = SpatialTracker(metric_calibration=_metric_calibration(), clock=clock)

        tracker.update(
            (
                _observation('cards', 9, 'camera-0', 1, 0.0, 10),
                _observation('cards', 9, 'camera-1', 1, 0.0, 20),
            ),
        )
        clock.seconds = 1.0
        tracker.update(
            (
                _observation('cards', 9, 'camera-0', 2, 1.0, 10),
                _observation('cards', 9, 'camera-1', 2, 1.0, 20),
            ),
        )

        clock.seconds = 3.0
        newer_camera_state = tracker.update(
            (
                _observation('cards', 9, 'camera-0', 3, 2.0, 10),
                _observation('cards', 9, 'camera-1', 3, 3.0, 20),
            ),
        )
        assert newer_camera_state.observations[('cards', 9)].camera_slot == 'camera-1', (
            f'{newer_camera_state=}'
        )

        equal_time_state = tracker.update(
            (
                _observation('cards', 9, 'camera-0', 4, 3.0, 10),
                _observation('cards', 9, 'camera-1', 4, 3.0, 20),
            ),
        )
        assert equal_time_state.observations[('cards', 9)].camera_slot == 'camera-0', (
            f'{equal_time_state=}'
        )


        unwarmed_tracker = SpatialTracker(
            metric_calibration=_metric_calibration(),
            clock=FakeClock(11.0),
        )
        unwarmed_state = unwarmed_tracker.update(
            (
                _observation('cards', 10, 'camera-0', 1, 10.0, 10),
                _observation('cards', 10, 'camera-1', 2, 11.0, 20),
            ),
        )
        assert unwarmed_state.observations[('cards', 10)].camera_slot == 'camera-1', (
            f'{unwarmed_state=}'
        )

    def test_grace_has_no_prediction_and_recovers_at_the_boundary(self) -> None:
        clock = FakeClock()
        tracker = SpatialTracker(metric_calibration=_metric_calibration(), clock=clock)
        tracker.update((_observation('cards', 2, 'camera-0', 1, 0, 25),))

        clock.seconds = 4.999
        retained_state = tracker.update(())
        assert retained_state.observations[('cards', 2)].surface.centre == Point2D(25, 100)
        assert retained_state.last_seen_monotonic_seconds[('cards', 2)] == 0.0

        clock.seconds = 5.0
        expired_state = tracker.snapshot
        assert ('cards', 2) not in expired_state.observations, f'{expired_state=}'

        recovered_state = tracker.update(
            (_observation('cards', 2, 'camera-0', 2, 5.0, 40),),
        )
        assert recovered_state.observations[('cards', 2)].surface.centre == Point2D(40, 100)

    def test_expiry_removes_old_histories_and_readers_do_not_republish(self) -> None:
        clock = FakeClock()
        tracker = SpatialTracker(
            history_length=2,
            grace_period_seconds=5.0,
            metric_calibration=_metric_calibration(),
            clock=clock,
        )
        for marker_id in range(20):
            tracker.update((_observation('cards', marker_id, 'camera-0', 1, 0, marker_id),))

        assert len(tracker.histories) == 20, f'{tracker.histories=}'
        clock.seconds = 5.0
        expired_state = tracker.snapshot
        assert expired_state.observations == {}, f'{expired_state=}'
        assert tracker.histories == {}, f'{tracker.histories=}'

        stable_state = tracker.snapshot
        assert stable_state is expired_state, f'{stable_state=}, {expired_state=}'
        assert stable_state.generation == expired_state.generation, f'{stable_state=}'

    def test_protection_margin_is_a_physical_polygon_offset(self) -> None:
        tracker = SpatialTracker(metric_calibration=_metric_calibration(), clock=FakeClock())
        state = tracker.update((_observation('cards', 8, 'camera-0', 1, 0, 10),))

        assert state.protection_regions[('cards', 8)] == (
            Point2D(0, 90),
            Point2D(20, 90),
            Point2D(20, 110),
            Point2D(0, 110),
        ), f'{state=}'

    def test_metric_authority_named_tuple_and_stale_record_fail_closed(self) -> None:
        clock = FakeClock()
        metric_pair = MetricHomographyPair(
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        )
        pair_tracker = SpatialTracker(metric_calibration=metric_pair, clock=clock)
        pair_state = pair_tracker.update(
            (_observation('cards', 3, 'camera-0', 1, 0, 10),),
        )
        assert pair_state.observations, f'{pair_state=}'

        stale_authority = SimpleNamespace(
            get_record=lambda: SimpleNamespace(
                state='STALE',
                projector_to_surface=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ),
        )
        stale_tracker = SpatialTracker(
            metric_calibration=stale_authority,
            clock=clock,
        )
        stale_state = stale_tracker.update(
            (_observation('cards', 4, 'camera-0', 1, 0, 10),),
        )
        assert stale_state.observations == {}, f'{stale_state=}'

        mutable_authority = _metric_calibration()
        mutable_tracker = SpatialTracker(
            metric_calibration=mutable_authority,
            clock=clock,
        )
        mutable_tracker.update(
            (_observation('cards', 5, 'camera-0', 1, 0, 10),),
        )
        mutable_authority.projector_to_surface = (
            (1, 0, 100),
            (0, 1, 0),
            (0, 0, 1),
        )
        mutable_authority.surface_to_projector = (
            (1, 0, -100),
            (0, 1, 0),
            (0, 0, 1),
        )
        changed_state = mutable_tracker.update(())
        assert changed_state.observations == {}, f'{changed_state=}'
        assert mutable_tracker.histories == {}, f'{mutable_tracker.histories=}'

    def test_in_place_metric_changes_reset_surface_history(self) -> None:
        clock = FakeClock()
        projector_to_surface = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        metric_calibration = SimpleNamespace(
            is_usable=True,
            projector_to_surface=projector_to_surface,
        )
        tracker = SpatialTracker(metric_calibration=metric_calibration, clock=clock)
        tracker.update((_observation('cards', 3, 'camera-0', 1, 0, 10),))

        projector_to_surface[0][2] = 100
        clock.seconds = 1
        state = tracker.update((_observation('cards', 3, 'camera-0', 2, 1, 10),))

        assert len(tracker.get_history('cards', 3, 'camera-0')) == 1, f'{tracker.histories=}'
        assert math.isinf(state.stability_scores[('cards', 3)]), f'{state=}'

    def test_metric_and_camera_authorities_gate_candidates(self) -> None:
        clock = FakeClock()
        tracker = SpatialTracker(metric_calibration=None, clock=clock)
        observation = _observation('cards', 3, 'camera-0', 1, 0, 10)
        assert tracker.update((observation,)).observations == {}, f'{tracker.state=}'

        tracker.set_metric_calibration(_metric_calibration())
        tracker.update_camera_generation('camera-0', 2, 1)
        assert tracker.update((observation,)).observations == {}, f'{tracker.state=}'
        valid_observation = observation._replace(
            camera_lifecycle_generation=2,
            camera_calibration_generation=1,
        )
        assert tracker.update((valid_observation,)).observations, f'{tracker.state=}'
        tracker.invalidate_camera('camera-0')
        assert tracker.state.observations == {}, f'{tracker.state=}'

    def test_invalid_geometry_is_unusable_and_failures_are_diagnostic(self) -> None:
        clock = FakeClock()
        tracker = SpatialTracker(metric_calibration=_metric_calibration(), clock=clock)
        failure = GroupTrackingError('broken', 'DICT_5X5_1000', 'FAILED', 'not available')
        malformed = _observation('cards', 4, 'camera-0', 1, 0, 10)._replace(
            projector=None,
        )
        malformed_surface_observation = _observation(
            'cards',
            10,
            'camera-0',
            1,
            0,
            10,
        )
        assert malformed_surface_observation.surface is not None
        malformed_surface = malformed_surface_observation._replace(
            surface=malformed_surface_observation.surface._replace(
                corners=(
                    Point2D(math.nan, 95),
                    Point2D(15, 95),
                    Point2D(15, 105),
                    Point2D(5, 105),
                ),
            ),
        )
        horizon_tracker = SpatialTracker(
            metric_calibration=SimpleNamespace(
                is_usable=True,
                projector_to_surface=((1, 0, 0), (0, 1, 0), (0.2, 0, -1)),
            ),
            clock=clock,
        )
        missing_matrix_tracker = SpatialTracker(
            metric_calibration=SimpleNamespace(is_usable=True),
            clock=clock,
        )
        state = tracker.update((malformed, malformed_surface), (failure,))
        horizon_state = horizon_tracker.update((_observation('cards', 5, 'camera-0', 1, 0, 0),))
        missing_matrix_state = missing_matrix_tracker.update(
            (_observation('cards', 9, 'camera-0', 1, 0, 0),),
        )
        assert state.observations == {}, f'{state=}'
        assert state.detector_failures == (failure,), f'{state=}'
        assert horizon_state.observations == {}, f'{horizon_state=}'
        assert missing_matrix_state.observations == {}, f'{missing_matrix_state=}'

    def test_snapshots_are_complete_and_immutable(self) -> None:
        tracker = SpatialTracker(
            metric_calibration=_metric_calibration(),
            clock=FakeClock(),
        )
        old_state = tracker.update(
            (_observation('cards', 6, 'camera-0', 1, 0, 1),),
        )
        new_state = tracker.update(
            (
                _observation('cards', 6, 'camera-0', 2, 0, 2),
                _observation('tokens', 6, 'camera-0', 2, 0, 3),
            ),
        )
        assert ('tokens', 6) not in old_state.observations, f'{old_state=}'
        assert ('tokens', 6) in new_state.observations, f'{new_state=}'
        with self.assertRaises(TypeError):
            old_state.observations[('cards', 6)] = old_state.observations[('cards', 6)]  # type: ignore[index]


def _metric_calibration() -> SimpleNamespace:
    return SimpleNamespace(
        is_usable=True,
        projector_to_surface=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        surface_to_projector=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )


def _observation(
    group: str,
    marker_id: int,
    camera_slot: str,
    frame_counter: int,
    timestamp: float,
    x_pos: float,
    orientation_degrees: float = 0.0,
) -> FiducialObservation:
    angle_radians = math.radians(orientation_degrees)
    corners = tuple(
        Point2D(
            x_pos + math.cos(angle_radians) * x_offset
            - math.sin(angle_radians) * y_offset,
            100 + math.sin(angle_radians) * x_offset
            + math.cos(angle_radians) * y_offset,
        )
        for x_offset, y_offset in (
            (-5, -5),
            (5, -5),
            (5, 5),
            (-5, 5),
        )
    )
    return build_fiducial_observation(
        DetectedMarker(marker_id, corners),
        group,
        20,
        camera_to_projector=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        projector_to_surface=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        camera_slot=camera_slot,
        frame_counter=frame_counter,
        received_monotonic_seconds=timestamp,
    )


if __name__ == '__main__':
    unittest.main()
