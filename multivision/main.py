from multivision.application import MultiVisionService
from multivision.display import DisplayConfiguration, PygameDisplayRuntime
from multivision.server import ApiServerRuntime


def main() -> None:
    service = MultiVisionService()
    display_runtime = PygameDisplayRuntime(
        service,
        DisplayConfiguration(
            projector_resolution=service.configuration.projector_resolution,
            preview_mode=service.configuration.preview_mode,
            preview_low_rate_hz=service.configuration.preview_low_rate_hz,
        ),
        calibration_pattern=service.calibration_pattern,
    )
    api_runtime = ApiServerRuntime(service)
    try:
        service.start()
        api_runtime.start()
        display_runtime.run()
    finally:
        try:
            display_runtime.shutdown()
        finally:
            try:
                api_runtime.shutdown()
            finally:
                service.shutdown()


if __name__ == '__main__':
    main()
