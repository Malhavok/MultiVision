from camera import enumerate_devices, save_screen


def main() -> None:
    devices = enumerate_devices()
    for index in devices.values():
        save_screen(index)


if __name__ == '__main__':
    main()
