# Installation

Install the package and its runtime dependencies into an isolated environment:

```sh
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install .
```

The installed CLI is then available as:

```sh
multivision status
multivision cameras list
multivision overlay text --spec-json '...'
multivision metric rotation --camera camera-1 --at-mm=-80,80
```

The `metric rotation` helper reports the local surface-space angle that
straightens the selected camera view at the requested surface position. It
uses the service's current camera and metric calibration records and does not
mutate overlay state.

Install development and test dependencies with the optional extra:

```sh
.venv/bin/python -m pip install '.[dev]'
```

The service runtime has its own console entry point:

```sh
multivision-server
```

`python -m multivision` remains available as an equivalent module entry point
for the CLI.
