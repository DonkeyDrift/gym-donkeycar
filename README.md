# Gymnasium Environments for Donkey Car

[![pypi](https://img.shields.io/pypi/v/gym-donkeycar.svg)](https://pypi.python.org/pypi/gym-donkeycar) [![CI](https://github.com/tawnkramer/gym-donkeycar/workflows/CI/badge.svg)](https://github.com/tawnkramer/gym-donkeycar/actions) [![Documentation Status](https://readthedocs.org/projects/gym-donkeycar/badge/?version=latest)](https://gym-donkeycar.readthedocs.io/en/latest/?badge=latest)

Donkey Car Gymnasium (formerly OpenAI Gym)

  - Free software: MIT license
  - Documentation: <https://gym-donkeycar.readthedocs.io/en/latest/>

## Installation

Download simulator binaries: https://github.com/tawnkramer/gym-donkeycar/releases

Install master version of gym donkey car:

```shell
pip install git+https://github.com/tawnkramer/gym-donkeycar
```

## Example Usage

A short and compact introduction for people who know gym/gymnasium environments,
but want to understand this one. Simple example code:

```python
import os
import gymnasium as gym
import gym_donkeycar
import numpy as np

# SET UP ENVIRONMENT
# You can also launch the simulator separately
# in that case, you don't need to pass a `conf` object
exe_path = f"{PATH_TO_APP}/donkey_sim.exe"
port = 9091

conf = { "exe_path" : exe_path, "port" : port }

env = gym.make("donkey-generated-track-v0", conf=conf)

# PLAY
obs, info = env.reset()
for t in range(100):
  action = np.array([0.0, 0.5]) # drive straight with small speed
  # execute the action
  obs, reward, terminated, truncated, info = env.step(action)

# Exit the scene
env.close()
```

or if you already launched the simulator:

```python
import gymnasium as gym
import numpy as np

import gym_donkeycar

env = gym.make("donkey-warren-track-v0")

obs, info = env.reset()
try:
    for _ in range(100):
        # drive straight with small speed
        action = np.array([0.0, 0.5])  
        # execute the action
        obs, reward, terminated, truncated, info = env.step(action)
except KeyboardInterrupt:
    # You can kill the program using ctrl+c
    pass

    # Exit the scene
env.close()
```

- see more examples: https://github.com/tawnkramer/gym-donkeycar/tree/master/examples

## Managed Simulator Installation

Instead of downloading the simulator from the
[release page](https://github.com/tawnkramer/gym-donkeycar/releases) by hand,
you can let `gym_donkeycar` manage the binary for you. The package ships the
`donkey-sim-update` command line tool, which keeps an installation directory
(`~/.donkeycar/simulators` by default) in sync with the upstream releases.

### One-command update

```shell
donkey-sim-update update
```

This downloads the latest release asset for your platform (`DonkeySimMac.zip`,
`DonkeySimLinux.zip` or `DonkeySimWin.zip`), verifies its SHA-256 checksum,
stages it and applies it. Download progress is printed as a percentage.

Other subcommands:

```shell
donkey-sim-update status            # installed/latest version and executable path
donkey-sim-update check             # report whether a newer version is available
donkey-sim-update update --dry-run  # show repo/current/latest/asset/size/target only
donkey-sim-update use v25.10.06     # switch to an already installed version
donkey-sim-update clean             # remove staging files and old versions
donkey-sim-update launch            # start the managed simulator directly
```

### Using the managed simulator from the environment

Pass `exe_path="managed"` in the environment configuration and the simulator
is resolved from the managed installation:

```python
conf = {"exe_path": "managed", "port": 9091}
env = gym.make("donkey-generated-track-v0", conf=conf)
```

### Check and auto modes

`status` and `check` are read-only: they never modify installed or staged
versions (at most they refresh the 24-hour release lookup cache).
`update` runs in auto mode by default: download, verify and stage the new
version and apply it immediately. If you pass `--pid` while a simulator is
still running, the update is only staged until that process is gone.
`update --version <version>` switches to an already installed version without
downloading; only the latest release can be downloaded.

### Applying after exit, and restarting

To apply an update after a running simulator exits:

```shell
donkey-sim-update update --apply-after-exit --pid 12345
```

Add `--restart` to launch the newly installed version right after applying.
`--host`, `--port` and `--headless` control how it is started:

```shell
donkey-sim-update update --apply-after-exit --pid 12345 --restart --headless
```

### Safety

- Every download is checksum-verified (SHA-256) before it is staged; corrupt
  or truncated downloads are discarded.
- `use <version>` can switch back to a previously installed version.
- With `--pid`, the update never replaces the binary while the simulator may
  be running; with `--apply-after-exit` it waits for the process to exit
  first.
- `status` and `check` never change the installed or staged version.

### Upstream repository

The release source is configurable with `--repo`; `--root` changes the
installation directory, and `--timeout` the network timeout. These options
work before or after the subcommand:

```shell
donkey-sim-update --repo tawnkramer/gym-donkeycar --root ~/donkey-sim update
donkey-sim-update update --root ~/donkey-sim --dry-run
```

Exit codes: `0` success, `2` invalid arguments or installation state,
`3` network, download or verification failure.

## Action space

A permissable action is a numpy array of length two with first steering
and throttle, respectively. E.g. `np.array([0,1])` goes straight at full
speed, `np.array([-5,1])` turns left etc.

Action Space: Box(2,)

Action names: `['steer', 'throttle']`

What you receive back on step:

- obs: The image that the donkey is seeing (np.array shape
  (120,160,3))
- reward: a reward that combines game over, how far from center and
  speed (max=1, min approx -2)
- done: Boolean. Game over if cte > max_cte or hit != "none"
- info contains:
    - cte: Cross track error (how far from center line)
    - positions: x,y,z
    - speed: positive forward, negative backward
    - hit: 'none' if all is good.
    - last_lap_time: time of last successful lap in seconds, 0.0 if there isn't one

Example info:

```python
{'pos': (51.49209, 0.7399381, 117.3004),
 'cte': -5.865292,
 'speed': 9.319956,
 'hit': 'none',
 'last_lap_time': 34.93437361717224}
```

## Environments

- "donkey-warehouse-v0"
- "donkey-generated-roads-v0"
- "donkey-avc-sparkfun-v0"
- "donkey-generated-track-v0"
- "donkey-roboracingleague-track-v0"
- "donkey-waveshare-v0"
- "donkey-minimonaco-track-v0"
- "donkey-warren-track-v0"
- "donkey-thunderhill-track-v0"
- "donkey-circuit-launch-track-v0"


## Codestyle

We are using [black codestyle](https://github.com/psf/black) (max line length of 127 characters) together with [isort](https://github.com/timothycrosley/isort) to sort the imports.

**Please run `make format`** to reformat your code. You can check the codestyle using `make check-codestyle` and `make lint`.

## Tests

Type checking with `pytype`:

```
make type
```

Codestyle check with `black`, `isort` and `ruff`:

```
make check-codestyle
make lint
```

To run `pytype`, `format` and `lint` in one command:
```
make commit-checks
```

Build the documentation:

```
make docs
```


## Credits

Original Source Code

Tawn Kramer - <https://github.com/tawnkramer/gym-donkeycar>

Roma Sokolkov - <https://github.com/r7vme/gym-donkeycar> cloned with
permission from <https://github.com/tawnkramer/sdsandbox>

Maintainer

Maxime Ellerbach - <https://github.com/Maximellerbach/gym-donkeycar>

Release Engineer
Leigh Johnson: https://github.com/leigh-johnson

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
