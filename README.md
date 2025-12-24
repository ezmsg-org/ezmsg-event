# ezmsg-event

ezmsg namespace package for working with signal events like neural spikes and heartbeats

## Overview

``ezmsg-event`` ezmsg namespace package for working with signal events like neural spikes and heartbeats.

Key features:

* **Event detection** - Detect and track signal events in neural data
* **Spike handling** - Process neural spike events
* **Heartbeat tracking** - Monitor physiological heartbeat events
* **Event timestamps** - Precise timing for event occurrences

## Installation

```bash
pip install ezmsg-event
```

## Dependencies

- `ezmsg`
- `numpy`
- `ezmsg.baseproc`
- `ezmsg.sigproc`

## Usage

TODO: Add usage examples

## Development

We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for development.

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.
2. Fork this repository and clone your fork locally.
3. Open a terminal and `cd` to the cloned folder.
4. Run `uv sync` to create a `.venv` and install dependencies.
5. (Optional) Install pre-commit hooks: `uv run pre-commit install`
6. After making changes, run the test suite: `uv run pytest tests`

## License

MIT License - see [LICENSE](LICENSE) for details.
