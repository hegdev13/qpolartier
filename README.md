# QPolarTier Workspace

This workspace is the local development base for the PolarQuant-derived QPolarTier prototype described in [Implementation.md](Implementation.md).

## Layout

- `Implementation.md`: the working implementation plan.
- `qpolartier/`: new package for the prototype logic.
- `experiments/`: validation and benchmarking scripts.

## Repository Setup

This workspace is intended to be a git repository that can be edited locally while the model runs on a remote Lightning AI machine over SSH later.

### Upstream reference

- https://github.com/ericshwu/PolarQuant

## Quick Start

Install the Python dependencies listed in [requirements.txt](requirements.txt), then run the phase-1 validation or evaluation scripts from the remote Lightning machine once SSH is configured.

Examples:

```bash
python experiments/validate_importance.py
python -m qpolartier.evaluate
```
