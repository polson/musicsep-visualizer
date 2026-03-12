# musicsep-visualizer

`musicsep-visualizer` is a CUDA/OpenGL tensor viewer for PyTorch models. It lets you drop lightweight hooks into a network and inspect intermediate activations in a separate realtime window while training or debugging.

## What it does

- Streams CUDA tensors from model hooks into a shared GPU ring buffer
- Renders 2D views at interactive frame rates using OpenGL
- Supports 2D / 3D / 4D tensors (batch/channel selection logic included)
- Shows live stats (shape, min/max, mean/std, FPS, gamma, dB range)
- Lets you switch between named hook points at runtime

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA
- Working OpenGL context (desktop environment)
- PyTorch with CUDA support

Runtime dependencies (installed automatically):

- `torch>=2.8.0`
- `cuda-python`
- `pygame>=2.6.1`
- `PyOpenGL>=3.1.10`

## Installation

Install from Git with `pip`:

```bash
pip install "git+https://github.com/polson/musicsep-visualizer.git"
```

Install from Git with `uv`:

```bash
uv pip install "git+https://github.com/polson/musicsep-visualizer.git"
```

## Quick start

```python
import torch
from musicsep_visualizer import VisualizationHook

model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, 3, padding=1),
    VisualizationHook("encoder.conv1"),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64, 64, 3, padding=1),
    VisualizationHook("encoder.conv2", gamma=2.0),
).cuda()

x = torch.randn(1, 3, 512, 512, device="cuda")

for _ in range(1000):
    _ = model(x)

# Optional explicit cleanup (also runs on interpreter exit)
VisualizationHook.stop_visualization()
```

The first few forward passes are treated as warmup. After warmup, the visualizer process launches and opens a window automatically.

## Controls

Inside the visualizer window:

- `Left/Right`: switch active hook
- Mouse click on sidebar item: select hook
- `C`: cycle channel for 3D tensors
- `Up/Down`: increase/decrease gamma
- `[ / ]`: decrease/increase `db_min`
- `- / =`: decrease/increase `db_max`
- `R`: resize window to current tensor dimensions
- Sidebar `Play Waveform` button: plays currently viewed waveform once at 44.1 kHz
- `Esc` or window close: exit visualizer

## API

### `VisualizationHook(name: str, gamma: float = 2.2)`

Drop-in module compatible with `torch.nn.Sequential` or manual layer wiring.

- `name`: display name for this hook in the sidebar
- `gamma`: initial gamma value used by the visualizer process

### `VisualizationHook.stop_visualization()`

Stops the background visualizer process and frees shared CUDA resources.

## Notes and limitations

- CUDA tensors are required for visualization writes; CPU tensors are ignored.
- Extremely large tensors are downsampled to fit GPU texture limits.
- Rendering is rate-limited to reduce training overhead.
- Current implementation is designed around desktop OpenGL usage.

## Development

Run tests:

```bash
pytest
```

The package exports:

```python
from musicsep_visualizer import VisualizationHook
```
