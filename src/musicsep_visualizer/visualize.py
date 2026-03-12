import signal
import atexit
import signal
import sys
import time
import weakref
import zlib
from typing import Optional

import pygame
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from OpenGL.GL import *
from cuda.bindings import driver as cu
from pygame.locals import *

from .stft import STFT
from .visualizer_ui import Sidebar
from .waveform_playback import WaveformPlaybackHandler

# =============================================================================
# Exception Hook for Crash Cleanup
# =============================================================================
_original_excepthook = sys.excepthook
_cleanup_refs = weakref.WeakSet()


def _cleanup_excepthook(exc_type, exc_val, exc_tb):
    """Custom excepthook that cleans up visualization resources on crash."""
    # Perform cleanup for all registered VisualizationHook classes
    for hook_cls in list(_cleanup_refs):
        try:
            hook_cls.stop_visualization()
        except Exception:
            pass
    # Call original excepthook
    _original_excepthook(exc_type, exc_val, exc_tb)


sys.excepthook = _cleanup_excepthook


# =============================================================================
# Configuration
# =============================================================================

class Config:
    TARGET_FPS = 60
    # Increased to 32M to support 4k tensors (4096*4096 = ~16.8M elements)
    MAX_BUFFER_ELEMENTS = 32_000_000
    RING_SIZE = 3
    # Safe texture limit for most modern GPUs.
    # Tensors larger than this will be downsampled.
    MAX_TEXTURE_DIM = 8192
    # Set to True to see operational logs (setup, discovery, stats)
    DEBUG_MODE = False


# =============================================================================
# Multiprocessing Context
# =============================================================================
ctx = mp.get_context('spawn')


# =============================================================================
# CUDA Helper Functions
# =============================================================================

def check_cuda_error(result):
    if isinstance(result, tuple):
        err = result[0] if len(result) > 0 else result
    else:
        err = result
    if err != cu.CUresult.CUDA_SUCCESS:
        err_name = ""
        try:
            name_res = cu.cuGetErrorName(err)
            if name_res[0] == cu.CUresult.CUDA_SUCCESS:
                err_name = f" ({name_res[1].decode_codes('utf-8')})"
        except:
            pass
        raise RuntimeError(f"CUDA Error Code: {err.value}{err_name}")
    return result


def initialize_cuda():
    check_cuda_error(cu.cuInit(0))
    device = cu.cuCtxGetDevice()
    if isinstance(device, tuple):
        return device[1]
    return 0


# =============================================================================
# Shared Memory Architecture
# =============================================================================

class SharedRingBuffer:
    # Max hook names we can track (avoids Queue which uses semaphores)
    MAX_HOOKS = 64
    MAX_NAME_LEN = 128

    def __init__(self):
        self.buffer = torch.zeros(
            (Config.RING_SIZE, Config.MAX_BUFFER_ELEMENTS),
            dtype=torch.float32,
            device='cuda'
        ).share_memory_()

        self.meta_buffer = ctx.Array('i', Config.RING_SIZE * 8)
        self.write_index = ctx.Value('i', 0)
        self.active_name_hash = ctx.Value('q', 0)

        # Replace Queue with shared arrays (no semaphores!)
        # Store hook names as bytes in shared memory
        self.discovery_names = ctx.Array('c', self.MAX_HOOKS * self.MAX_NAME_LEN)
        self.discovery_count = ctx.Value('i', 0)

    def close(self):
        """Release CUDA shared memory."""
        if hasattr(self, 'buffer'):
            del self.buffer
        self.buffer = None

    def announce_hook(self, name: str):
        """Add a hook name to the shared discovery list (replaces Queue.put)."""
        idx = self.discovery_count.value
        if idx >= self.MAX_HOOKS:
            return  # Full, ignore

        # Encode name and store in shared array
        name_bytes = name.encode('utf-8')[:self.MAX_NAME_LEN - 1]
        start = idx * self.MAX_NAME_LEN
        for i in range(len(name_bytes)):
            self.discovery_names[start + i] = name_bytes[i:i+1]  # Single byte
        self.discovery_names[start + len(name_bytes)] = b'\x00'  # Null terminate

        self.discovery_count.value = idx + 1

    def get_discovered_hooks(self) -> list[str]:
        """Read all discovered hook names (replaces Queue.get)."""
        names = []
        count = self.discovery_count.value
        for idx in range(count):
            start = idx * self.MAX_NAME_LEN
            name_bytes = b''
            for i in range(self.MAX_NAME_LEN):
                b = self.discovery_names[start + i]
                if b == b'\x00':
                    break
                name_bytes += b
            if name_bytes:
                names.append(name_bytes.decode('utf-8'))
        return names

    def write(self, tensor: torch.Tensor, name_hash: int):
        target_hash = self.active_name_hash.value
        if name_hash != target_hash:
            return

        flat = tensor.detach().flatten()
        numel = flat.numel()

        if numel > Config.MAX_BUFFER_ELEMENTS:
            if Config.DEBUG_MODE:
                print(f"[Ring] Tensor too large to write: {numel} > {Config.MAX_BUFFER_ELEMENTS}")
            return

        current_idx = (self.write_index.value + 1) % Config.RING_SIZE

        if not flat.is_cuda:
            return

        self.buffer[current_idx, :numel].copy_(flat)

        shape = tensor.shape
        start_meta = current_idx * 8
        for i in range(8):
            self.meta_buffer[start_meta + i] = 0
        for i, dim in enumerate(shape[:8]):
            self.meta_buffer[start_meta + i] = dim

        # Ensure GPU copy is complete before publishing the index
        torch.cuda.synchronize()

        self.write_index.value = current_idx

    def read_latest(self) -> Optional[torch.Tensor]:
        idx = self.write_index.value
        start_meta = idx * 8
        shape = []
        for i in range(8):
            val = self.meta_buffer[start_meta + i]
            if val == 0: break
            shape.append(val)

        if not shape: return None
        numel = 1
        for s in shape: numel *= s

        return self.buffer[idx, :numel].view(*shape)

    def set_active_hook(self, name_str: str):
        h = zlib.adler32(name_str.encode('utf-8')) & 0x7FFFFFFF
        if Config.DEBUG_MODE:
            print(f"[Vis] Setting Active Hook: '{name_str}'")
        self.active_name_hash.value = h


# =============================================================================
# Visualization Hook
# =============================================================================

class VisualizationHook(torch.nn.Identity):
    _shared_ring: Optional[SharedRingBuffer] = None
    _process = None
    _lock = ctx.RLock()
    _gamma = 2.2  # Class-level default, set by first hook
    _warmup_counter = 0
    _WARMUP_FRAMES = 10  # Wait this many forward passes before starting

    def __init__(self, name: str, gamma: float = 2.2):
        super().__init__()
        self.name = name
        self.name_hash = zlib.adler32(name.encode('utf-8')) & 0x7FFFFFFF
        self._sent_discovery = False
        self.gamma = gamma
        self.last_write_time = 0

    @classmethod
    def _ensure_initialized(cls, gamma: float):
        """Lazily initialize shared ring buffer and visualizer process."""
        if cls._shared_ring is not None:
            return True

        # Count warmup frames before starting (avoids crash noise)
        cls._warmup_counter += 1
        if cls._warmup_counter < cls._WARMUP_FRAMES:
            return False

        with cls._lock:
            # Double-check after acquiring lock
            if cls._shared_ring is not None:
                return True

            cls._gamma = gamma
            cls._shared_ring = SharedRingBuffer()
            if Config.DEBUG_MODE:
                print("[Hook] Starting Visualizer Process...")
            cls._process = ctx.Process(
                target=start_visualizer,
                args=(cls._shared_ring, cls._gamma),
                daemon=True
            )
            cls._process.start()
            atexit.register(cls.stop_visualization)
            _cleanup_refs.add(cls)
        return True

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # Rate limit to ~30 FPS to prevent training slowdown
        now = time.time()
        if now - self.last_write_time < 0.033:
            return tensor
        self.last_write_time = now

        # Lazy init after warmup period
        if not VisualizationHook._ensure_initialized(self.gamma):
            return tensor

        # Check if visualizer process crashed and clean up if so
        if VisualizationHook._process is not None and not VisualizationHook._process.is_alive():
            if Config.DEBUG_MODE:
                print("[Hook] Visualizer process died, cleaning up CUDA memory...")
            VisualizationHook.stop_visualization()
            return tensor

        if VisualizationHook._shared_ring:
            if not self._sent_discovery:
                if Config.DEBUG_MODE:
                    print(f"[Hook] Announcing: {self.name}")
                VisualizationHook._shared_ring.announce_hook(self.name)
                self._sent_discovery = True

            # Write attempts
            VisualizationHook._shared_ring.write(tensor, self.name_hash)

        return tensor

    @classmethod
    def stop_visualization(cls):
        if cls._process and cls._process.is_alive():
            # Try graceful termination first (SIGTERM)
            cls._process.terminate()
            cls._process.join(timeout=5)
            
            # If still alive, force kill
            if cls._process.is_alive():
                if Config.DEBUG_MODE:
                    print("[Hook] Visualizer hung, forcing kill...")
                cls._process.kill()
                cls._process.join(timeout=1)

        cls._process = None
        # Release the shared CUDA memory and multiprocessing resources
        if cls._shared_ring is not None:
            cls._shared_ring.close()
            cls._shared_ring = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass  # May fail during interpreter shutdown


# =============================================================================
# Visualizer Engine
# =============================================================================

class TensorVisualizer:
    def __init__(self, shared_ring: SharedRingBuffer, initial_gamma: float):
        self.ring = shared_ring
        self.gamma = initial_gamma
        self.db_min = -115.0
        self.db_max = 0.0
        self.known_names = []
        self.active_hook_index = 0
        self.render_buffer_indices = None
        self.render_buffer_float = None
        self.inferno_cmap = None
        self.clock = None
        self.pbo = None
        self.texture = None
        self.cuda_res = None
        self.current_shape = None
        self.display_size = (800, 600)
        self.gl_ready = False
        self.frame_count = 0
        self.current_channel = 0  # State for channel cycling
        self._should_exit = False
        self.sidebar = Sidebar()
        self._stft_n_fft = 2048
        self._stft_hop_length = 512
        self._stft_win_length = 2048
        self._stft_center = True
        self._stft_converter: Optional[STFT] = None
        self._waveform_playback = WaveformPlaybackHandler(sample_rate=44_100, debug_mode=Config.DEBUG_MODE)
        
        # Stats
        self.current_min = 0.0
        self.current_max = 0.0
        self.current_mean = 0.0
        self.current_std = 0.0

    def _is_waveform_tensor_2d(self, tensor: torch.Tensor) -> bool:
        if tensor.dim() != 2:
            return False

        h, w = tensor.shape
        long_enough = 1024

        return ((h in (1, 2) and w >= long_enough) or
                (w in (1, 2) and h >= long_enough))

    def _waveform_to_spectrogram(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            if tensor.dim() != 2:
                return None

            if not torch.is_floating_point(tensor):
                tensor = tensor.float()

            if not tensor.is_cuda:
                tensor = tensor.cuda(non_blocking=True)

            # Normalize to [channels, samples]
            if tensor.shape[0] not in (1, 2) and tensor.shape[1] in (1, 2):
                tensor = tensor.transpose(0, 1)

            if tensor.shape[0] not in (1, 2):
                return None

            tensor = tensor.contiguous()
            if self._stft_converter is None:
                self._stft_converter = STFT(
                    n_fft=self._stft_n_fft,
                    hop_length=self._stft_hop_length,
                    normalized=True,
                )

            converter = self._stft_converter.to(device=tensor.device)
            packed = converter(tensor.unsqueeze(0))
            spec_mag = STFT.to_magnitude(packed).squeeze(0)

            if spec_mag.numel() == 0 or not torch.isfinite(spec_mag).all():
                return None

            return spec_mag
        except Exception as e:
            if Config.DEBUG_MODE:
                print(f"[Vis] STFT conversion failed: {e}")
            return None

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully."""
        if Config.DEBUG_MODE:
            print(f"[Vis] Received signal {signum}, shutting down...")
        self._should_exit = True
        raise InterruptedError("Termination Signal")

    def _init_colormaps(self):
        float_cmap = torch.tensor([
            [0.001, 0.000, 0.013], [0.042, 0.028, 0.141], [0.122, 0.047, 0.281],
            [0.217, 0.036, 0.383], [0.328, 0.057, 0.427], [0.472, 0.110, 0.428],
            [0.621, 0.164, 0.388], [0.735, 0.215, 0.330], [0.823, 0.275, 0.266],
            [0.885, 0.342, 0.202], [0.923, 0.399, 0.155], [0.949, 0.455, 0.110],
            [0.974, 0.536, 0.048], [0.985, 0.608, 0.024], [0.987, 0.652, 0.045],
            [0.982, 0.751, 0.147], [0.966, 0.836, 0.261], [0.946, 0.930, 0.442],
            [0.957, 0.971, 0.556], [0.988, 0.998, 0.644]
        ], dtype=torch.float32, device='cuda')
        self.inferno_cmap = (float_cmap * 255).byte()

    def _downsample_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Safeguard: Downsample huge tensors to fit within GPU texture limits.
        """
        h, w = tensor.shape
        if h > Config.MAX_TEXTURE_DIM or w > Config.MAX_TEXTURE_DIM:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            new_h = min(h, Config.MAX_TEXTURE_DIM)
            new_w = min(w, Config.MAX_TEXTURE_DIM)
            tensor = F.interpolate(
                tensor, size=(new_h, new_w), mode='bilinear', align_corners=False
            )
            tensor = tensor.squeeze(0).squeeze(0)
        return tensor

    def _normalize_tensor(self, tensor: torch.Tensor):
        """
        Applies Spek-like Logarithmic Scaling (dB) for visual intensity:
        1. Magnitude: abs(tensor)
        2. Log Scaling: dB = 20 * log10(magnitude + eps)
        3. Normalization: Map [db_min, db_max] to [0, 1]
        4. Gamma (Optional): Extra contrast adjustment
        """
        if tensor.numel() == 0:
            return

        # 1. Take Magnitude (Absolute Value)
        # This handles signed data gracefully by focusing on intensity
        torch.abs(tensor, out=self.render_buffer_float)

        # 2. Convert to Decibels
        # Formula: dB = 20 * log10(magnitude + eps)
        # 1e-7 provides a floor around -140dB, safe for most audio
        eps = 1e-7
        self.render_buffer_float.add_(eps)
        torch.log10(self.render_buffer_float, out=self.render_buffer_float)
        self.render_buffer_float.mul_(20.0)

        # 3. Spek-style Clamping and Normalization
        # level = (dB - db_min) / (db_max - db_min)
        db_range = self.db_max - self.db_min
        if db_range == 0: db_range = 1e-6
        
        torch.sub(self.render_buffer_float, self.db_min, out=self.render_buffer_float)
        torch.div(self.render_buffer_float, db_range, out=self.render_buffer_float)

        # Safety: Remove NaNs/Infs
        torch.nan_to_num_(self.render_buffer_float, nan=0.0, posinf=1.0, neginf=0.0)

        # 4. Final clamping to [0, 1] range
        torch.clamp(self.render_buffer_float, 0.0, 1.0, out=self.render_buffer_float)

        # 5. Optional Gamma (Secondary contrast tweak)
        if self.gamma != 1.0:
            torch.pow(self.render_buffer_float, self.gamma, out=self.render_buffer_float)

        # Scale to Colormap Index (0-19)
        torch.mul(self.render_buffer_float, 19.0, out=self.render_buffer_float)
        self.render_buffer_float.clamp_(0.0, 19.0)
        self.render_buffer_indices.copy_(self.render_buffer_float)


    def run(self):
        # Register signal handlers for graceful termination
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        try:
            initialize_cuda()
            self._init_colormaps()
            pygame.init()

            info = pygame.display.Info()
            self.display_size = (info.current_w // 2, info.current_h // 2)

            pygame.display.set_mode(self.display_size, OPENGL | DOUBLEBUF | RESIZABLE)
            pygame.display.set_caption("Waiting for Tensors...")
            self.clock = pygame.time.Clock()

            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            self._setup_gl()

            last_name_check = 0

            while not self._should_exit:
                try:
                    if not self._handle_input():
                        break

                    now = time.time()
                    if now - last_name_check > 1.0:
                        # Poll shared memory for new hook names (no Queue/semaphores)
                        discovered = self.ring.get_discovered_hooks()
                        for name in discovered:
                            if name not in self.known_names:
                                self.known_names.append(name)
                                if Config.DEBUG_MODE:
                                    print(f"[Vis] Discovered: {name}")
                                if len(self.known_names) == 1:
                                    self.ring.set_active_hook(name)
                        last_name_check = now

                    glClearColor(0.0, 0.0, 0.0, 1.0)
                    glClear(GL_COLOR_BUFFER_BIT)
                    
                    # Draw Sidebar first
                    sidebar_width = self.sidebar.draw(self.display_size)
                    
                    channel_idx_used = 0
                    num_channels = 0

                    # Wrap tensor processing in try-except to prevent crashes
                    # from malformed/unexpected tensor data
                    try:
                        tensor = self.ring.read_latest()

                        if tensor is not None:
                            if tensor.dim() == 4: tensor = tensor[0]

                            self._waveform_playback.update_from_tensor(tensor)

                            if self._is_waveform_tensor_2d(tensor):
                                spectrogram = self._waveform_to_spectrogram(tensor)
                                if spectrogram is None:
                                    tensor = None
                                else:
                                    tensor = spectrogram

                            if tensor is not None and tensor.dim() == 3:
                                num_channels = tensor.shape[0]
                                channel_idx_used = self.current_channel % num_channels
                                tensor = tensor[channel_idx_used]

                            if tensor is not None and tensor.dim() == 2:
                                tensor = self._downsample_tensor(tensor)
                                self._render_tensor(tensor, sidebar_width)
                        else:
                            self._waveform_playback.update_from_tensor(None)
                    except Exception as e:
                        # Log but don't crash - just skip this frame
                        print(f"[Vis] Frame error (skipping): {e}")

                    # Update sidebar text (done after render to get latest stats/channels)
                    active_name = self.known_names[self.active_hook_index] if self.known_names else "Waiting..."
                    status_str = f"Hook: {active_name}\n"
                    if self.current_shape:
                        status_str += f"Shape: {self.current_shape}\n"
                        status_str += f"Min: {self.current_min:.4f}\n"
                        status_str += f"Max: {self.current_max:.4f}\n"
                        status_str += f"Mean: {self.current_mean:.4f}\n"
                        status_str += f"Std: {self.current_std:.4f}\n"

                    status_str += f"FPS: {self.clock.get_fps():.1f}\n"
                    status_str += f"Gamma: {self.gamma:.2f}\n"
                    status_str += f"dB Range: [{self.db_min:.0f}, {self.db_max:.0f}]\n"
                    status_str += f"Ch: {channel_idx_used}"
                    self.sidebar.update_content(status_str)
                    self.sidebar.set_hooks(self.known_names, self.active_hook_index)
                    self.sidebar.set_play_button_visible(self._waveform_playback.is_play_available)

                    # Pygame Caption (Keep minimal or remove since we have sidebar)
                    pygame.display.set_caption(f"Visualizer | {active_name}")

                    pygame.display.flip()
                    self.clock.tick(Config.TARGET_FPS)
                    self.frame_count += 1
                except InterruptedError:
                    break

        except Exception as e:
            if isinstance(e, InterruptedError):
                pass  # Graceful exit
            else:
                print(f"VISUALIZER CRASH: {e}")
                import traceback
                traceback.print_exc()
        finally:
            self._waveform_playback.cleanup()

            # Clean up CUDA-GL interop resources before pygame
            self._cleanup_gl()
            
            # Explicitly release shared tensor reference
            if hasattr(self, 'ring') and self.ring is not None:
                 self.ring.close()
                 self.ring = None
                 
            pygame.quit()

    def _render_tensor(self, tensor, offset_x=0):

        if tensor.numel() == 0:
            return

        h, w = tensor.shape
        self.current_shape = (h, w)
        self.current_min = float(tensor.min())
        self.current_max = float(tensor.max())
        self.current_mean = float(tensor.mean())
        self.current_std = float(tensor.std())

        if self.render_buffer_float is None or self.render_buffer_float.shape != tensor.shape:
            self.render_buffer_float = torch.empty_like(tensor)
            self.render_buffer_indices = torch.empty((h, w), dtype=torch.long, device='cuda')

        self._normalize_tensor(tensor)

        pixels = self.inferno_cmap[self.render_buffer_indices]
        pixels = pixels.contiguous()

        if Config.DEBUG_MODE and self.frame_count % 120 == 0:
            print(f"[Vis] Displaying: {self.known_names[self.active_hook_index]} | Shape: {tensor.shape}")

        torch.cuda.synchronize()
        self._cuda_copy_to_pbo(pixels)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        self._draw_quad(w, h, offset_x)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _draw_quad(self, tex_w, tex_h, offset_x=0):
        win_w, win_h = self.display_size
        
        # Available width for the tensor view
        avail_w = win_w - offset_x
        if avail_w <= 0: return

        # Calculate scale to fit in the available space (keeping aspect ratio)
        scale = min(avail_w / tex_w, win_h / tex_h)
        dw, dh = tex_w * scale, tex_h * scale
        
        # Center in the available space
        # Relative x in available space
        rel_x = (avail_w - dw) / 2
        rel_y = (win_h - dh) / 2
        
        # Absolute x
        abs_x = offset_x + rel_x
        
        # Convert to NDC [-1, 1]
        x1 = 2 * abs_x / win_w - 1
        y1 = 1 - 2 * rel_y / win_h
        x2 = 2 * (abs_x + dw) / win_w - 1
        y2 = 1 - 2 * (rel_y + dh) / win_h

        # Calculate UV coordinates based on actual size vs max texture size
        u_max = tex_w / Config.MAX_TEXTURE_DIM
        v_max = tex_h / Config.MAX_TEXTURE_DIM

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0);
        glVertex2f(x1, y2)
        glTexCoord2f(u_max, 0);
        glVertex2f(x2, y2)
        glTexCoord2f(u_max, v_max);
        glVertex2f(x2, y1)
        glTexCoord2f(0, v_max);
        glVertex2f(x1, y1)
        glEnd()

    def _cuda_copy_to_pbo(self, pixels):
        if not self.cuda_res or not self.gl_ready:
            return

        try:
            check_cuda_error(cu.cuGraphicsMapResources(1, self.cuda_res, cu.CUstream(0)))
            try:
                _, ptr, size = check_cuda_error(cu.cuGraphicsResourceGetMappedPointer(self.cuda_res))
                expected_bytes = pixels.numel()
                check_cuda_error(cu.cuMemcpyDtoD(ptr, pixels.data_ptr(), expected_bytes))
            finally:
                cu.cuGraphicsUnmapResources(1, self.cuda_res, cu.CUstream(0))
        except Exception as e:
            # Always print CUDA errors - these are critical to diagnose
            print(f"[Vis] CUDA copy error (skipping frame): {e}")

    def _cleanup_gl(self):
        """Release CUDA-GL interop resources to prevent memory leaks."""
        # Unregister CUDA resource FIRST (before deleting GL objects)
        if self.cuda_res:
            try:
                cu.cuGraphicsUnregisterResource(self.cuda_res)
            except Exception as e:
                if Config.DEBUG_MODE:
                    print(f"[Vis] Warning: Failed to unregister CUDA resource: {e}")
            self.cuda_res = None

        # Then delete GL objects
        if self.pbo:
            try:
                glDeleteBuffers(1, [self.pbo])
            except Exception:
                pass
            self.pbo = None

        if self.texture:
            try:
                glDeleteTextures([self.texture])
            except Exception:
                pass
            self.texture = None

        # Clear render buffers to free CUDA memory
        # Note: Don't clear inferno_cmap - it's set once in _init_colormaps
        self.render_buffer_float = None
        self.render_buffer_indices = None
        
        if self.sidebar:
            self.sidebar.cleanup()

        # Force CUDA to release memory
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        self.gl_ready = False

    def _setup_gl(self):
        # Clean up any existing resources first
        self._cleanup_gl()

        max_dim = Config.MAX_TEXTURE_DIM

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, max_dim, max_dim, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

        self.pbo = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, max_dim * max_dim * 3, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        _, self.cuda_res = check_cuda_error(cu.cuGraphicsGLRegisterBuffer(
            self.pbo,
            cu.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD
        ))

        self.gl_ready = True

    def _handle_input(self):
        for event in pygame.event.get():
            if event.type == QUIT: return False
            if event.type == MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click
                    x, y = event.pos
                    if x < self.sidebar.width_px:
                        action = self.sidebar.handle_click(x, y)
                        if action is None:
                            continue

                        action_type, action_value = action
                        if action_type == "select_hook" and action_value is not None and action_value < len(self.known_names):
                            self.active_hook_index = action_value
                            self.ring.set_active_hook(self.known_names[self.active_hook_index])
                            self.current_channel = 0
                        if action_type == "play_waveform":
                            self._waveform_playback.play_cached_waveform()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE: return False
                if event.key == K_c: self.current_channel += 1
                if event.key == K_RIGHT and self.known_names:
                    self.active_hook_index = (self.active_hook_index + 1) % len(self.known_names)
                    self.ring.set_active_hook(self.known_names[self.active_hook_index])
                    self.current_channel = 0
                if event.key == K_LEFT and self.known_names:
                    self.active_hook_index = (self.active_hook_index - 1) % len(self.known_names)
                    self.ring.set_active_hook(self.known_names[self.active_hook_index])
                    self.current_channel = 0
                if event.key == K_UP: self.gamma += 0.1
                if event.key == K_DOWN: self.gamma = max(0.1, self.gamma - 0.1)
                # Adjust dB Range
                if event.key == K_LEFTBRACKET: self.db_min -= 5
                if event.key == K_RIGHTBRACKET: self.db_min += 5
                if event.key == K_MINUS: self.db_max -= 5
                if event.key == K_EQUALS: self.db_max += 5
                
                if event.key == K_r and self.current_shape:
                    h, w = self.current_shape
                    new_w = w + self.sidebar.width_px
                    new_h = h
                    self.display_size = (new_w, new_h)
                    pygame.display.set_mode(self.display_size, OPENGL | DOUBLEBUF | RESIZABLE)
                    glViewport(0, 0, new_w, new_h)
            if event.type == VIDEORESIZE:
                self.display_size = (event.w, event.h)
                pygame.display.set_mode(self.display_size, OPENGL | DOUBLEBUF | RESIZABLE)
                glViewport(0, 0, event.w, event.h)
        return True


def start_visualizer(ring, gamma):
    # Ensure this process dies if the parent process dies (Linux only)
    try:
        import ctypes
        import signal
        libc = ctypes.CDLL("libc.so.6")
        # PR_SET_PDEATHSIG = 1. SIGTERM (15) allows graceful cleanup via handlers.
        libc.prctl(1, signal.SIGTERM)
    except Exception:
        pass

    viz = TensorVisualizer(ring, gamma)
    viz.run()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(0)

    mp.set_start_method('spawn', force=True)

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        VisualizationHook("Layer 1 (Pre-ReLU)"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, 3, padding=1),
        VisualizationHook("Layer 2 (Conv Output)")
    ).cuda()

    print("Running... Use LEFT/RIGHT keys to switch layers.")

    x = torch.randn(1, 3, 512, 512).cuda()
    try:
        while True:
            model(x)
            time.sleep(0.016)
            x = torch.roll(x, 1, dims=2)
    except KeyboardInterrupt:
        VisualizationHook.stop_visualization()
