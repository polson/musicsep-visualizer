import array
import time
from typing import Optional

import pygame
import torch


class WaveformPlaybackHandler:
    def __init__(self, sample_rate: int = 44_100, debug_mode: bool = False):
        self.sample_rate = sample_rate
        self.debug_mode = debug_mode
        self._latest_waveform: Optional[torch.Tensor] = None
        self._audio_ready = False
        self._audio_init_failed = False
        self._current_sound = None
        self._current_channel_obj = None
        self._playback_started_at: Optional[float] = None
        self._playback_duration_seconds = 0.0
        self._played_waveform_num_samples: Optional[int] = None

    @property
    def is_play_available(self) -> bool:
        return self._latest_waveform is not None

    @property
    def is_playing(self) -> bool:
        if self._current_channel_obj is None:
            return False

        try:
            if not self._current_channel_obj.get_busy():
                self._clear_playback_state()
                return False
        except Exception:
            self._clear_playback_state()
            return False

        if self._playback_progress() >= 1.0:
            self._clear_playback_state()
            return False

        return True

    @staticmethod
    def _is_waveform_tensor_2d(tensor: torch.Tensor) -> bool:
        if tensor.dim() != 2:
            return False

        h, w = tensor.shape
        long_enough = 1024
        return ((h in (1, 2) and w >= long_enough) or
                (w in (1, 2) and h >= long_enough))

    def _extract_waveform_for_playback(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._is_waveform_tensor_2d(tensor):
            return None

        if not torch.is_floating_point(tensor):
            tensor = tensor.float()

        # Normalize orientation to [channels, samples]
        if tensor.shape[0] not in (1, 2) and tensor.shape[1] in (1, 2):
            tensor = tensor.transpose(0, 1)

        if tensor.shape[0] not in (1, 2):
            return None

        return tensor.detach().contiguous().clone()

    def update_from_tensor(self, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None:
            self._latest_waveform = None
            return

        self._latest_waveform = self._extract_waveform_for_playback(tensor)

    def _clear_playback_state(self) -> None:
        self._playback_started_at = None
        self._playback_duration_seconds = 0.0
        self._played_waveform_num_samples = None

    def _playback_progress(self) -> float:
        if self._playback_started_at is None or self._playback_duration_seconds <= 0.0:
            return 0.0

        elapsed = time.perf_counter() - self._playback_started_at
        return max(0.0, min(1.0, elapsed / self._playback_duration_seconds))

    def get_playback_progress(self) -> Optional[float]:
        if not self.is_playing:
            return None
        return self._playback_progress()

    def should_draw_scan_line(self) -> bool:
        if not self.is_playing or self._latest_waveform is None:
            return False

        return self._latest_waveform.shape[-1] == self._played_waveform_num_samples

    def _ensure_audio_initialized(self) -> bool:
        if self._audio_ready:
            return True

        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(
                    frequency=self.sample_rate,
                    size=-16,
                    channels=2,
                )
            self._audio_ready = True
            return True
        except Exception:
            self._audio_init_failed = True
            self._audio_ready = False
            return False

    def play_cached_waveform(self) -> None:
        if self._latest_waveform is None:
            return

        if not self._ensure_audio_initialized():
            return

        waveform = self._latest_waveform
        try:
            if waveform.shape[0] not in (1, 2):
                return

            waveform = waveform.float()
            waveform = torch.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
            waveform = torch.clamp(waveform, -1.0, 1.0)

            # pygame expects interleaved samples for stereo output.
            if waveform.shape[0] == 1:
                mono = torch.round(waveform[0] * 32767.0).to(torch.int16)
                pcm = torch.stack((mono, mono), dim=1).contiguous().cpu()
            else:
                pcm = torch.round(waveform.transpose(0, 1) * 32767.0).to(torch.int16).contiguous().cpu()

            try:
                pcm_bytes = pcm.numpy().tobytes()
            except Exception:
                pcm_bytes = array.array('h', pcm.view(-1).tolist()).tobytes()

            if self._current_channel_obj is not None and self._current_channel_obj.get_busy():
                self._current_channel_obj.stop()
            self._clear_playback_state()

            self._current_sound = pygame.mixer.Sound(buffer=pcm_bytes)
            self._current_channel_obj = self._current_sound.play()
            if self._current_channel_obj is None:
                self._clear_playback_state()
                return

            self._playback_started_at = time.perf_counter()
            self._playback_duration_seconds = waveform.shape[-1] / float(self.sample_rate)
            self._played_waveform_num_samples = waveform.shape[-1]
        except Exception:
            self._clear_playback_state()

    def cleanup(self) -> None:
        try:
            if self._current_channel_obj is not None:
                self._current_channel_obj.stop()
        except Exception:
            pass

        self._current_channel_obj = None
        self._current_sound = None
        self._latest_waveform = None
        self._clear_playback_state()

        try:
            if pygame.mixer.get_init() is not None:
                pygame.mixer.quit()
        except Exception:
            pass

        self._audio_ready = False
