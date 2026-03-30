import array
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

    @property
    def is_play_available(self) -> bool:
        return self._latest_waveform is not None

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
        except Exception as e:
            if self.debug_mode and not self._audio_init_failed:
                print(f"[Vis] Audio init failed: {e}")
            self._audio_init_failed = True
            self._audio_ready = False
            return False

    def play_cached_waveform(self) -> None:
        if self._latest_waveform is None:
            print("[Vis] play_cached_waveform: No waveform available")
            return

        if not self._ensure_audio_initialized():
            print("[Vis] play_cached_waveform: Audio init failed")
            return

        print(f"[Vis] play_cached_waveform: Playing waveform with shape {self._latest_waveform.shape}")

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

            self._current_sound = pygame.mixer.Sound(buffer=pcm_bytes)
            self._current_channel_obj = self._current_sound.play()
        except Exception as e:
            if self.debug_mode:
                print(f"[Vis] Audio playback failed: {e}")

    def cleanup(self) -> None:
        try:
            if self._current_channel_obj is not None:
                self._current_channel_obj.stop()
        except Exception:
            pass

        self._current_channel_obj = None
        self._current_sound = None
        self._latest_waveform = None

        try:
            if pygame.mixer.get_init() is not None:
                pygame.mixer.quit()
        except Exception:
            pass

        self._audio_ready = False
