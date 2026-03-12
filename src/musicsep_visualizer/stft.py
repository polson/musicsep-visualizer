import torch
import torch.nn as nn


class STFT(nn.Module):
    def __init__(self, n_fft: int = 2048, hop_length: int = 512, normalized: bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.normalized = normalized
        self.register_buffer("window", torch.hann_window(self.win_length))
        self._length: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        self._length = t
        x = x.float().reshape(b * c, t)

        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            normalized=self.normalized,
            return_complex=True,
        )

        x[:, 0, :].imag = x[:, -1, :].real
        x = x[:, :-1, :]

        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2).contiguous().reshape(b, c * 2, x.shape[1], x.shape[2])
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        b, c_times_two, f, t_prime = x.shape
        c = c_times_two // 2

        x = x.float().reshape(b, c, 2, f, t_prime).permute(0, 1, 3, 4, 2).contiguous()
        x = x.reshape(b * c, f, t_prime, 2)
        x = torch.view_as_complex(x)

        nyquist_real = x[:, 0, :].imag.clone()
        x[:, 0, :].imag = 0.0

        nyquist_bin = torch.zeros_like(x[:, :1, :])
        nyquist_bin.real = nyquist_real.unsqueeze(1)
        x = torch.cat([x, nyquist_bin], dim=1)

        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            normalized=self.normalized,
            length=self._length,
        )
        return x.reshape(b, c, x.shape[-1])

    @staticmethod
    def to_real(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = torch.view_as_real(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = torch.view_as_real(x)
            b, c, f, t, two = x.shape
            x = x.reshape(b, c, f, t, two).permute(0, 1, 4, 2, 3).contiguous().reshape(b, c * 2, f, t)
        return x

    @staticmethod
    def to_complex(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.shape[1] == 2:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = torch.view_as_complex(x)
        else:
            c = x.shape[1] // 2
            b, _, f, t = x.shape
            x = x.reshape(b, c, 2, f, t).permute(0, 1, 3, 4, 2).contiguous()
            x = torch.view_as_complex(x)
        return x

    @staticmethod
    def to_polar(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not x.is_complex():
            x = STFT.to_complex(x)
        return torch.abs(x), torch.angle(x)

    @staticmethod
    def to_magnitude(x: torch.Tensor) -> torch.Tensor:
        magnitude, _ = STFT.to_polar(x)
        return magnitude

    @staticmethod
    def from_polar(magnitude: torch.Tensor, phase: torch.Tensor, return_real: bool = True) -> torch.Tensor:
        x = magnitude * torch.exp(1j * phase)
        if return_real:
            x = STFT.to_real(x)
        return x
