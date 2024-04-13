import math
from typing import List, Tuple, Optional
import random

import torch


import torchaudio


class AudioTransform:
    def __init__(self):
        pass

    def __call__(self, waveform:torch.Tensor):
        pass


class SpeedPerturb(AudioTransform):
    def __init__(self, speed_ratios: List[float]):
        super().__init__()
        self.resamplers = []
        for speed_ratio in speed_ratios:
            self.resamplers.append(torchaudio.transforms.Resample(16000, int(16000 * speed_ratio)))

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        resample = random.choice(self.resamplers)
        waveform = resample(waveform)
        return waveform


class Reverb(AudioTransform):
    def __init__(self, scale_factors: Tuple[float, float], rir_wavs: str):
        super().__init__()
        self.scale_factors = scale_factors
        self.rir_wavs = []
        with open(rir_wavs, 'r') as fin:
            for line in fin:
                self.rir_wavs.append(line.strip())

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        rir_wav = random.choice(self.rir_wavs)
        rir_waveform, _ = torchaudio.load(rir_wav)
        assert waveform.shape[0] == rir_waveform.shape[0] == 1, "only suport mono audios"

        scale = random.uniform(*self.scale_factors)
        if scale != 1:
            rir_waveform = torch.nn.functional.interpolate(
                rir_waveform.unsqueeze(0),
                scale_factor=scale,
                mode="linear",
                align_corners=False,
            ).squeeze(0)

        orig_amplitude = torch.mean(torch.abs(waveform), dim=1, keepdim=True)

        _, direct_index = rir_waveform.abs().max(dim=1, keepdim=True)

        #waveform = torch.from_numpy(waveform)

        zero_length = waveform.size(-1) - rir_waveform.size(-1)
        if zero_length < 0:
            rir_waveform = rir_waveform[..., :zero_length]
            zero_length = 0

        zeros = torch.zeros(rir_waveform.size(0), zero_length, device=rir_waveform.device)
        after_index = rir_waveform[..., direct_index:]
        before_index = rir_waveform[..., :direct_index]
        rir_ = torch.cat((after_index, zeros, before_index), dim=-1)
        f_signal = torch.fft.rfft(waveform)
        f_kernel = torch.fft.rfft(rir_)
        samples = torch.fft.irfft(f_signal * f_kernel, n=waveform.size(-1))

        current_amplitude = torch.mean(torch.abs(samples), dim=1, keepdim=True)
        # Add eps for make sure this does not divide by zero
        samples *= orig_amplitude / (current_amplitude + 1e-14)

        return samples


class Noise(AudioTransform):
    def __init__(self, snrs: List[float], noise_wavs: str):
        super().__init__()
        self.snrs = snrs
        self.noise_wavs = []
        with open(noise_wavs, 'r') as fin:
            for line in fin:
                self.noise_wavs.append(line.strip())

    def __call__(self, waveform: torch.Tensor):
        noise_wav = random.choice(self.noise_wavs)
        noise_waveform = torchaudio.load(noise_wav)
        clean_amp = torch.sqrt(torch.mean(noise_waveform ** 2, dim=1, keepdim=True))

        snr = random.uniform(*self.snrs)
        noise_amp_factor = 1 / (10 ** (snr / 20))
        new_noise_amp = noise_amp_factor * clean_amp
        waveform *= 1 - noise_amp_factor

        noise_amp = torch.sqrt(torch.mean(noise_waveform ** 2, dim=1, keepdim=True))
        noise_waveform *= new_noise_amp / (noise_amp + 1e-14)

        waveform += noise_waveform
        return waveform


class SpecAugment:
    """
    SpecAugment performs three augmentations:
    - time warping of the feature matrix
    - masking of ranges of features (frequency bands)
    - masking of ranges of frames (time)

    The current implementation works with batches, but processes each example separately
    in a loop rather than simultaneously to achieve different augmentation parameters for
    each example.
    """

    def __init__(
            self,
            num_feature_masks: int = 2,
            features_mask_size: int = 27,
            num_frame_masks: int = 10,
            frames_mask_size: int = 100,
            max_frames_mask_fraction: float = 0.15,
            p=0.9,
    ):
        """
        SpecAugment's constructor.

        :param num_feature_masks: how many feature masks should be applied. Set to ``0`` to disable.
        :param features_mask_size: the width of the feature mask (expressed in the number of masked feature bins).
            This is the ``F`` parameter from the SpecAugment paper.
        :param num_frame_masks: the number of masking regions for utterances. Set to ``0`` to disable.
        :param frames_mask_size: the width of the frame (temporal) masks (expressed in the number of masked frames).
            This is the ``T`` parameter from the SpecAugment paper.
        :param max_frames_mask_fraction: limits the size of the frame (temporal) mask to this value times the length
            of the utterance (or supervision segment).
            This is the parameter denoted by ``p`` in the SpecAugment paper.
        :param p: the probability of applying this transform.
            It is different from ``p`` in the SpecAugment paper!
        """
        super().__init__()
        assert 0 <= p <= 1
        assert num_feature_masks >= 0
        assert num_frame_masks >= 0
        assert features_mask_size > 0
        assert frames_mask_size > 0
        self.num_feature_masks = num_feature_masks
        self.features_mask_size = features_mask_size
        self.num_frame_masks = num_frame_masks
        self.frames_mask_size = frames_mask_size
        self.max_frames_mask_fraction = max_frames_mask_fraction
        self.p = p

    def __call__(
            self,
            features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes SpecAugment for a batch of feature matrices.

        Since the batch will usually already be padded, the user can optionally
        provide a ``supervision_segments`` tensor that will be used to apply SpecAugment
        only to selected areas of the input. The format of this input is described below.

        :param features: a batch of feature matrices with shape ``(B, T, F)``.
        :param supervision_segments: an int tensor of shape ``(S, 3)``. ``S`` is the number of
            supervision segments that exist in ``features`` -- there may be either
            less or more than the batch size.
            The second dimension encoder three kinds of information:
            the sequence index of the corresponding feature matrix in `features`,
            the start frame index, and the number of frames for each segment.
        :return: an augmented tensor of shape ``(B, T, F)``.
        """
        for sequence_idx in range(len(features)):
            features[sequence_idx] = self._forward_single(features[sequence_idx])

        return features

    def _forward_single(
            self, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply SpecAugment to a single feature matrix of shape (T, F).
        """
        if random.random() > self.p:
            # Randomly choose whether this transform is applied
            return features
        mean = 0
        # Frequency masking
        features = mask_along_axis_optimized(
            features,
            mask_size=self.features_mask_size,
            mask_times=self.num_feature_masks,
            mask_value=mean,
            axis=2,
        )
        # Time masking
        max_tot_mask_frames = self.max_frames_mask_fraction * features.size(0)
        num_frame_masks = min(
            self.num_frame_masks,
            math.ceil(max_tot_mask_frames / self.frames_mask_size),
        )
        max_mask_frames = min(
            self.frames_mask_size, max_tot_mask_frames // num_frame_masks
        )
        features = mask_along_axis_optimized(
            features,
            mask_size=max_mask_frames,
            mask_times=num_frame_masks,
            mask_value=mean,
            axis=1,
        )

        return features


def mask_along_axis_optimized(
        features: torch.Tensor,
        mask_size: int,
        mask_times: int,
        mask_value: float,
        axis: int,
) -> torch.Tensor:
    """
    Apply Frequency and Time masking along axis.
    Frequency and Time masking as described in the SpecAugment paper.

    :param features: input tensor of shape ``(T, F)``
    :mask_size: the width size for masking.
    :mask_times: the number of masking regions.
    :mask_value: Value to assign to the masked regions.
    :axis: Axis to apply masking on (1 -> time, 2 -> frequency)
    """
    if axis not in [1, 2]:
        raise ValueError("Only Frequency and Time masking are supported!")

    features = features.unsqueeze(0)
    features = features.reshape([-1] + list(features.size()[-2:]))

    values = torch.randint(int(0), int(mask_size), (1, mask_times))
    min_values = torch.rand(1, mask_times) * (features.size(axis) - values)
    mask_starts = (min_values.long()).squeeze()
    mask_ends = (min_values.long() + values.long()).squeeze()

    if axis == 1:
        if mask_times == 1:
            features[:, mask_starts:mask_ends] = mask_value
            return features.squeeze(0)
        for (mask_start, mask_end) in zip(mask_starts, mask_ends):
            features[:, mask_start:mask_end] = mask_value
    else:
        if mask_times == 1:
            features[:, :, mask_starts:mask_ends] = mask_value
            return features.squeeze(0)
        for (mask_start, mask_end) in zip(mask_starts, mask_ends):
            features[:, :, mask_start:mask_end] = mask_value

    features = features.squeeze(0)
    return features
