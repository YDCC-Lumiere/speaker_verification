import random
from typing import Tuple, List, Optional
import os

import torch
import torchaudio
from torch.utils.data.dataset import Dataset


class SpeakerTrainingDataset(Dataset):
    def __init__(self, wav_file) -> None:
        self.id_wavs = []
        self.id_range = []
        self.durations = []
        current_idx = -1
        with open(wav_file, 'r') as fin:
            for i, line in enumerate(fin):
                idx, wav_path, duration = line.strip().split(',')
                idx, duration = int(idx), int(duration)
                if idx != current_idx:
                    self.id_range.append(i)
                    current_idx = idx
                self.id_wavs.append((idx, wav_path))
                self.durations.append(duration)

        self.id_range.append(len(self.id_wavs))
        assert len(self.id_range) > 2

    def __len__(self) -> int:
        return len(self.id_wavs)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        idx, wav_path = self.id_wavs[item]

        audio, _ = torchaudio.load(wav_path)
        audio = audio.squeeze()
        duration = audio.size(0)
        star = random.randint(0, duration - 16000)
        audio = audio[star:star + 16000]

        return audio, idx


def collect_testing_batch(batch) -> Tuple[torch.Tensor, List[int], List[int]]:
    chunks_audios = [item[0] for item in batch]
    lengths = [0]
    for item in batch:
        lengths.append(lengths[-1] + item[1])
        lengths.append(lengths[-1] + item[2])

    labels = [item[3] for item in batch]

    chunks_audios = torch.cat(chunks_audios, dim=0)
    # labels = torch.LongTensor(labels)
    return chunks_audios, lengths, labels


class SpeakerTestingDataset(Dataset):
    def __init__(self, root_dir: str, wav_file: str):
        super().__init__()
        self.root_dir = root_dir
        self.pair_wavs = []
        self.labels = []
        with open(wav_file, 'r') as fin:
            for line in fin:
                wav_1, wav_2, label = line.strip().split(',')
                self.pair_wavs.append((wav_1, wav_2))
                self.labels.append(int(label))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, item) -> Tuple[torch.Tensor, int, int, int]:
        wav_1, wav_2 = self.pair_wavs[item]
        audio_1, _ = torchaudio.load(os.path.join(self.root_dir, wav_1))
        audio_1 = audio_1.squeeze()
        audio_2, _ = torchaudio.load(os.path.join(self.root_dir, wav_2))
        audio_2 = audio_2.squeeze()

        length_1 = audio_1.size(0)
        audio_1 = audio_1[:length_1 // 8000 * 8000]
        length_1 = audio_1.size(0)
        chunk_audio_1 = torch.stack([audio_1[8000 * i:8000 * (i + 2)] for i in range(0, length_1 // 8000 - 1)],
                                     dim=0)

        length_2 = audio_2.size(0)
        audio_2 = audio_2[:length_2 // 8000 * 8000]
        length_2 = audio_2.size(0)
        chunk_audio_2 = torch.stack([audio_2[8000 * i:8000 * (i + 2)] for i in range(0, length_2 // 8000 - 1)],
                                     dim=0)

        length_1 = chunk_audio_1.size(0)
        length_2 = chunk_audio_2.size(0)
        chunk_audio = torch.cat((chunk_audio_1, chunk_audio_2), dim=0)
        return chunk_audio, length_1, length_2, self.labels[item]
