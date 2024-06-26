import os
from typing import Tuple, List, Dict, Optional
import json

import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.nn import functional as F
from lightning import LightningModule
import torchaudio

from dataset.feature_extractor import FilterbankFeatures
from dataset.transforms import SpecAugment
from losses.aam_loss import AngularMarginLoss
from utils import compute_eer, compute_mindcf
from optim.lr_scheduler import CosineAnnealing


class SqueezeExcitation(nn.Module):
    def __init__(self,
                 size: int,
                 reduction_rate: int):
        super().__init__()
        assert size % reduction_rate == 0

        self.excitation = nn.Sequential(
            nn.Linear(size, size // reduction_rate, bias=False),
            nn.ReLU(),
            nn.Linear(size // reduction_rate, size, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x C
        squeezed_x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        ratios = self.excitation(squeezed_x).unsqueeze(-1)
        return x * ratios.expand_as(x)


class MegaBlock(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 kernel_size: int,
                 reduction_rate: int,
                 num_blocks: int,
                 drop_out: float
                 ):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.base_blocks = [
                nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, bias=False, padding=padding,
                              groups=hidden_size),
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False),
                    nn.BatchNorm1d(hidden_size, eps=1e-3),
                    nn.ReLU(),

                )
                for _ in range(num_blocks - 1)
            ]
        self.base_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, bias=False, padding=padding,
                              groups=hidden_size),
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False),
                    nn.BatchNorm1d(hidden_size, eps=1e-3),
                )
        )
        self.base_blocks = nn.Sequential(*self.base_blocks)

        self.se = SqueezeExcitation(hidden_size, reduction_rate)

        self.residual_connection = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size, eps=1e-3)
        )

        self.drop_out = drop_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_x = x
        x = self.base_blocks(x)
        x = self.se(x)
        x = F.relu(self.residual_connection(residual_x) + x)
        return x


class AttentiveStatsPooling(nn.Module):
    def __init__(self, size: int, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(size * 3, hidden_size, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size, eps=1e-3),
            nn.Tanh(),
            nn.Conv1d(hidden_size, size, kernel_size=1),
            nn.Softmax(dim=-2)
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3
        feature_size = x.size(2)
        # attention_rate = self.attention(x.transpose(1, 2)).transpose(1, 2)

        means = torch.sum(x, dim=2)
        var = torch.sum(x ** 2, dim=2) - means ** 2
        stds = torch.sqrt(var.clamp(min=self.eps))

        means = means.unsqueeze(2).repeat(1, 1, feature_size)
        stds = stds.unsqueeze(2).repeat(1, 1, feature_size)
        attn = torch.cat([x, means, stds], dim=1)

        # attention statistics
        attn = self.attention(attn)  # attention pass
        attention_rate = F.softmax(attn, dim=2)

        means = torch.sum(attention_rate * x, dim=2)
        var = torch.sum(attention_rate * x ** 2, dim=2) - means ** 2
        stds = torch.sqrt(var.clamp(min=self.eps))

        return torch.cat([means, stds], dim=1).unsqueeze(2)


def verify_embs(original_emb, verify_emb):
    # embeddings = self.get_embeddings(audio)
    # scores = F.cosine_similarity(true_embedding.expand_as(embeddings), embeddings)
    # scores = (scores + 1) / 2
    # similar_segments = (scores > 0.7).mean().item()
    # return similar_segments > 0.7

    original_emb = torch.tensor(original_emb)
    verify_emb = torch.tensor(verify_emb)
    score = torch.nn.functional.cosine_similarity(original_emb, verify_emb, dim=0)
    score = (score + 1) / 2
    return score.item() > 0.7


class TitaNet(LightningModule):
    def __init__(self,
                 mels: int,
                 hidden_size: int,
                 se_reduction_rate: int,
                 encoder_size: int,
                 sub_blocks: int,
                 prolog_kernel_size: int,
                 epilog_kernel_size: int,
                 embedding_size: int,
                 hidden_pooling_size: int,
                 drop_out: float,
                 # just use for training
                 num_classes: int,
                 scale: float,
                 margin: float,
                 # optimizer config
                 optimizer_config: Dict,
                 scheduler_config: Optional[Dict],
                 # spec augmentation config
                 num_time_masks: int = 5,
                 time_width: float = 0.03,
                 num_freq_masks: int = 3,
                 freq_width: int = 4,
                 p: float = 0.5,
                 ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = FilterbankFeatures(
            normalize="per_feature",
            n_window_size=int(0.025 * 16000),
            n_window_stride=int(0.01 * 16000),
            window="hann",
            nfilt=80,
            n_fft=512,
            frame_splicing=1,
            dither=0.00001
        )

        # Prolog Block and Epilog Block do not have dropout
        self.prolog = nn.Sequential(
            nn.Conv1d(mels, mels, kernel_size=prolog_kernel_size, bias=False, padding=1, groups=mels),
            nn.Conv1d(mels, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size, eps=1e-3),
            SqueezeExcitation(hidden_size, se_reduction_rate),
            nn.ReLU(),
        )

        mega_block_kernel_size = [7, 11, 15]
        self.mega_blocks = nn.Sequential(
            *[
                MegaBlock(hidden_size, mega_block_kernel_size[i], se_reduction_rate, sub_blocks, drop_out)
                for i in range(3)
            ]
        )

        padding = (epilog_kernel_size - 1) // 2
        # Prolog Block and Epilog Block do not have dropout
        self.epilog = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=epilog_kernel_size, padding=padding, bias=False, groups=hidden_size),
            nn.Conv1d(hidden_size, encoder_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(encoder_size, eps=1e-3),
            SqueezeExcitation(encoder_size, se_reduction_rate),
            nn.ReLU(),
        )

        self.pooling = AttentiveStatsPooling(encoder_size, hidden_pooling_size)

        self.embedding_linear = nn.Sequential(
            nn.BatchNorm1d(encoder_size * 2, eps=1e-3),
            nn.Conv1d(encoder_size * 2, embedding_size, kernel_size=1)
        )

        self.spec_augment = SpecAugment(num_freq_masks, freq_width, num_time_masks, int(100 * time_width), p)

        # just use for training
        self.loss = AngularMarginLoss(embedding_size, num_classes, scale, margin)

        self.validation_step_scores = []
        self.validation_step_labels = []

        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lengths = x.size(1)
        batch_size = x.size(0)
        lengths = torch.ones(batch_size, device=x.device) * lengths
        x, _ = self.feature_extractor(x, lengths)
        if self.training:
            x = self.spec_augment(x)

        x = self.prolog(x)
        x = self.mega_blocks(x)
        x = self.epilog(x)

        x = self.pooling(x)
        x = self.embedding_linear(x).squeeze(-1)
        x = F.normalize(x, p=2, dim=1)
        return x

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx) -> torch.Tensor:
        inputs, targets = batch
        embeddings = self.forward(inputs)

        loss = self.loss(embeddings, targets)
        self.log('loss', loss.item(), on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, List[int], List[int]], batch_idx) -> None:
        chunks_audios, lengths, labels = batch
        embeddings = self.forward(chunks_audios)
        avg_embeddings_speaker_1 = []
        avg_embeddings_speaker_2 = []
        for i in range(len(lengths) - 1):
            avg_embedding = embeddings[lengths[i]:lengths[i + 1]].mean(dim=0)
            if i % 2 == 0:
                avg_embeddings_speaker_1.append(avg_embedding)
            else:
                avg_embeddings_speaker_2.append(avg_embedding)

        avg_embeddings_speaker_1 = torch.stack(avg_embeddings_speaker_1, dim=0)
        avg_embeddings_speaker_2 = torch.stack(avg_embeddings_speaker_2, dim=0)
        distance = F.cosine_similarity(avg_embeddings_speaker_1, avg_embeddings_speaker_2, dim=-1)

        self.validation_step_scores += distance.tolist()
        self.validation_step_labels += labels

    def on_validation_epoch_end(self) -> None:
        eer = compute_eer(self.validation_step_scores, self.validation_step_labels)
        mindcf = compute_mindcf(self.validation_step_scores, self.validation_step_labels)
        self.log('eer', eer, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('mindcf', mindcf, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.validation_step_scores.clear()
        self.validation_step_labels.clear()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.optimizer_config['name'] == 'SGD':
            self.optimizer_config.pop('name')
            optimizer = torch.optim.SGD(self.parameters(), **self.optimizer_config)
        if self.optimizer_config['name'] == 'AdamW':
            self.optimizer_config.pop('name')
            optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_config)
        elif self.optimizer_config['name'] == 'Adam':
            self.optimizer_config.pop('name')
            optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        else:
            raise ValueError

        if self.scheduler_config is None:
            return optimizer

        interval = self.scheduler_config.pop('interval')
        if self.scheduler_config['name'] == 'OneCycleLR':
            self.scheduler_config.pop('name')
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **self.scheduler_config)
        elif self.scheduler_config['name'] == 'CosineAnnealingLR':
            self.scheduler_config.pop('name')
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.scheduler_config)
        elif self.scheduler_config['name'] == 'CosineAnnealingWarmup':
            self.scheduler_config.pop('name')
            scheduler = CosineAnnealing(optimizer, **self.scheduler_config)
        else:
            raise ValueError
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": interval,
        }
        return [optimizer], [lr_scheduler_config]

    def get_embeddings(self, audio: torch.Tensor) -> torch.Tensor:
        audio, _ = torchaudio.load(audio)
        audio = audio.squeeze()
        length = audio.size(0)
        if length < 8000:
            print(audio.size())
            audio = torch.cat((audio, torch.zeros(8000 - length)))
            length = 8000
        audio = audio[:length // 8000 * 8000]
        length = audio.size(0)
        chunk_audio = torch.stack([audio[8000 * i:8000 * (i + 2)] for i in range(0, length // 8000 - 1)], dim=0)
        embeddings = self.forward(chunk_audio)
        return embeddings

    def get_avg_embedding(self, audio: str) -> torch.Tensor:
        embeddings = self.get_embeddings(audio)
        return embeddings.mean(dim=0).tolist()

    def predict(self, wav_1, wav_2):
        audio_1, _ = torchaudio.load(wav_1)
        audio_1 = audio_1.squeeze()
        audio_2, _ = torchaudio.load(wav_2)
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
        chunk_audio = chunk_audio.cuda()

        embeddings = self.forward(chunk_audio)
        avg_embedding_1 = embeddings[:length_1].mean(dim=0)
        avg_embedding_2 = embeddings[length_1:].mean(dim=0)

        distance = torch.nn.functional.cosine_similarity(avg_embedding_1, avg_embedding_2, dim=0)
        return distance.item()


if __name__ == '__main__':
    model = TitaNet
    # norm_filename = '/home/anhkhoa/workplace/speaker_verification/temp/norm_audio.wav'
    # os.system(f'ffmpeg -y -i /home/anhkhoa/workplace/speaker_verification/temp/3f997cc0fffd53d82398ccc6174a1b5f.wav -ar 16000 -ab 256000 -ac 1 {norm_filename}')
    #
    # model.get_avg_embedding(norm_filename)
    original = torch.rand(192).tolist()
    verify = torch.rand(192).tolist()
    model.verify(original, verify)

