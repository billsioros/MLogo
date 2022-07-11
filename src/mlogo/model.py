"""An ML approach to generating logos from text."""
from pathlib import Path

import torch
import torch.nn as nn


class Base(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path: Path):
        torch.save(self.state_dict(), path)

    @classmethod
    def from_device(cls, device: torch.device, *args, **kwargs):
        model = cls(*args, **kwargs)

        model = model.to(device)

        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        return model

    @classmethod
    def from_state(cls, path: Path, device: torch.device, *args, **kwargs):
        map_location = "cuda:%d" % (device.index,) if device.type == "cuda" else "cpu"

        model = cls(*args, **kwargs)

        model.load_state_dict(torch.load(path, map_location=map_location))

        model = model.to(device)

        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        return model


class CNNGenerator(Base):
    def __init__(self, in_channels=100, out_channels=256, training_channels=3):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.training_channels = training_channels

        self.feature_dims = (in_channels, 1, 1)

        # 1, 56, 768
        self.linear = nn.Linear(1 * 56 * 768, 100)

        self.module_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels * 16, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(out_channels * 16),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(out_channels * 16, out_channels * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels * 8),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(out_channels * 8, out_channels * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels * 4),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(out_channels * 4, out_channels * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels * 2),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(out_channels * 2, training_channels, 8, 8, bias=False),
                    nn.BatchNorm2d(training_channels),
                    nn.ReLU(True),
                ),
            ]
        )

    def forward(self, batch):
        batch = self.linear(batch.reshape(batch.size(0), -1)).reshape(batch.size(0), 100, 1, 1)

        for module in self.module_list:
            batch = module(batch)

        return batch


class MLogoDiscriminator(Base):
    def __init__(self, out_channels=3, in_channels=256):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels

        self.module_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, in_channels, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(in_channels * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels * 2, in_channels * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(in_channels * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels * 4, in_channels * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(in_channels * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels * 8, 1, 8, 1, 0, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
            ]
        )

        # 1, 15, 33
        self.linear = nn.Sequential(nn.Linear(81, 1), nn.Sigmoid())

    def forward(self, batch):
        for module in self.module_list:
            batch = module(batch)

        return self.linear(batch.reshape(batch.size(0), -1))
