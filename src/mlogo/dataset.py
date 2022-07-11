import itertools
import json
import tarfile
from io import BytesIO
from pathlib import Path

import requests
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from PIL import Image
from torch.utils.data.dataset import Dataset


class MLogoDataset(Dataset):
    """MLogo Dataset."""

    TRANSFORM = transforms.Compose(
        [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Resize((256, 256))]
    )

    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        caption, url = self._data[idx]

        image = self.process_image(url)

        return caption, image

    @classmethod
    def from_tar(cls, directory: Path):
        paths = directory.iterdir()
        paths = filter(lambda filename: filename.suffix == '.tar', paths)
        paths = [directory / path for path in paths]

        data = []
        for path in paths[:1]:
            with tarfile.open(path) as tar:
                for metadata in filter(
                    lambda filename: filename.name[-4:] == 'json', tar.getmembers()
                ):
                    file = json.load(tar.extractfile(metadata))
                    url = file['url']
                    caption = file['caption']

                    data.append((caption, url))

        return cls(data)

    @classmethod
    def from_directory(cls, directory: Path):
        paths = directory.iterdir()
        paths = [path for path in paths]

        data = []
        for path in paths:
            with open(path) as file:
                file = json.load(file)
                url = file['url']
                caption = file['caption']

                data.append((caption, url))

        return cls(data)

    @classmethod
    def process_image(cls, url):
        try:
            response = requests.get(url).content
            image = Image.open(BytesIO(response))
            tensor = functional.to_tensor(image)
        except requests.exceptions.ConnectionError:
            tensor = torch.zeros((3, 256, 256))

        return cls.TRANSFORM(tensor)


if __name__ == '__main__':
    dataset = MLogoDataset.from_directory(Path.cwd() / 'data' / 'small')

    for (image, caption) in itertools.islice(dataset, 10):
        print(image, caption)
