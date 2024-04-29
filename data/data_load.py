import os
from PIL import Image as Image
from .data_augment import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor, PairCenterCrop
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=16, patch_size=64, num_workers=0, use_transform=True, test_every=None):
    image_dir = path
    test_every = test_every

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(patch_size),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        RainDataset(image_dir, transform=transform, test_every=test_every, batch_size=batch_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    #image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        RainDataset(path, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


class RainDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False, test_every=None, batch_size=None):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'rain/'))
        self._check_image(self.image_list)
        self.image_list.sort()

        self.transform = transform
        self.is_test = is_test

        self.test_every = test_every
        self.file_num = len(self.image_list)
        self.batch = batch_size

        if not self.is_test:
            self.repeat = self.test_every // (self.file_num // self.batch)

    def __len__(self):
        if not self.is_test:
            #num = len(self.image_list) * self.repeat
            return self.file_num * self.repeat
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'rain', self.image_list[idx % self.file_num]))
        label = Image.open(os.path.join(self.image_dir, 'norain', self.image_list[idx % self.file_num]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
