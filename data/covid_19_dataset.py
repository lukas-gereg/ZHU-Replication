import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset



class Covid19Dataset(Dataset):
    @staticmethod
    def default_loader(path: str, load_type: str = "RGB") -> Image:
        """
        Loads image in load_type, default RGB
        """

        return Image.open(path).convert(load_type)

    def __init__(self,
                 path: str,
                 color_channels: int = 3,
                 item_transform: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
                 target_transform: any = None
                 ) -> None:

        self.load_type = "RGB" if color_channels == 3 else "L"
        self.item_transform = item_transform
        self.target_transform = target_transform
        self.data, self.classes = self._init_data(path)


    def _init_data(self, path: str):
        classes = dict()
        items = []

        for idx, class_path in enumerate(["COVID", "Normal"]):
            classes.update({idx: class_path})

            items_path = os.path.join(path, class_path, "images")

            if os.path.exists(items_path):
                item_paths = os.listdir(items_path)
                items.extend([(os.path.join(items_path, item_path), idx) for item_path in item_paths])

        return items, classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, any, str]:
        item_path, label = self.data[idx]
        item = self.default_loader(item_path, load_type=self.load_type)

        if self.item_transform is not None:
            item = self.item_transform(item)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return item, label, item_path