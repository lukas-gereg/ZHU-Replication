from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self):
        self.classes = dict()