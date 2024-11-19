import os
import torch
import wandb
import torch.nn as nn
from torchvision import transforms
from sklearn.model_selection import train_test_split


from assignment1.data.custom_subset import CustomSubset
from assignment1.data.covid_19_dataset import Covid19Dataset
from assignment1.utils.cross_validation import CrossValidation
from assignment1.models.combined_vvg_model import CombinedVVGModel


if __name__ == '__main__':
    debug = False
    folds = 5
    random_seed = 42

    scheduler = None
    early_stopping = 25

    BATCH_SIZE = 64
    image_size = (150, 150)
    color_channels = 3

    epochs = 10000
    lr = 0.0003

    item_transform = transforms.Compose([transforms.ToTensor(),
                                         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         transforms.Resize(image_size, antialias=True)])
    base_dataset = Covid19Dataset(os.path.join(".", "COVID-19_Radiography_Dataset"),
                                  color_channels=color_channels,
                                  item_transform=item_transform)

    x, y = zip(*[item for item in base_dataset.data])
    y_ids = [i for i in range(len(base_dataset))]

    train_ids, test_ids, train_y, test_y = train_test_split(y_ids, y, stratify=y, test_size=0.3, random_state=random_seed)
    train_ids, validation_ids = train_test_split(train_ids, stratify=train_y, test_size=0.3, random_state=random_seed)

    train_dataset = CustomSubset(base_dataset, train_ids)
    test_dataset = CustomSubset(base_dataset, test_ids)
    validation_dataset = CustomSubset(base_dataset, validation_ids)

    model_properties = {'color_channels': color_channels, 'image_size': image_size, 'pooling_method_constructor': nn.AdaptiveMaxPool2d}
    model = CombinedVVGModel(model_properties)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    wandb_config = dict(project="First-Experiments", entity="ZHU-assignment-1", config={
        "model properties": model_properties,
        "learning rate": lr,
        "image_transforms": str(item_transform),
        "epochs": epochs,
        "early stopping": early_stopping,
        "model": str(model),
        "optimizer": str(optimizer),
        "loss calculator": str(loss),
        "LR reduce scheduler": str(scheduler),
        "debug": debug,
        "batch_size": 64,
        "random_seed": random_seed
    })

    wandb_login_key = "a9f105e8b3bc98e07700e93201d4b02c1c75106d"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if wandb_login_key is not None:
        wandb.login(key=wandb_login_key)

    CrossValidation(BATCH_SIZE, folds, debug, random_seed)(epochs, device, optimizer, model, loss,
                                                                         train_dataset, validation_dataset, test_dataset,
                                                                         early_stopping, scheduler, wandb_config)

    print(f"training of model complete")