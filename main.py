import os
import wandb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.utils.data as torch_utils
from sklearn.model_selection import train_test_split


from assignment1.data.covid_19_dataset import Covid19Dataset
from assignment1.utils.training import Training
from assignment1.utils.evaluation import Evaluation
from assignment1.models.combined_vvg_model import CombinedVVGModel


if __name__ == '__main__':
    debug = False

    wandb_config = None
    wandb_login_key = None

    scheduler = None
    early_stopping = None

    batch_size = 64
    image_size = (150, 150)
    color_channels = 3

    epochs = 25
    lr = 0.0003

    item_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(image_size, antialias=True)])
    base_dataset = Covid19Dataset("C:\\Škola\\TUKE\\ING\\ZHU\\ZHU_PROGRAMS\\assignment1\\archive\\COVID-19_Radiography_Dataset"
                                  ,color_channels=color_channels,
                                  item_transform=item_transform)

    x, y = zip(*[item for item in base_dataset.data])
    y_ids = [i for i in range(len(base_dataset))]

    train_ids, test_ids, train_y, test_y = train_test_split(y_ids, y, stratify=y, test_size=0.3)
    train_ids, validation_ids = train_test_split(train_ids, stratify=train_y, test_size=0.2)

    train_sub_sampler = torch_utils.SubsetRandomSampler(train_ids)
    test_sub_sampler = torch_utils.SubsetRandomSampler(test_ids)
    validation_sub_sampler = torch_utils.SubsetRandomSampler(validation_ids)

    train_loader = torch_utils.DataLoader(base_dataset, batch_size=batch_size, sampler=train_sub_sampler)
    test_loader = torch_utils.DataLoader(base_dataset, batch_size=batch_size, sampler=test_sub_sampler)
    validation_loader = torch_utils.DataLoader(base_dataset, batch_size=batch_size, sampler=validation_sub_sampler)

    model = CombinedVVGModel(color_channels, image_size, nn.AdaptiveAvgPool2d)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("model_params", exist_ok=True)

    if wandb_login_key is not None and wandb_config is not None:
        wandb.init(**wandb_config)

    losses = Training(debug)(epochs, device, optimizer, model, loss, train_loader, validation_loader,
                             early_stopping, scheduler)

    total_loss, results = Evaluation(debug)(loss, test_loader, model, device)

    wandb.finish()

    print(f'results: {results}')
    print(f'Loss per item in test: {total_loss}')

    plt.title('Loss Graph')
    plt.xlabel('Epochs')
    plt.ylabel('Loss per item')

    plt.plot(losses)

    plt.show()

    print(f"training of model complete")