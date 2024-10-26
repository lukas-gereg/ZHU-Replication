import torch
import wandb
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, matthews_corrcoef, jaccard_score


class Validation:
    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, epoch, validation_loader, device, model, loss, scheduler=None):
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            epoch_validation_loss = 0

            x = torch.tensor([])
            y = torch.tensor([])

            for validation_loader_data, validation_labels, item_name in validation_loader:
                validation_loader_data = validation_loader_data.to(device)
                validation_labels = validation_labels.to(device)

                val_prediction = model(validation_loader_data)

                validation_loss = loss(val_prediction, validation_labels)
                epoch_validation_loss += validation_loss.item()

                if self.debug:
                    print(f"validation item: {item_name}, prediction: {val_prediction.cpu().detach().numpy().tolist()}, "
                          f"class prediction: {val_prediction.cpu().detach().numpy().argmax(-1).tolist()}, ground "
                          f"truth: {validation_labels.cpu().detach().numpy().tolist()}")

                x = torch.cat((x, val_prediction.cpu()))
                y = torch.cat((y, validation_labels.cpu()))

            val_loss = epoch_validation_loss / len(validation_loader)

            if scheduler is not None:
                scheduler.step(val_loss)

            class_report = classification_report(
                y.detach().numpy(),
                x.detach().numpy().argmax(-1),
                labels=list(validation_loader.dataset.classes.keys()),
                target_names=list(validation_loader.dataset.classes.values()),
                output_dict=True,
                zero_division=0.0
            )

            mathews_coeff = matthews_corrcoef(y.detach().numpy(), x.detach().numpy().argmax(-1))
            jacc_score = jaccard_score(y.detach().numpy(), x.detach().numpy().argmax(-1))
            balanced_accuracy = balanced_accuracy_score(y.detach().numpy(), x.detach().numpy().argmax(-1))

            conf_mat = confusion_matrix(
                y.detach().numpy(),
                x.detach().numpy().argmax(-1),
                labels=list(validation_loader.dataset.classes.keys())
            )

            column_names = list(validation_loader.dataset.classes.values())
            conf_mat = conf_mat.tolist()

            print(
                f"Epoch {epoch} validation_report: {dict({'mathews_correlation_coefficient': mathews_coeff, 'jaccard_score': jacc_score, 'balanced_accuracy': balanced_accuracy, **class_report})}, validation_loss: {val_loss}, validation_confusion_matrix: {conf_mat}")

            if wandb.run is not None:
                for index, row in enumerate(conf_mat):
                    row.insert(0, column_names[index])

                wandb_table = wandb.Table(data=conf_mat, columns=["names (real ↓/predicted →)"] + column_names)

                wandb.log({f"validation_report": {'loss': val_loss, 'mathews_correlation_coefficient': mathews_coeff, 'jaccard_score': jacc_score, 'balanced_accuracy': balanced_accuracy, **class_report},
                           f"validation_confusion_matrix": wandb_table,
                           }, step=epoch)

            return val_loss
