import torch
from segmentation_models_pytorch import utils


def train_model(model, train_dataloader, valid_dataloader, loss, metrics, optimizer, device, epochs, checkpoint_path):
    train_epoch = utils.train.TrainEpoch(
        model, loss=loss, metrics=metrics, optimizer=optimizer, device=device, verbose=True
    )
    valid_epoch = utils.train.ValidEpoch(
        model, loss=loss, metrics=metrics, device=device, verbose=True
    )

    max_score = 0
    loss_logs = {"train": [], "val": []}
    metric_logs = {"train": [], "val": []}

    for i in range(epochs):
        print(f"\nEpoch: {i}")
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)

        train_loss, _, train_iou = list(train_logs.values())
        val_loss, _, val_iou = list(valid_logs.values())

        loss_logs["train"].append(train_loss)
        loss_logs["val"].append(val_loss)
        metric_logs["train"].append(train_iou)
        metric_logs["val"].append(val_iou)

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, checkpoint_path)
            print('Model saved!')

        if i > 0 and i % 10 == 0:
            print('Decreasing LR')
            optimizer.param_groups[0]['lr'] /= 2

    return loss_logs, metric_logs
