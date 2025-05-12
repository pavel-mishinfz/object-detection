import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms as T
import torch


def get_transform():
    transforms = [
        T.ToTensor()
    ]
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def apply_nms(prediction, iou_thresh=0.3):
    keep = torch.ops.torchvision.nms(prediction['boxes'], prediction['scores'], iou_thresh)
    prediction['boxes'] = prediction['boxes'][keep]
    prediction['scores'] = prediction['scores'][keep]
    prediction['labels'] = prediction['labels'][keep]
    return prediction


def visualize(img, target, prediction=None, ax=None, title="Image"):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    img = T.ToPILImage()(img).convert('RGB')
    ax.imshow(img)

    for box in target['boxes']:
        xmin, ymin, xmax, ymax = box.tolist()
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

    if prediction:
        for box in prediction['boxes']:
            xmin, ymin, xmax, ymax = box.tolist()
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    ax.set_title(title)
    ax.axis('off')


def plot_metrics(train_loss, val_metrics):
    start_epoch = 0
    epochs = list(range(1 + start_epoch, len(train_loss) + start_epoch + 1))

    plt.figure(figsize=(12, 5))

    # График функции потерь
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, "b-o", label="Train")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # График mAP
    plt.subplot(1, 3, 2)
    plt.plot(epochs, [m["mAP"] for m in val_metrics], 'r-o', label="mAP")
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP@[0.5:0.95]')
    plt.legend()
    plt.grid(True)

    # График AP@0.5
    plt.subplot(1, 3, 3)
    plt.plot(epochs, [m["AP@0.5"] for m in val_metrics], 'r-o', label="AP@0.5")
    plt.xlabel('Epoch')
    plt.ylabel('AP@0.5')
    plt.title('AP@0.5')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
