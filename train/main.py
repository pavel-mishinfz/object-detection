import torch
import matplotlib.pyplot as plt
from train import get_model, test_loader, NUM_CLASSES, MODEL_DIR
from utils import apply_nms, visualize
import random


def load_trained_model():
    """Загружает предварительно обученную модель"""
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_DIR / "faster_rcnn_epoch_50_v5.pth",
                                     map_location="cpu"))
    model.eval()
    return model


def get_random_samples(loader, num_samples=10):
    """Возвращает случайные выборки из датасета"""
    all_samples = list(loader)
    return random.sample(all_samples, num_samples)


def plot_comparison(images, targets, predictions):
    """Визуализирует изображение с GT и предсказаниями"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # Визуализация Ground Truth
    visualize(images[0], targets[0], ax=ax1, title="Ground Truth")

    # Визуализация предсказаний
    visualize(images[0], targets[0], prediction=predictions[0],
              ax=ax2, title="Predictions")

    plt.tight_layout()
    plt.show()


def main():
    # Инициализация
    model = load_trained_model()

    # Получение случайных примеров
    samples = get_random_samples(test_loader)

    # Обработка и визуализация
    with torch.no_grad():
        for images, targets in samples:
            images = [img for img in images]
            predictions = model(images)
            predictions = [apply_nms(pred) for pred in predictions]
            plot_comparison(images, targets, predictions)


if __name__ == "__main__":
    main()
