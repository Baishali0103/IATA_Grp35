import random
import torch
from torch.utils.data import DataLoader, Subset

from dataset import ShapesDataset
from model import CNNMultiHeadSmall, CNNMultiHeadDeep, ResNet18MultiHead
from utils import build_sentence_from_labels

CSV_PATH = "output/metadata.csv"
IMAGE_DIR = "output/images"

MODEL_NAME = "small"   # 可选: "small", "deep", "resnet18"
MODEL_PATH = f"checkpoints/{MODEL_NAME}_best.pt"

NUM_SAMPLES_TO_SHOW = 5


def get_model(model_name: str):
    if model_name == "small":
        return CNNMultiHeadSmall()
    elif model_name == "deep":
        return CNNMultiHeadDeep()
    elif model_name == "resnet18":
        return ResNet18MultiHead()
    else:
        raise ValueError(f"Unknown MODEL_NAME: {model_name}")


def decode_outputs(outputs):
    preds = {}
    for key in outputs:
        preds[key] = outputs[key].argmax(dim=1)
    return preds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Predicting with model:", MODEL_NAME)
    print("Loading checkpoint:", MODEL_PATH)

    dataset = ShapesDataset(CSV_PATH, IMAGE_DIR)

    num_samples = min(NUM_SAMPLES_TO_SHOW, len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)
    subset = Subset(dataset, sample_indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    model = get_model(MODEL_NAME).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        for i, (images, labels, sentence) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            preds = decode_outputs(outputs)

            pred_dict = {k: preds[k][0].item() for k in preds}
            pred_sentence = build_sentence_from_labels(pred_dict)

            print(f"\nSample {i + 1}")
            print("Gold sentence:     ", sentence[0])
            print("Predicted sentence:", pred_sentence)


if __name__ == "__main__":
    main()