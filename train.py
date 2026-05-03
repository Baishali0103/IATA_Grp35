import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from dataset import ShapesDataset
from model import CNNMultiHeadSmall, CNNMultiHeadDeep, ResNet18MultiHead


CSV_PATH = "output/metadata.csv"
IMAGE_DIR = "output/images"
CHECKPOINT_DIR = "checkpoints"

MODEL_NAME = "resnet18"   # choose: "small", "deep", "resnet18"

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
SEED = 42


def get_model(model_name: str):
    if model_name == "small":
        return CNNMultiHeadSmall()
    elif model_name == "deep":
        return CNNMultiHeadDeep()
    elif model_name == "resnet18":
        return ResNet18MultiHead()
    else:
        raise ValueError(f"Unknown MODEL_NAME: {model_name}")


def compute_loss(outputs, labels, criterion):
    loss = 0
    for key in outputs:
        loss += criterion(outputs[key], labels[key])
    return loss


def decode_outputs(outputs):
    preds = {}
    for key in outputs:
        preds[key] = outputs[key].argmax(dim=1)
    return preds


def exact_match_count(preds, labels):
    batch_size = preds["relation"].shape[0]
    correct = 0

    for i in range(batch_size):
        ok = True
        for key in preds:
            if preds[key][i].item() != labels[key][i].item():
                ok = False
                break
        if ok:
            correct += 1
    return correct


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_exact = 0
    total_samples = 0

    slot_correct = {
        "obj1_size": 0,
        "obj1_color": 0,
        "obj1_shape": 0,
        "relation": 0,
        "obj2_size": 0,
        "obj2_color": 0,
        "obj2_shape": 0,
    }

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            outputs = model(images)
            loss = compute_loss(outputs, labels, criterion)
            total_loss += loss.item()

            preds = decode_outputs(outputs)

            total_exact += exact_match_count(preds, labels)
            total_samples += images.size(0)

            for key in preds:
                slot_correct[key] += (preds[key] == labels[key]).sum().item()

    avg_loss = total_loss / len(loader)
    exact_acc = total_exact / total_samples
    slot_acc = {k: v / total_samples for k, v in slot_correct.items()}

    return avg_loss, exact_acc, slot_acc


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Training model:", MODEL_NAME)

    # Load dataset
    dataset = ShapesDataset(CSV_PATH, IMAGE_DIR)

    total_size = len(dataset)
    train_size = int(total_size * TRAIN_RATIO)
    val_size = int(total_size * VAL_RATIO)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    model = get_model(MODEL_NAME).to(device)

    criterion = nn.CrossEntropyLoss()


    if MODEL_NAME == "resnet18":
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_exact = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            outputs = model(images)
            loss = compute_loss(outputs, labels, criterion)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_exact, val_slot = evaluate(model, val_loader, device, criterion)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Exact Match Accuracy: {val_exact:.4f}")
        print(f"Val obj1_size Accuracy:   {val_slot['obj1_size']:.4f}")
        print(f"Val obj1_color Accuracy:  {val_slot['obj1_color']:.4f}")
        print(f"Val obj1_shape Accuracy:  {val_slot['obj1_shape']:.4f}")
        print(f"Val relation Accuracy:    {val_slot['relation']:.4f}")
        print(f"Val obj2_size Accuracy:   {val_slot['obj2_size']:.4f}")
        print(f"Val obj2_color Accuracy:  {val_slot['obj2_color']:.4f}")
        print(f"Val obj2_shape Accuracy:  {val_slot['obj2_shape']:.4f}")

        if val_exact > best_val_exact:
            best_val_exact = val_exact
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")

    print("\nTraining finished.")
    print(f"Best validation exact match accuracy: {best_val_exact:.4f}")

    # load best model and test
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, test_exact, test_slot = evaluate(model, test_loader, device, criterion)

    print("\n===== Test Results =====")
    print(f"Model: {MODEL_NAME}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Exact Match Accuracy: {test_exact:.4f}")
    print(f"obj1_size:  {test_slot['obj1_size']:.4f}")
    print(f"obj1_color: {test_slot['obj1_color']:.4f}")
    print(f"obj1_shape: {test_slot['obj1_shape']:.4f}")
    print(f"relation:   {test_slot['relation']:.4f}")
    print(f"obj2_size:  {test_slot['obj2_size']:.4f}")
    print(f"obj2_color: {test_slot['obj2_color']:.4f}")
    print(f"obj2_shape: {test_slot['obj2_shape']:.4f}")


if __name__ == "__main__":
    main()