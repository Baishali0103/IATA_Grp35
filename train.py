import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TicTacToeDataset
from model import CNNModel

dataset = TicTacToeDataset("data/labels.txt","data/images")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for images, labels in loader:

        outputs = model(images)

        loss = 0
        for i in range(9):
            loss += criterion(outputs[:,i,:], labels[:,i])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")