import numpy as np

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as fn
import torch

from cls_dataset import SeqDataset
from cls_model import TemplateNetwork
from funct_metrics import *


def train_one_epoch(model, train_loader, optimizer, criterion, device_id):
    # Put model into train mode
    model = model.train()
    for train_index, (train_data, train_labels) in enumerate(train_loader):
        # --- Training begins --- #
        # Send data to gpu, if there is GPU
        if torch.cuda.is_available():
            train_data = train_data.cuda(device_id)
            train_labels = train_labels.cuda(device_id)
        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(train_data)
        # Calculate loss
        loss = criterion(outputs, train_labels)
        # Backward pass
        loss.backward()
        # Update gradients
        optimizer.step()
        # --- Training ends --- #


def make_prediction(model, data_loader, device_id):
    prediction_list = []
    label_list = []

    # Put model into evaluation mode
    model = model.eval()

    for data_index, (data, labels) in enumerate(data_loader):
        # Send data to gpu
        if torch.cuda.is_available():
            data = data.cuda(device_id)
        # Forward pass without gradients
        with torch.no_grad():
            outputs = model(data)
        # If the model is in GPU, get data to cpu
        if torch.cuda.is_available():
            outputs = outputs.cpu()

        # Add predictions and labels to respective lists
        preds = torch.argmax(outputs, dim=1)
        label_list.extend(labels.tolist())
        prediction_list.extend(preds.tolist())
    return np.array(prediction_list), np.array(label_list)


if __name__ == "__main__":
    # Group-ID - don't forget to fill here
    group_id = None
    assert isinstance(group_id, int), 'Dont forget to add your group id'
    device_id = group_id % 2
    if torch.cuda.is_available():
        print('Using GPU:', device_id)
        print('Warning: If you are using a server with a single gpu')
        print('Manually change device_id to 0 at line 64')

    # Hyperparameters
    # -----> Tune hyperparameters here
    learning_rate = 9.99
    momentum = 9.99
    weight_decay = 9.9999
    batch_size = 99
    num_epoch = 1
    # -----> Tune hyperparameters here

    # Define datasets and data loaders
    tr_dataset = SeqDataset([('../data/splice_train_pos.txt', 1),
                             ('../data/splice_train_neg.txt', 0)])
    tr_loader = DataLoader(dataset=tr_dataset, batch_size=batch_size,
                           shuffle=True, num_workers=8)

    ts_dataset = SeqDataset([('../data/splice_val_pos.txt', 1),
                             ('../data/splice_val_neg.txt', 0)])
    ts_loader = DataLoader(dataset=ts_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=8)

    # Model
    model = TemplateNetwork()
    if torch.cuda.is_available():
        model = model.cuda(device_id)

    # Loss
    # For additional losses see: https://pytorch.org/docs/stable/nn.html
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    # For additional optimizers see: https://pytorch.org/docs/stable/optim.html
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay,
                                momentum=momentum)
    for epoch in range(num_epoch):
        print('Epoch: ', epoch, 'starts')
        # Train the model
        train_one_epoch(model, tr_loader, optimizer, criterion, device_id)
        # Make prediction
        preds, labels = make_prediction(model, ts_loader, device_id)

        # -----> Calculate metrics here
        #
        #
        # -----> Calculate metrics here

        # -----> Save the model performance here
        #
        #
        # -----> Save the model performance here
        print('Epoch: ', epoch, 'ends')

    torch.save(model.cpu(), 'my_model.pth')
    # Save the model
