import torch
import torch.nn as nn

from sound_data import make_dataloader
from model import make_model
from trainer import train_model, test_model

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train_loader, valid_loader, test_loader = make_dataloader(path = 'D:/data/mbe_project/Snoring Dataset', \
        img_size = 224, train_test_ratio = 0.05, valid_ratio = 0.15, batch_size = 32)

    model = make_model(category = 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    num_epochs = 30
    saved_dir = 'D:\workspaces\mbe_project\saved_dir'

    saved_epoch = train_model(model, train_loader, valid_loader, criterion, optimizer, \
        num_epochs, saved_dir, device)

    test_model(model, test_loader, saved_dir, saved_epoch, device)
