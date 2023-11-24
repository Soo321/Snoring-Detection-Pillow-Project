import os
import time
import torch
from pandas import DataFrame

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, saved_dir, device):
    print('----------Start----------')
    
    model.train()
    best_loss = 100

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        cnt = 0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            total += imgs.size(0)
        
            optimizer.zero_grad()
        
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()

            cnt += 1

        valid_loss, valid_acc = valid_model(model, valid_loader, criterion, device)

        if best_loss > valid_loss:
            best_loss = valid_loss
            check_point = {
                'saved_model': model.state_dict()
            }
            saved_epoch = epoch + 1
            os.makedirs(saved_dir, exist_ok=True)
            saved_path = os.path.join(saved_dir, 'best_model_at_epoch_{}.pt'.format(saved_epoch))
            torch.save(check_point, saved_path)

        print('Train Epoch {} Loss: {:.4f} Accuracy: {:.2f}%'.format(
            epoch + 1, total_loss / cnt, correct / total * 100
            ))
        print('Valid Epoch {} Loss: {:.4f} Accuracy: {:.2f}%'.format(epoch + 1, valid_loss, valid_acc))
    
    print('----------Finish----------')
    print('Best performance at Epoch {}'.format(saved_epoch))
    
    return saved_epoch

def valid_model(model, valid_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        correct = 0
        cnt = 0
        total = 0
        for i, (imgs, labels) in enumerate(valid_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            total += imgs.size(0)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()
            cnt += 1
    
    valid_loss = total_loss / cnt
    valid_acc = correct / total * 100

    model.train()

    return valid_loss, valid_acc

def test_model(model, test_loader, saved_dir, saved_epoch, device):
    model_path = saved_dir + '/' + 'best_model_at_epoch_{}.pt'.format(saved_epoch)
    check_point = torch.load(model_path, map_location = device)
    state_dict = check_point['saved_model']
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        total_label = []
        total_result = []

        for i, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, argmax = torch.max(outputs, 1)
            total += imgs.size(0)
            correct += (labels == argmax).sum().item()
            total_label.extend(labels)
            total_result.extend(argmax)
        
        print('Test accuracy: {:.2f}'.format(correct / total * 100))

        raw_result = {
            'target': map(int, total_label),
            'output': map(int, total_result)}
        result = DataFrame(raw_result)
        result_path = saved_dir + '/result'


       os.makedirs(result_path, exist_ok =True)
        end = time.time()
        result.to_csv(result_path + '/' + 'test_at_{}.csv'.format(end))
