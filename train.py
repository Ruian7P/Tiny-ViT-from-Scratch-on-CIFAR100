import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

def get_dataloaders(image_size, batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

def warmup_cosine_lr(epoch, warmup_epochs, num_epochs):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs  # Linear warmup
    else:
        return 0.5 * (math.cos((epoch - warmup_epochs) / (num_epochs - warmup_epochs) * math.pi) + 1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_metrics(train_losses, train_accs, val_losses, val_accs, save_dir='pictures'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'losses.png'))
    plt.close()
    
    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracies.png'))
    plt.close()

def train_model(model, trainloader, testloader, num_epochs, device, lr=1e-3, weight_decay=5e-4):
    # Print model parameters
    num_params = count_parameters(model)
    print(f"\nNumber of trainable parameters: {num_params:,}\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler with warmup
    warmup_epochs = 10
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda epoch: warmup_cosine_lr(epoch, warmup_epochs, num_epochs)
    )

    # Lists to store metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate epoch metrics
        train_acc = 100 * correct / total
        epoch_loss = epoch_loss / len(trainloader)
        train_losses.append(epoch_loss)
        train_accs.append(train_acc)
        
        print(f"Epoch: {epoch + 1}, Training Accuracy: {train_acc:.2f}%, Training Loss: {epoch_loss:.3f}")
        scheduler.step()

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_epoch_loss = 0.0
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_epoch_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        val_epoch_loss = val_epoch_loss / len(testloader)
        val_losses.append(val_epoch_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch: {epoch + 1}, Validation Accuracy: {val_acc:.2f}%, Validation Loss: {val_epoch_loss:.3f}")
        print("------------------------------------------")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot and save metrics
    plot_metrics(train_losses, train_accs, val_losses, val_accs)
    
    return best_val_acc 