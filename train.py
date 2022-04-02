import os

import torch
import torch.nn as nn
from torch.utils.data import sampler
import torchvision
import torchvision.transforms as transforms
from model import Model
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import numpy as np

MAX_EPOCH = 150
early_stop_steps = 10
opt_name = "adam"
batch_size=64
lr = 1e-4

batch_norm=1
dense=0
dropout=0

momentum = 0 #For sgd
wd = 3e-4 # For adam

split_ratio = 0.1
seed =3136
torch.manual_seed(seed)

save_folder = f"./bs_{batch_size}_lr_{lr}_opt_{opt_name}_bn_{batch_norm}_do_{dropout}_dense_{dense}_momentum_{momentum}/"
os.makedirs(save_folder, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_train=transforms.Compose([
    transforms.Pad(4),
    transforms.RandomCrop(size=(32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

valset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_test)

n_obs = len(trainset)
indices = list(range(n_obs))
split_idx = int(np.floor(split_ratio * n_obs))
np.random.seed(seed)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split_idx:], indices[:split_idx]
train_sampler = sampler.SubsetRandomSampler(train_idx)
valid_sampler = sampler.SubsetRandomSampler(valid_idx)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, sampler=train_sampler, num_workers=10)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=False, sampler=valid_sampler, num_workers=10)

if dense:
    fc_out = (1024, 128)
    out_channels = (128,64,32)
else:
    fc_out = (512, 128)
    out_channels = (256,128,64)

model = Model(batch_norm=batch_norm, out_channels=out_channels, drop_out=dropout, dense=dense, fc_out=fc_out).to(device)
print(model)
loss_fn = nn.CrossEntropyLoss()
torch.save(model, save_folder +"model_obj.pkl")

if opt_name == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=wd)
elif opt_name =="sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
elif opt_name == "rmsprop":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
else:
    raise ValueError("Optimizer not supported.")


train_losses=[]
train_accuracies=[]
val_losses=[]
val_accuracies=[]
best_val_acc = 0
early_stop_count = 0
for epoch in range(MAX_EPOCH):
    train_loss = 0
    train_corrects = 0
    for images, labels in tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        loss = loss_fn(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(out, 1)

        train_loss += loss.item()
        train_corrects += torch.sum(preds == labels.data)

    val_loss = 0
    val_corrects = 0
    with torch.no_grad():
        for val_images, val_labels in valloader:
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            val_out = model(val_images)
            loss = loss_fn(val_out, val_labels)

            _, preds = torch.max(val_out, 1)

            val_loss += loss.item()
            val_corrects += torch.sum(preds == val_labels.data)

    train_loss = train_loss/(len(trainloader))
    train_acc = train_corrects/(len(trainloader)*batch_size)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    val_loss = val_loss/(len(valloader))
    val_acc = val_corrects/(len(valloader)*batch_size)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    if val_acc > best_val_acc:
        early_stop_count=0
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_folder + "best_model.pkl")
    else:
        early_stop_count+=1

    if early_stop_count >= early_stop_steps:
        break

    print(f"""
    Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}
                   Train Acc:  {train_acc},  Val Acc:  {val_acc}""")

    if epoch==20:
        torch.save(model.state_dict(), save_folder + "middle_model.pkl")

metrics = {}
metrics["train_accuracy"] = train_accuracies
metrics["validation_accuracy"] = val_accuracies
metrics["train_loss"] = train_losses
metrics["validation_loss"] = val_losses
metrics["batch_size"] = batch_size
metrics["learning_rate"] = lr
metrics["batch_norm"] = batch_norm
metrics["drop_out"] = dropout
metrics["momentum"] = momentum
metrics["weight_decay"] = wd

with open(save_folder + "summary.pkl", "wb") as file:
    pickle.dump(metrics, file)

plt.style.use("ggplot")
plt.figure(figsize=(12,8))
plt.plot(list(range(len(train_accuracies))), train_accuracies, label='Train Accuracy')
plt.plot(list(range(len(val_accuracies))), val_accuracies, label='Validation Accuracy')
plt.xlabel("# of Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(save_folder + "accuracy_plot.png")

plt.figure(figsize=(12,8))
plt.plot(list(range(len(train_losses))), train_losses, label='Train Loss')
plt.plot(list(range(len(val_losses))), val_losses, label='Validation Loss')
plt.xlabel("# of Epoch")
plt.ylabel("CE Loss")
plt.legend()
plt.savefig(save_folder  + "loss_plot.png")

print("Best Acc: ", best_val_acc)
