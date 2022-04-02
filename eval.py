import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import Model
import numpy as np
from sklearn.manifold import TSNE
import pickle
plt.style.use("ggplot")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

plot_latent_space =False
model_folder = "./bs_64_lr_0.0001_opt_adam_bn_1_do_0_dense_0_momentum_0/"

model_path = model_folder + "model_obj.pkl"
state_dict_path = model_folder + "best_model.pkl"
batch_size = 64

transform_test=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=10)


model = torch.load(model_path).to(device=device)
print(model)
model.load_state_dict(torch.load(state_dict_path))
model.eval()
loss_fn = torch.nn.CrossEntropyLoss()
latent_space_end = []

test_loss = 0
test_corrects = 0
with torch.no_grad():
    for test_images, test_labels in testloader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        test_out = model(test_images)
        loss = loss_fn(test_out, test_labels)
        _, preds = torch.max(test_out, 1)
        test_loss += loss.item()
        test_corrects += torch.sum(preds == test_labels.data)

test_loss = (test_loss/(len(testloader)))
test_acc = test_corrects/(len(testloader)*batch_size)

print(f"Test acc: {test_acc}, test loss: {test_loss}")

with open( model_folder + "summary.pkl","rb") as file:
    summary = pickle.load(file)

model_index = np.argmax(summary["validation_accuracy"])
print(
    f"""
Train Loss: {summary["train_loss"][model_index]}
Validation Loss: {summary["validation_loss"][model_index]}
Train Accuracy: {summary["train_accuracy"][model_index]}
Validation Accuracy" {summary["validation_accuracy"][model_index]}
    """
)

if plot_latent_space:
    with torch.no_grad():

        layer_out = {}
        def hook(module_, input_, output_):
            layer_out["value"] = output_.detach().cpu().numpy()

        begin_latent = []
        middle_latent = []
        end_latent = []
        labels = []
        begin_model = torch.load(model_path).to(device=device)
        middle_model = torch.load(model_path).to(device=device)
        middle_model.load_state_dict(torch.load(model_folder + "middle_model.pkl"))

        for test_images, test_labels in testloader:
            test_images = test_images.to(device)
            labels.append(test_labels.cpu().numpy())

            begin_model.fc2.register_forward_hook(hook)
            _ = begin_model(test_images)
            begin_latent.append(layer_out["value"])

            middle_model.fc2.register_forward_hook(hook)
            _ = middle_model(test_images)
            middle_latent.append(layer_out["value"])

            model.fc2.register_forward_hook(hook)
            _ = model(test_images)
            end_latent.append(layer_out["value"])

    begin_latent = np.concatenate(begin_latent, axis=0)
    middle_latent = np.concatenate(middle_latent, axis=0)
    end_latent = np.concatenate(end_latent, axis=0)
    labels = np.concatenate(labels, axis=0)

    def plot_latent_space(values, labels, name=""):
        classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=3136)
        values_tsne = tsne_model.fit_transform(values)
        x = values_tsne[:,0]
        y = values_tsne[:,1]

        label_numbers = np.unique(labels)
        col_map = {label:f"C{i}" for i, label in enumerate(label_numbers)}
        plt.figure(figsize=(12,12))
        for label_number in label_numbers:
            label_name = classes[label_number]
            plt.scatter(x[labels==label_number], y[labels==label_number], label=label_name)
        plt.legend()
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.title(f"TSNE Plot of {name} Latent Space")
        plt.savefig(f"{model_folder}{name}.png")

    plot_latent_space(begin_latent, labels, name="Begin")
    plot_latent_space(middle_latent, labels, name="Middle")
    plot_latent_space(end_latent, labels, name="End")
