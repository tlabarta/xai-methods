import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import copy
import pandas as pd
import os

# Set GPU device
print(torch.cuda.is_available())
device = torch.device("cuda:0")

# Load data
TRAIN_ROOT = "data/brain_mri/training"
TEST_ROOT = "data/brain_mri/testing"
train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_ROOT)
test_dataset = torchvision.datasets.ImageFolder(root=TEST_ROOT)

# Building the model (ResNet50)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)

        # Adjust the output size of the final layer according to your problem
        num_classes = 4  # Change this to your desired number of classes
        self.resnet50.fc = nn.Linear(2048, num_classes)  # Assuming you want to change the final fully connected layer

    def forward(self, x):
        x = self.resnet50(x)
        return x

model = CNNModel()
model.to(device)
model

# Checkpoint paths
checkpoint_path = "resnet50_checkpoint.pth"

# Load a pre-trained checkpoint if it exists
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()  # Set the model to evaluation mode for inference
else:
    # Prepare data for the pretrained model
    train_dataset = torchvision.datasets.ImageFolder(
        root=TRAIN_ROOT,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet50 expects 224x224 images
            transforms.ToTensor()
        ])
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=TEST_ROOT,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )

    # Create data loaders
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Train
    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    epochs = 10

    for epoch in range(epochs):
        for i, batch in enumerate(train_loader, 0):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cross_entropy_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            print(loss)

    # Save the trained model checkpoint
    torch.save(model.state_dict(), checkpoint_path)

# Inspect predictions for the first batch
import pandas as pd
inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.numpy()
outputs = model(inputs).max(1).indices.detach().cpu().numpy()
comparison = pd.DataFrame()
print("Batch accuracy: ", (labels == outputs).sum() / len(labels))
comparison["labels"] = labels
comparison["outputs"] = outputs

# Layerwise relevance propagation for ResNet50 with skip connections

def apply_lrp_on_resnet50(model, image):
    # Rest of the LRP code (as previously provided)
    # ...
    return relevances[0]

# Now you can use this modified LRP function with ResNet50
image_id = 31
image_relevances = apply_lrp_on_resnet50(model, inputs[image_id])

# %%
# Calculate relevances for first image in this test batch
image_id = 31
image_relevances = apply_lrp_on_vgg16(model, inputs[image_id])
image_relevances = image_relevances.permute(0,2,3,1).detach().cpu().numpy()[0]
image_relevances = np.interp(image_relevances, (image_relevances.min(),
                                                image_relevances.max()), 
                                                (0, 1))
# Show relevances
pred_label = list(test_dataset.class_to_idx.keys())[
             list(test_dataset.class_to_idx.values())
            .index(labels[image_id])]
if outputs[image_id] == labels[image_id]:
    print("Groundtruth for this image: ", pred_label)

    # Plot images next to each other
    plt.axis('off')
    plt.subplot(1,2,1)
    plt.imshow(image_relevances[:,:,0], cmap="seismic")
    plt.subplot(1,2,2)
    plt.imshow(inputs[image_id].permute(1,2,0).detach().cpu().numpy())
    plt.show()
else:
    print("This image is not classified correctly.")

# %%
