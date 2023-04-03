import os
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import numpy as np

os.chdir(os.path.join(os.getcwd(), "dataset", "poses"))
print(os.listdir())

batch_size = 4

poses = ['downdog', "goddess", "mountain", "tree", "warrior1", "warrior2"]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train = torchvision.datasets.ImageFolder(root="train/",
                                           transform=transform)
train_loader = torch.utils.data.DataLoader(train,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=4)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{poses[labels[j]]:5s}' for j in range(batch_size)))