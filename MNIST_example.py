from model import *
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torchvision

BATCH_SIZE_TRAIN = 1
BATCH_SIZE_TEST = BATCH_SIZE_TRAIN

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./datasets', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=BATCH_SIZE_TRAIN, shuffle=True)

# Test set not needed for now
'''
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./datasets', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=BATCH_SIZE_TRAIN, shuffle=True)
'''

# Define model architecture
layer_widths = [100, 100, 100, 784]
last_layer_index = len(layer_widths) - 1 

# Which layers will be clamped? Let's clamp the last layer to the inputs
clamped_layers = [last_layer_index]

# Define our activation tuple
# ReLU: first element of tuple is ReLU, second element is first derivative
activations = (lambda x: x if x > 0 else 0, lambda x: 1 if x > 0 else 0)

# Define model using our parameters, we'll use the default value for the update_rate
pcmodel = PC_Model(layer_widths=layer_widths, activations=activations, clamped_layers=clamped_layers, update_rate=0.01)

unsettled_energies = []
settled_energies = []
for x in tqdm(train_loader):
    x = x[0].flatten()
    pcmodel.reset_activity()
    pcmodel.set_activity(last_layer_index, x)
    unsettled_energies.append(pcmodel.get_energy())
    pcmodel.settle(settle_ratio=0.01)
    settled_energies.append(pcmodel.get_energy())
    pcmodel.settle_weights(settle_ratio=0.01)
    #x = x.reshape((28, 28))
    #plt.imshow(x)
    #plt.show()
    #break

plt.plot(unsettled_energies)
plt.plot(settled_energies)
plt.show()
