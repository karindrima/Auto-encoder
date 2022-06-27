#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt


# In[4]:


# Transforms images to a PyTorch Tensor
tensor_transform = transforms.ToTensor()

# Download the MNIST Dataset
dataset = datasets.MNIST(root = "./data",
						train = True,
						download = True,
						transform = tensor_transform)

# DataLoader is used to load the dataset
# for training
loader = torch.utils.data.DataLoader(dataset = dataset,
									batch_size = 32,
									shuffle = True)


# In[5]:


# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class AE(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		# Building an linear encoder with Linear
		# layer followed by Relu activation function
		# 784 ==> 9
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(28 * 28, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 36),
			torch.nn.ReLU(),
			torch.nn.Linear(36, 18),
			torch.nn.ReLU(),
			torch.nn.Linear(18, 9)
		)
		
		# Building an linear decoder with Linear
		# layer followed by Relu activation function
		# The Sigmoid activation function
		# outputs the value between 0 and 1
		# 9 ==> 784
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(9, 18),
			torch.nn.ReLU(),
			torch.nn.Linear(18, 36),
			torch.nn.ReLU(),
			torch.nn.Linear(36, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 28 * 28),
			torch.nn.Sigmoid()
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded


# In[6]:


# Model Initialization
model = AE()

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
							lr = 1e-1,
							weight_decay = 1e-8)


# In[10]:


epochs = 20
outputs = []
losses = []
for epoch in range(epochs):
	for (image, _) in loader:
		
	# Reshaping the image to (-1, 784)
	 image = image.reshape(-1, 28*28)
		
	# Output of Autoencoder
	reconstructed = model(image)
		
	# Calculating the loss function
	loss = loss_function(reconstructed, image)
		
	# The gradients are set to zero,
	# the the gradient is computed and stored.
	# .step() performs parameter update
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
		
	# Storing the losses in a list for plotting
	losses.append(loss)
	outputs.append((epochs, image, reconstructed))
	print("Epoch: %d"%epoch)



# In[23]:


losses


# In[27]:


l = [i.item() for i in losses]
# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')

# Plotting the last 100 values
plt.plot(l)


# In[31]:


image.size()


# In[33]:


with torch.no_grad():
	for i, item in enumerate(image):
		
		# Reshape the array for plotting
		item = item.reshape(-1, 28, 28)
		plt.imshow(item[0])

	for i, item in enumerate(reconstructed):
		item = item.reshape(-1, 28, 28)
		plt.imshow(item[0])


# In[ ]:




