import numpy as np
from extract_features  import extract_features as ef
from matplotlib.pyplot import stem, show
import torch
import torch.nn as nn
from torch.autograd import Variable
from sys import exit

# TRAINING PART:
# -------------

# Extracting Training Data:
(X_train, y_train) = ef.fetch_train()

# Create dummy input and target tensors (data)
x = Variable(torch.Tensor(X_train).type(torch.FloatTensor))
y = torch.Tensor([i for i in y_train]).type(torch.LongTensor)

class myModel(torch.nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.layer_inp = nn.Linear(19, 11,  bias=True)
        self.layer_hid = nn.Linear(11, 6,  bias=True)
        self.layer_out = nn.Linear(6, 2, bias=True)
        #nn.init.xavier_uniform_(self.layer_inp.weight)
        #nn.init.xavier_uniform_(self.layer_hid.weight)
        #nn.init.xavier_uniform_(self.layer_out.weight)
        
    def forward(self, x1):
        h1 = torch.tanh(self.layer_inp(x1)) #.clamp(min = 0)
        h2 = torch.tanh(self.layer_hid(h1)) #.clamp(min = 0)
        y1 = self.layer_out(h2)
        return torch.softmax(y1,1)

model = myModel()

# Construct the loss function
criterion = torch.nn.CrossEntropyLoss() #reduction = 'sum')

# Construct the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Gradient Descent
Losses = []
for epoch in range(50):
    # Forward pass: Compute predicted y by passing x to the model
    #skip = (450 - batch_size) / 50
    #x_batch = x[epoch * skip: epoch * skip + batch_size]
    #x = x[:10]
    #y = y[:10]
    #x = torch.rand(450, 105)
    y_pred  = model(x)
    # Compute and print loss
    #y_batch = y[epoch * skip: epoch * skip + batch_size]
    loss = criterion(y_pred, y)
    print('Epoch:', epoch, 'Loss:', loss)
    Losses.append( loss.item() )
    
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    
    # perform a backward pass (backpropagation)
    for i in range(6):
        list(model.parameters())[i].retain_grad()
    loss.backward()
    
    # Update the parameters
    optimizer.step()

print('Initial loss:', Losses[0])
print('Final  loss:', Losses[-1])

decisions = np.zeros(150)
labels    = list(y)
for i in range(150):
    if y_pred[i,1] > y_pred[i,0]:
        decisions[i] = 1
decisions = list(decisions)
correct = 0
for i in range(150):
    if decisions[i] == labels[i]:
        correct += 1

print('Accuracy:', float(correct)/150.0)
