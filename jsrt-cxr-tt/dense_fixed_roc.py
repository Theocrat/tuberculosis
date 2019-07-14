import numpy as np
from extract_features  import extract_features as ef
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import roc_curve
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
for epoch in range(500):
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

# Testing the model
x_test, y_test = ef.fetch_test()
x_ts = Variable(torch.Tensor(x_test).type(torch.FloatTensor))
y_ts = torch.Tensor([i for i in y_test]).type(torch.LongTensor)
y_pred = model(x_ts)

labels = y_ts.detach().numpy()
decs_1 = y_pred[:,1].detach().numpy()
print('Labels:', labels.shape)
print('Decisions:', decs_1.shape)
fpr, tpr, thr = roc_curve(labels, decs_1)

# Plotting the result:
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# Evaluating the AUC:
area = np.trapz(tpr, fpr)
print('Area under the AUC curve:', area)
