import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms, datasets

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # In: 28x28 for each pixel in image
        # Out: 64 for next layer
        self.fc1 = nn.Linear(28 * 28, 64)
        # In: 64 from previous layer
        # Out: 64 for next layer
        self.fc2 = nn.Linear(     64, 64)
        # In: 64 from previous layer
        # Out: 64 for next layer
        self.fc3 = nn.Linear(     64, 64)
        # In: 64 from previous layer
        # Out: 10 for number classes
        self.fc4 = nn.Linear(     64, 10)
    
    def forward(self, x):
        # Evaluate which neurons are firing on layer 1
        x = F.relu(self.fc1(x))
        # Evaluate which neurons are firing on layer 2
        x = F.relu(self.fc2(x))
        # Evaluate which neurons are firing on layer 3
        x = F.relu(self.fc3(x))
        # Just pass the data to layer 4
        x = self.fc4(x)
        
        # The final backpropagation function
        return F.log_softmax(x, dim=1)

training_set = datasets.MNIST(
    root='data/',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

test_set = datasets.MNIST(
    root='data/',
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

trainset = data.DataLoader(training_set, batch_size=16, shuffle=True)
testset = data.DataLoader(test_set, batch_size=16, shuffle=True)

net = Net()

optimizer = optim.Adam(net.parameters(), lr=1e-3)

# 1 EPOCH is a total pass through the data
EPOCHS = 3

# X[0] is the greyscale pixel data
# y[0] is the number pictured
for epoch in range(EPOCHS):
    for data in trainset:
        # Data is a batch of feature sets and labels
        X, y = data
        
        # Set each gradient of the network to zero
        net.zero_grad()
        
        # Calculate the prediction
        output = net(X.view(-1, 28 * 28))
        
        # Calculate loss (how severly to edit the weights based on the predictions)
        loss = F.nll_loss(output, y)
        
        # Backpropagate
        loss.backward()
        
        # Actually adjusts the weights
        optimizer.step()
    
correct = 0
total = 0

with torch.no_grad():
    for data in trainset:
        # Data is a batch of feature sets and labels
        X, y = data
        output = net(X.view(-1, 28 * 28))
        
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
            
            if correct == 0 and total == 0:
                accuracy = 0
            else:
                accuracy = correct / total
                
            print(f'Actual: { y[idx] }, Prediction: { torch.argmax(i) }, Accuracy: { round((accuracy) * 100, 3) }%')

print(f'Overall Accuracy: { round((correct / total) * 100, 3) }%')