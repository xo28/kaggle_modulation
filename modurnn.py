import numpy as np
import torch
import torch.nn as nn
import h5py
import matplotlib.pyplot as plt
import pandas as pd

#loading data
f = h5py.File("data.hdf5", 'r')
X_train = np.array(f['train'])
X_train = torch.from_numpy(X_train)
X_test = np.array(f['test'])
X_test = torch.from_numpy(X_test)
Y_train = pd.read_csv('train_labels.csv').to_numpy()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_index = {
 'FM': 0,
 'OQPSK':1,
 'BPSK':2,
 '8PSK':3,
 'AM-SSB-SC':4,
 '4ASK':5,
 '16PSK':6,
 'AM-DSB-SC':7, 
 'QPSK': 8, 
 'OOK':9
}

classnum = 10
def classToIndex(catg):
    return class_index[catg]

# turn a class into a <1 x 10> Tensor
def classToTensor(catg):
    tensor = torch.zeros(1, classnum)
    tensor[0][classToIndex(catg)] = 1
    return tensor

# Turn dataset into a <lines x 1 x classnum>,
# or an array of one-hot letter vectors
def setToTensor(dataset):
    tensor = torch.zeros(dataset.shape[0], 1, classnum)
    for li, catg in enumerate(dataset[:,1]):
        tensor[li][0][classToIndex(catg)] = 1
    return tensor

#direct mapping, not one-hot
def setToNum(dataset):
    tensor = torch.zeros(dataset.shape[0])
    for li, catg in enumerate(dataset[:,1]):
        tensor[li] = classToIndex(catg)
    return tensor

# Hyper-parameters
sequence_length = 1024
input_size = 2
hidden_size = 128
num_layers = 1
num_classes = 10
batch_size = 30000
num_epochs = 10000
learning_rate = 0.01

Y_number = setToNum(Y_train)

# divide it into batch
train_dataset = Data.TensorDataset(X_train,Y_number)
train_loader = Data.DataLoader(dataset=train_dataset,
                              batch_size=batch_size, 
                              shuffle=True)
test_loader = Data.DataLoader(dataset = X_test,
                              batch_size=batch_size,
                              shuffle=False)

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (samples, labels) in enumerate(train_loader):
#         print(samples.shape)
        samples = samples.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
#         print(labels.shape)
        
        # Forward pass
        outputs = model(samples)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 300 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Train accuracy
model.eval()
with torch.no_grad():
    traincorrect = 0
    total = 30000
    
    for trainsamples, labels in train_loader:
        trainsamples = trainsamples.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        trainoutputs = model(trainsamples)
        _, trainpredicted = torch.max(trainoutputs.data, 1)
        traincorrect += (trainpredicted == labels).sum().item()
        
    print('Train Accuracy of the model over the 30000 train modulations: {} %'.format(100 * traincorrect / total)) 

# Test the model
model.eval()
with torch.no_grad():

    for testsamples in test_loader:
        testsamples = images.reshape(-1, sequence_length, input_size).to(device)
        
        outputs = model(testsamples)
        _, predicted = torch.max(outputs.data, 1)
        prediction = pd.DataFrame(predicted, header = True, index = False)
        prediction.to_csv('test_prediction.csv')