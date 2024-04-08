#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import random
import tqdm


# In[ ]:


PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0


# In[ ]:


attn_data = pd.read_csv('../gpt2_small/attention_map_labels/all_attn_labels_modified.csv', index_col = False)


# In[ ]:


attn_data.shape


# label_mapping = {'B':0, 'D':1, 'M1':2, 'O':3}  
# D - Diagonal,
# O - Overall,
# M1 - Majority in part 1, 
# B - Block,
# 

# In[ ]:


attn_data.shape


# In[ ]:


grouped_df = attn_data.groupby('num_labels')
count_values = grouped_df.count()
count_values


# In[ ]:


indices_to_delete = attn_data.index[attn_data['num_labels'] == 3].tolist()
to_keep = 250
random.shuffle(indices_to_delete)
attn_data = attn_data.drop(indices_to_delete[to_keep:])
attn_data = attn_data.reset_index(drop=True)
grouped_df = attn_data.groupby('num_labels')
count_values = grouped_df.count()
count_values


# In[ ]:


attn_data.shape


# In[ ]:


train, test = train_test_split(attn_data, test_size=0.2, random_state=42, stratify = attn_data['num_labels'])
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
train.shape, test.shape


# In[ ]:


grouped_df = train.groupby('num_labels')
count_values = grouped_df.count()
count_values


# In[ ]:


del attn_data


# In[ ]:


grouped_df = test.groupby('num_labels')
count_values = grouped_df.count()
count_values


# In[ ]:


class MyDataset(Dataset):
    def __init__(self, attn_data):
        self.attn_data = attn_data

    def __len__(self):
        return len(self.attn_data)

    def __getitem__(self, idx):
        labels = self.attn_data.loc[idx, 'num_labels']
        labels = torch.tensor(labels)
        loaded_array = np.load(self.attn_data.loc[idx, 'numpy_path'])
        float32_array = loaded_array.astype(np.float32)
        float32_array = torch.tensor(float32_array)
        float32_array = float32_array.unsqueeze(0)
        return float32_array, labels


# In[ ]:


my_dataset = MyDataset(train)
batch_size = 4
train_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

my_dataset = MyDataset(test)
batch_size = 4
test_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=False)


# In[ ]:


for b in train_dataloader:
    array, label = b
    print(label.shape, array.shape)
    break


# In[ ]:


class CNN(nn.Module):
    def __init__(self, classes):
        super(CNN, self).__init__()  
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride = 2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride = 2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride = 2)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride = 2)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride = 2)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride = 2)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride = 2)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride = 2)
        self.bn8 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 3 * 3, classes)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        #print('1',x.shape)
        x = self.relu(self.bn2(self.conv2(x)))
        #print('2',x.shape)
        x = self.relu(self.bn3(self.conv3(x)))
        #print('3',x.shape)
        x = self.relu(self.bn4(self.conv4(x)))
        #print('4',x.shape)
        x = self.relu(self.bn5(self.conv5(x)))
        #print('5',x.shape)
        x = self.relu(self.bn6(self.conv6(x)))
        #print('6',x.shape)
        x = self.relu(self.bn7(self.conv7(x)))
        #print('7',x.shape)
        x = self.relu(self.bn8(self.conv8(x)))
        #print('8',x.shape)
        x = x.view(x.shape[0], 64 * 3 * 3)
        #print('flat',x.shape)
        x = self.fc(x)
        #print('fc',x.shape)
        return x


# In[ ]:


classes = 4
model = CNN(classes)
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters in the model: {total_params}')


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
epochs = 20
device = torch.device('mps')
model = model.to(device)

for epoch in range(0, epochs):
    print(epoch)
    model.train()
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_dataloader)):
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data) 
        loss = criterion(output, target)    
        loss.backward()    
        optimizer.step()    
        optimizer.zero_grad()
        
    correct = 0
    total = 0
    model.eval()  
    for data, target in train_dataloader:  
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data) 
        _, predicted = torch.max(output.data, 1)  
        total += target.size(0)
        correct += (predicted == target).sum().item()
    accuracy = correct / total
    print(f'Accuracy on the train set: {accuracy:.4f}')
    
    correct = 0
    total = 0
    model.eval()  
    for data, target in test_dataloader:
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data) 
        _, predicted = torch.max(output.data, 1)  
        total += target.size(0)
        correct += (predicted == target).sum().item()

    accuracy = correct / total
    print(f'Accuracy on the test set: {accuracy:.4f}')
    
    
  


# In[ ]:


train_TP = [0] * 4  
train_FP = [0] * 4  
train_FN = [0] * 4  

test_TP = [0] * 4  
test_FP = [0] * 4  
test_FN = [0] * 4  

# Evaluate on test dataset
correct = 0
total = 0

model.eval()  
for data, target in test_dataloader:
    data = data.to(device).float()
    target = target.to(device).long()
    output = model(data) 
    
    _, predicted = torch.max(output.data, 1)  
    
    total += target.size(0)
    correct += (predicted == target).sum().item()
    
    for c in range(4):  # Iterate over each class
        test_TP[c] += torch.sum((predicted == c) & (target == c)).item()
        test_FP[c] += torch.sum((predicted == c) & (target != c)).item()
        test_FN[c] += torch.sum((predicted != c) & (target == c)).item()

accuracy = correct / total
print(f'Accuracy on the test set: {accuracy:.4f}')

# Evaluate on train dataset
correct = 0
total = 0

for data, target in train_dataloader:
    data = data.to(device).float()
    target = target.to(device).long()
    output = model(data) 
    
    _, predicted = torch.max(output.data, 1)  
    
    total += target.size(0)
    correct += (predicted == target).sum().item()
    
    for c in range(4):  # Iterate over each class
        train_TP[c] += torch.sum((predicted == c) & (target == c)).item()
        train_FP[c] += torch.sum((predicted == c) & (target != c)).item()
        train_FN[c] += torch.sum((predicted != c) & (target == c)).item()

accuracy = correct / total
print(f'Accuracy on the train set: {accuracy:.4f}')

# Calculate overall F1 score for both train and test datasets
def calculate_overall_f1_score(TP, FP, FN):
    f1_scores = []
    for c in range(4):
        precision = TP[c] / (TP[c] + FP[c] + 1e-9)
        recall = TP[c] / (TP[c] + FN[c] + 1e-9)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
        f1_scores.append(f1_score)
    return np.mean(f1_scores)

overall_train_f1 = calculate_overall_f1_score(train_TP, train_FP, train_FN)
overall_test_f1 = calculate_overall_f1_score(test_TP, test_FP, test_FN)

print(f'Overall F1 Score on the train set: {overall_train_f1:.4f}')
print(f'Overall F1 Score on the test set: {overall_test_f1:.4f}')


# In[ ]:


torch.save(model.state_dict(), '../trained_models/easy_transformer_gpt2small_cnn.pth')


# In[ ]:




