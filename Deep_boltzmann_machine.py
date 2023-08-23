## Importing the libraries

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

## Importing the dataset
## Preparing the training set and the test set

# Loading training data from a file and converting it into a NumPy array
training_set = pd.read_csv(r"E:\Code\DL\ml-100k\u1.base", delimiter="\t")
training_set = np.array(training_set, dtype="int")

# Loading test data from a file and converting it into a NumPy array
test_set = pd.read_csv(r"E:\Code\DL\ml-100k/u1.test", delimiter="\t")
test_set = np.array(test_set, dtype="int")

## Getting the number of users and movies

# Calculating the number of users by finding the maximum user ID in both training and test sets
nb_users = int(
    max(
        max(
            training_set[:, 0],
        ),
        max(test_set[:, 0]),
    )
)

# Calculating the number of movies by finding the maximum movie ID in both training and test sets
nb_movies = int(
    max(
        max(
            training_set[:, 1],
        ),
        max(test_set[:, 1]),
    )
)

## Converting the data into an array with users in lines and movies in columns

# Function to convert the data into a user-movie matrix where each row corresponds to a user's ratings
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

# Converting training and test sets into user-movie matrices
training_set = convert(training_set)
test_set = convert(test_set)

## Converting the data into Torch tensors

# Converting user-movie matrices into PyTorch FloatTensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

## Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)

# Mapping ratings to binary values: 1 for ratings >= 3 (Liked), 0 for ratings < 3 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

## Creating the architecture of the Neural Network

# Defining the Deep Boltzmann Machine (DBM) class
class DBM:
    def __init__(self, layer_sizes):
        # Initializing DBM parameters: weights, biases for hidden and visible units
        self.num_layers = len(layer_sizes)
        self.W = [
            torch.randn(layer_sizes[i + 1], layer_sizes[i])
            for i in range(self.num_layers - 1)
        ]
        self.a = [
            torch.randn(1, layer_sizes[i + 1]) for i in range(self.num_layers - 1)
        ]
        self.b = [torch.randn(1, layer_sizes[i]) for i in range(self.num_layers - 1)]

    # Function to sample hidden units' states and return probabilities and binary samples
    def sample_hidden(self, x, layer_index):
        wx = torch.mm(x, self.W[layer_index].t())
        activation = wx + self.a[layer_index].expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    # Function to sample visible units' states and return probabilities and binary samples
    def sample_visible(self, y, layer_index):
        wy = torch.mm(y, self.W[layer_index])
        activation = wy + self.b[layer_index].expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    # Function to train the DBM using Contrastive Divergence
    def train(self, v0, vk, ph0, phk, layer_index):
        self.W[layer_index] += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b[layer_index] += torch.sum((v0 - vk), 0)
        self.a[layer_index] += torch.sum((ph0 - phk), 0)

# Setting up parameters and creating a Deep Boltzmann Machine (DBM)
nv = len(training_set[0])  # Number of visible units (movies)
nh1 = 10  # Number of units in the first hidden layer
nh2 = 10  # Number of units in the second hidden layer
nh3 = 10  # Number of units in the third hidden layer
batch_size = 100  # Batch size for training

# Defining the layer sizes for the DBM
layer_sizes = [nv, nh1, nh2, nh3]

# Creating a Deep Boltzmann Machine (DBM) instance
dbm = DBM(layer_sizes)

# Training loop
nb_epoch = 10  # Number of training epochs
for epoch in range(nb_epoch):
    train_loss = 0
    s = 0.0
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user : id_user + batch_size]
        v0 = training_set[id_user : id_user + batch_size]

        # Initialize hidden states and probabilities
        p_h0, h0 = dbm.sample_hidden(v0, layer_index=0)

        # Gibbs sampling for k steps
        for k in range(10):
            p_hk, hk = dbm.sample_hidden(vk, layer_index=0)
            p_vk, vk = dbm.sample_visible(hk, layer_index=0)
            vk[v0 < 0] = v0[v0 < 0]

        # Update hidden states and probabilities after k steps
        p_hk, hk = dbm.sample_hidden(vk, layer_index=0)

        # Update weights and biases using Contrastive Divergence
        dbm.train(v0, vk, p_h0, p_hk, layer_index=0)

        # Calculate and accumulate training loss
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1.0

    # Print training loss for the epoch
    print("Epoch:", epoch, "Loss:", train_loss / s)

# Testing the DBM
test_loss = 0
s = 0.0
for id_user in range(nb_users):
    v = training_set[id_user : id_user + 1]
    vt = test_set[id_user : id_user + 1]
    if len(vt[vt >= 0]) > 0:
        _, h = dbm.sample_hidden(v, layer_index=0)
        _, v = dbm.sample_visible(h, layer_index=0)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.0

# Print test loss
print("Test Loss:", test_loss / s)
