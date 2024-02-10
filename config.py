'''
Semesterproject Fluorescence Spectroscopy
'''
import os

# Autoencoder
chunk_size = 49920 # data preparation

batch_size = 128
epochs = 300
optimizer = 'adam'
loss = 'mse'

# Checkpoint
# Set your desired path
path = f"results/{batch_size}batchsize_{epochs}epochs_{optimizer.__class__.__name__}optimizer_{loss}loss/"

# Check if the directory exists
if not os.path.exists(path):
    # If it doesn't exist, create the directory
    os.makedirs(path)
    os.makedirs(path + 'Autoencoder/weights')
    os.makedirs(path + 'FC/weights')
    os.makedirs(path + 'FC/FC_dataset/')
    print(f"Directory '{path}' created.")
else:
    print(f"Directory '{path}' already exists. (check)")
