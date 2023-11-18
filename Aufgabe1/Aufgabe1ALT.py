#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.005
session = tf.compat.v1.Session(config=config)

""" config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
session = tf.Session(config=config) """

""" gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e) """

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)


# In[12]:


print('tensorflow version: ',tf.__version__) # need <2.11 on win
print('available gpus: ', tf.config.list_physical_devices('GPU')) # gpu?


# In[13]:


'''
Data preparation:
'''

data = pd.read_csv('XFEL_KW0_Results_2.csv', names=None)

# find constant columns, else normalization wouldn't work
data = data.loc[:, (data != data.iloc[0]).any()] 

# label input 0-6, output 0-4 just for clarity
num_inputs = 7
num_outputs = 5
data.columns = np.concatenate(
    (["Input %s" % i for i in range(num_inputs)], ["Output %s" % i for i in range(num_outputs)])
)


# In[14]:


# normalization to: mean=0, std=1
normalized_data = (data-data.mean())/data.std()

# TODO: k-fold cross-validation
train, test = train_test_split(normalized_data, test_size=0.2, random_state=42, shuffle=True)
#train, validation = train_test_split(train, test_size=0.2, random_state=42, shuffle=True)

# numpy indexing [columns_start : columns_end, rows_start : rows_end]
x_train, y_train = train.iloc[:,:num_inputs], train.iloc[:,num_inputs:]
#x_val, y_val = validation.iloc[:,:num_inputs], validation.iloc[:,num_inputs:]
x_test, y_test = test.iloc[:,:num_inputs], test.iloc[:,num_inputs:]
x_train

print('data preparation done')


# In[16]:


# Model creation
def create_model(hidden_depth = 1, lr=0.01, actv_fct='relu'):
    model = keras.Sequential([
        keras.Input(shape=(num_inputs)),
        *[
            keras.layers.Dense(16, activation=actv_fct) 
            for _ in range(hidden_depth)
        ],
        keras.layers.Dense(num_outputs)
    ])
    
    print(model.summary())
    print('hidden depth:', hidden_depth, 'learning rate: ', lr, ' activation function: ', actv_fct)

    optimizer = keras.optimizers.SGD(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

print('defined model creation method')


# In[17]:


# Gridsearch with k-fold cross validation

param_grid = {
    'hidden_depth': [i for i in range(1,8)],
    'lr': np.arange(0.025,0.151,0.025),
    'actv_fct': ['relu'],
    'batch_size': [700],
    'epochs': [20],
}
loss = keras.losses.MeanSquaredError()

model = KerasRegressor(build_fn=create_model)

# Specify the number of folds for cross-validation
n_splits = 3

# Create a cross-validation object (KFold in this case)
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Perform grid search with cross-validation
print('starting grid-search')
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error')
grid_result = grid.fit(x_train, y_train)

# Print the best hyperparameters and results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Save the results to file
joblib.dumb(grid_result, 'hd1-7_lr05-15_025.pkl')

# In[20]:


# Extract the results from the grid search
results = grid_result.cv_results_
hidden_depths = results['param_hidden_depth']
learning_rates = results['param_lr']
mean_test_scores = -results['mean_test_score']  # Negate for MSE

# Create a pivot table for the heatmap
pivot_table = pd.pivot_table(pd.DataFrame({'Hidden Depth': hidden_depths, 'Learning Rate': learning_rates, 'Mean Test Score': mean_test_scores}),
                            values='Mean Test Score', index='Hidden Depth', columns='Learning Rate')

# Create a heatmap using Seaborn
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='coolwarm', cbar=True)
plt.title('Grid Search Results for Hidden Depth and Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Hidden Depth')

plt.savefig('results_oder_sowas.png')


# In[85]:


""" model = tf.keras.Sequential([
    tf.keras.Input(shape=7),
    layers.Dense(16, activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(4),
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(0.1),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['accuracy']
)

history = model.fit(
    xtrain, ytrain,
    batch_size=200,
    epochs=8,
    validation_data = (xtest, ytest),
) """


# In[86]:


""" plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show() """


# In[84]:


""" # Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(xtest, ytest, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(xtest[:3])
print("example predictions:\n", predictions) """

