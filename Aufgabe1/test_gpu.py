import tensorflow as tf

print('tensorflow version: ',tf.__version__) # need <2.11 on win
print('available gpus: ', tf.config.list_physical_devices('GPU')) # gpu?
