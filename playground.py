import tensorflow as tf
import numpy as np
import tensorflow.keras.models as models
from matplotlib import pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train,x_labels),(y_test,y_labels) = mnist.load_data()
x_train, y_test = x_train/255.0, y_test/255.0
for i in range(9):
    plt.subplot(330+1+i)
    plt.imshow(x_train[i])
plt.show()