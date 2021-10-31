import tensorflow as tf
from keras.models import load_model
from keras import backend as K
import numpy as np

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


tf.compat.v1.disable_v2_behavior()
model = load_model('./models/tr_model.h5')
X_dev = np.load("X_dev.npy")
Y_dev = np.load("Y_dev.npy")
loss, accuracy, f1_score, precision, recall = model.evaluate(X_dev, Y_dev)
print("accuracy: ", accuracy)
print("precision:", precision)
print("recall:", recall)
print("f1_score:", f1_score)



