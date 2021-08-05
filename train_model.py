import model
import numpy as np
from keras.optimizers import Adam

Tx = 5511               
Ty = 1375               
n_freq = 101            

X_train = np.load('XY_train/X.npy')
Y_train = np.load('XY_train/Y.npy')

model = model(input_shape = (Tx, n_freq))
# model.summary()

opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

model.fit(X_train, Y_train, batch_size = 16, epochs=4)
model.save('models/pre_trained_model', save_format='h5')


