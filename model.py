from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, Conv1D
from keras.layers import GRU, BatchNormalization, Reshape
from keras.optimizers import Adam
import tensorflow as tf
from evaluate import f1_m, precision_m, recall_m
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)



def model(settings):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    
    X_input = Input(shape = (settings.Tx , settings.n_freq))
    
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)                                 
    X = BatchNormalization()(X)                                 
    X = Activation('relu')(X)                                 
    X = Dropout(0.8)(X)                                

    X = GRU(units=128, return_sequences=True, reset_after=True)(X)                                 
    X = Dropout(0.8)(X)                                 
    X = BatchNormalization()(X)                                 
    
    X = GRU(units=128, return_sequences=True, reset_after=True)(X)                           
    X = Dropout(0.8)(X)                                 
    X = BatchNormalization()(X)                               
    X = Dropout(0.8)(X)                               
    
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model

def load():
    '''Loads a pretrained model.'''
    tf.compat.v1.disable_v2_behavior()
    model = load_model('./models/tr_model.h5')
    return model


def train_model(model,data):
    '''Trains a model with the input data'''

    opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy',f1_m,precision_m, recall_m])
    model.fit(data.X_train, data.Y_train, batch_size=64, epochs=32)
    model.save('./models/trained_model.h5')




