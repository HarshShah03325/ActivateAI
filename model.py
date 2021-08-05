from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
import tensorflow as tf
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

def load_model():
    '''Loads a pretrained model.'''

    model = load_model('./models/tr_model.h5')
    return model


def train_model(model,data):
    '''Trains a model with the input data'''

    opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(data.X, data.Y, batch_size=32, epochs=5)
    model.save('./models/trained_model.h5')




