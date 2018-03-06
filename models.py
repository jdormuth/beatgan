from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Lambda
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import numpy as np 


def lightweight_generator():
    model_input = Input(shape=(128,))
    hidden = Reshape((1,1,128))(model_input)
    hidden = Conv2DTranspose(512,(1,1), strides = (1,1), name = 'h0')(hidden)
    
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)
    hidden_1 = Reshape((2,1,256))(hidden)
    hidden_1 = Conv2DTranspose(256, (2,1), strides = (2,1), name = 'h1')(hidden_1)
    hidden_1 = BatchNormalization()(hidden_1)
    hidden_1 = Activation('relu')(hidden_1)
    hidden_2 = Conv2DTranspose(128, (2,1), strides =(2,1), name = 'h2')(hidden_1)
    hidden_2 = BatchNormalization()(hidden_2)
    hidden_2 = Activation('relu')(hidden_2)
    hidden_3 = Conv2DTranspose(128, (2,1), strides = (2,1), name = 'h3')(hidden_2)
    hidden_3 = BatchNormalization()(hidden_3)
    hidden_3 = Activation('relu')(hidden_3)
    hidden_4 = Conv2DTranspose(64, (2,1), strides = (2,1), name = 'h4')(hidden_3)
    hidden_4 = BatchNormalization()(hidden_4)
    hidden_4 = Activation('relu')(hidden_4)
    hidden_5 = Conv2DTranspose(64, (3,1), strides= (3,1), name='h5')(hidden_4)
    hidden_5 = BatchNormalization()(hidden_5)
    hidden_5 = Activation('relu')(hidden_5)
    
    hidden_6 = Conv2DTranspose(8, (1,5), strides=(1,1), name = 'h6')(hidden_5)
    hidden_6 = BatchNormalization()(hidden_6)
    hidden_6 = Activation('relu')(hidden_6)
    hidden_7 = Conv2DTranspose(1, (1,6), strides = (1,1), name='h7')(hidden_6)
    

    output = Activation('tanh')(hidden_7)
    model = Model(inputs=model_input, outputs=output)
    return model

def lightweight_discriminator():
    img_shape = (96, 10, 1)
    test = np.ones((96,84,1))
    test = np.reshape(test,(1,96,84,1))

    model = Sequential()
    model.add(Conv2D(64, kernel_size=[1,12], strides=[1,12], input_shape=img_shape, padding="VALID"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(64, kernel_size=[1,7], strides=[1,7],padding="VALID"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(64, kernel_size=[2,1], strides=[2,1], padding="VALID"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(64, kernel_size=[2,1], strides=[2,1], padding="VALID"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(128, kernel_size=[4,1], strides=[2,1], padding="VALID"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(256, kernel_size=[3,1], strides=[2,1], padding="VALID"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=.2))
    model.add(Dense(1))
    return model

def super_lightweight_generator():
    model_input = Input(shape=(128,))
    hidden = Reshape((1,1,128))(model_input)
    hidden = Conv2DTranspose(256,(1,1), strides = (1,1), name = 'h0')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('relu')(hidden)
    hidden_1 = Reshape((2,1,128))(hidden)
    hidden_1 = Conv2DTranspose(128, (2,1), strides = (2,1), name = 'h1')(hidden_1)
    hidden_1 = BatchNormalization()(hidden_1)
    hidden_1 = Activation('relu')(hidden_1)
    hidden_2 = Conv2DTranspose(64, (2,1), strides =(2,1), name = 'h2')(hidden_1)
    hidden_2 = BatchNormalization()(hidden_2)
    hidden_2 = Activation('relu')(hidden_2)
    hidden_3 = Conv2DTranspose(64, (2,1), strides = (2,1), name = 'h3')(hidden_2)
    hidden_3 = BatchNormalization()(hidden_3)
    hidden_3 = Activation('relu')(hidden_3)
    hidden_4 = Conv2DTranspose(32, (2,1), strides = (2,1), name = 'h4')(hidden_3)
    hidden_4 = BatchNormalization()(hidden_4)
    hidden_4 = Activation('relu')(hidden_4)
    hidden_5 = Conv2DTranspose(32, (3,1), strides= (3,1), name='h5')(hidden_4)
    hidden_5 = BatchNormalization()(hidden_5)
    hidden_5 = Activation('relu')(hidden_5)
    hidden_6 = Conv2DTranspose(4, (1,7), strides=(1,1), name = 'h6')(hidden_5)
    hidden_6 = BatchNormalization()(hidden_6)
    hidden_6 = Activation('relu')(hidden_6)
    hidden_7 = Conv2DTranspose(1, (1,12), strides = (1,12), name='h7')(hidden_6)

    output = Activation('tanh')(hidden_7)
    model = Model(inputs=model_input, outputs=output)
    return model

def super_lightweight_discriminator():
    img_shape = (96, 10, 1)
    test = np.ones((96,10,1))
    test = np.reshape(test,(1,96,10,1))

    model = Sequential()
    model.add(Conv2D(32, kernel_size=[1,2], strides=[1,2], input_shape=img_shape, padding="VALID"))
    
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(32, kernel_size=[1,3], strides=[1,3],padding="VALID"))
    
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(32, kernel_size=[2,1], strides=[2,1], padding="VALID"))
    
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(32, kernel_size=[2,1], strides=[2,1], padding="VALID"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(64, kernel_size=[4,1], strides=[2,1], padding="VALID"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(128, kernel_size=[3,1], strides=[2,1], padding="VALID"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=.2))
    model.add(Dense(1))
    return model    


# model_input = Input(shape=(128,))
        # hidden = Reshape((1,1,128))(model_input)
        # hidden = Conv2DTranspose(1024,(1,1), strides = (1,1), name = 'h0')(hidden)
        # hidden = BatchNormalization()(hidden)
        # hidden = Activation('relu')(hidden)
        # hidden_1 = Reshape((2,1,512))(hidden)
        # hidden_1 = Conv2DTranspose(512, (2,1), strides = (2,1), name = 'h1')(hidden_1)
        # hidden_1 = BatchNormalization()(hidden_1)
        # hidden_1 = Activation('relu')(hidden_1)
        # hidden_2 = Conv2DTranspose(256, (2,1), strides =(2,1), name = 'h2')(hidden_1)
        # hidden_2 = BatchNormalization()(hidden_2)
        # hidden_2 = Activation('relu')(hidden_2)
        # hidden_3 = Conv2DTranspose(256, (2,1), strides = (2,1), name = 'h3')(hidden_2)
        # hidden_3 = BatchNormalization()(hidden_3)
        # hidden_3 = Activation('relu')(hidden_3)
        # hidden_4 = Conv2DTranspose(128, (2,1), strides = (2,1), name = 'h4')(hidden_3)
        # hidden_4 = BatchNormalization()(hidden_4)
        # hidden_4 = Activation('relu')(hidden_4)
        # hidden_5 = Conv2DTranspose(128, (3,1), strides= (3,1), name='h5')(hidden_4)
        # hidden_5 = BatchNormalization()(hidden_5)
        # hidden_5 = Activation('relu')(hidden_5)
        # hidden_6 = Conv2DTranspose(16, (1,7), strides=(1,1), name = 'h6')(hidden_5)
        
        # hidden_6 = BatchNormalization()(hidden_6)
        # hidden_6 = Activation('relu')(hidden_6)
        # hidden_7 = Conv2DTranspose(1, (1,12), strides = (1,12), name='h7')(hidden_6)
        
        # output = Activation('tanh')(hidden_7)
        # model = Model(inputs=model_input, outputs=output)
# model = Sequential()
        # model.add(Conv2D(128, kernel_size=[1,12], strides=[1,12], input_shape=img_shape, padding="VALID"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(128, kernel_size=[1,7], strides=[1,7],padding="VALID"))
        # model.add(LeakyReLU(alpha=.2))
        # model.add(Conv2D(128, kernel_size=[2,1], strides=[2,1], padding="VALID"))
        # model.add(LeakyReLU(alpha=.2))
        # model.add(Conv2D(128, kernel_size=[2,1], strides=[2,1], padding="VALID"))
        # model.add(LeakyReLU(alpha=.2))
        # model.add(Conv2D(256, kernel_size=[4,1], strides=[2,1], padding="VALID"))
        # model.add(LeakyReLU(alpha=.2))
        # model.add(Conv2D(512, kernel_size=[3,1], strides=[2,1], padding="VALID"))
        # model.add(LeakyReLU(alpha=.2))
        # model.add(Flatten())
        # model.add(Dense(1024))
        # model.add(LeakyReLU(alpha=.2))
        # model.add(Dense(1))

        # model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        # model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        # model.add(Flatten())


