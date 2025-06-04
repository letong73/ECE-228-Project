import keras
import keras.models as models
from keras.layers import concatenate, Layer
from keras.layers import Reshape, Dense, Dropout, Activation, Flatten, RepeatVector
from keras.layers import Convolution1D, Convolution2D, MaxPooling1D, AveragePooling1D, AveragePooling2D, MaxPooling2D, ConvLSTM2D
from keras.layers import Permute
from keras.layers import BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input,Dense,Flatten,MaxPool2D,MaxPool1D,Activation,LeakyReLU,LSTM,BatchNormalization,Dropout,Conv2D,Conv1D,Lambda
from keras.constraints import max_norm, MinMaxNorm
from keras.utils import register_keras_serializable
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf


def freehand_v4(input_shape, dropout_rate=0.5, conv_1_filters=16, conv_1_kernel_size=4, conv_2_filters=32,
                conv_2_kernel_size=2, first_dense_units=256, second_dense_units=256, third_dense_units=128,
                activation_function="relu", classes=11):
    input = keras.Input(shape=input_shape)
    reshape = Reshape(input_shape + [1])(input)
    batch_normalization = BatchNormalization()(reshape)

    conv_1 = Convolution2D(conv_1_filters, conv_1_kernel_size, padding="same", activation=activation_function)(
        batch_normalization)
    max_pool = MaxPooling2D(padding='same')(conv_1)
    batch_normalization_2 = BatchNormalization()(max_pool)
    fc1 = Dense(first_dense_units, activation=activation_function)(batch_normalization_2)
    conv_2 = Convolution2D(conv_2_filters, conv_2_kernel_size, padding="same", activation=activation_function)(fc1)
    batch_normalization_3 = BatchNormalization()(conv_2)
    max_pool = MaxPooling2D(padding='same')(batch_normalization_3)

    out_flatten = Flatten()(max_pool)
    dr = Dropout(dropout_rate)(out_flatten)
    fc2 = Dense(second_dense_units, activation=activation_function)(dr)
    batch_normalization_4 = BatchNormalization()(fc2)
    fc3 = Dense(third_dense_units, activation=activation_function)(batch_normalization_4)
    output = Dense(classes, name="output", activation="softmax")(fc3)

    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def freehand_scrnn_v4(input_shape, dropout_rate=0.5, conv_1_filters=16, conv_1_kernel_size=4, conv_2_filters=32,
                conv_2_kernel_size=2, first_dense_units=256, second_dense_units=256, third_dense_units=128,
                activation_function="relu", classes=11):
    input = keras.Input(shape=input_shape)
    input = Permute((2, 1))(input)
    reshape = Reshape(input_shape + [1])(input)
    batch_normalization = BatchNormalization()(reshape)

    conv_1 = Convolution2D(conv_1_filters, conv_1_kernel_size, padding="same", activation=activation_function)(
        batch_normalization)
    max_pool = MaxPool2D(padding='same')(conv_1)
    batch_normalization_2 = BatchNormalization()(max_pool)
    fc1 = Dense(first_dense_units, activation=activation_function)(batch_normalization_2)
    conv_2 = Convolution2D(conv_2_filters, conv_2_kernel_size, padding="same", activation=activation_function)(fc1)
    batch_normalization_3 = BatchNormalization()(conv_2)
    max_pool = MaxPool2D(padding='same')(batch_normalization_3)

    out_flatten = Flatten()(max_pool)
    dr = Dropout(dropout_rate)(out_flatten)
    fc2 = Dense(second_dense_units, activation=activation_function)(dr)
    batch_normalization_4 = BatchNormalization()(fc2)
    fc3 = Dense(third_dense_units, activation=activation_function)(batch_normalization_4)
    
    seq = RepeatVector(1)(fc3)  
    lstm1 = LSTM(128, return_sequences=True,
                 activation=activation_function,
                 unroll=True)(seq)
    dr2   = Dropout(dropout_rate)(lstm1)
    lstm2 = LSTM(128, return_sequences=True,
                 activation=activation_function,
                 unroll=True)(dr2)
    dr3   = Dropout(dropout_rate)(lstm2)
    flat  = Flatten()(dr3)
    
    output = Dense(classes, name="output", activation="softmax")(flat)

    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


 
@register_keras_serializable(package="Custom", name="TMaxAvgPooling1D")
class TMaxAvgPooling1D(Layer):
    def __init__(self, pool_size=3, strides=None, padding='valid', **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.strides   = strides or pool_size
        self.padding   = padding

        # built-in layers for max/avg
        self._maxpool = MaxPool1D(pool_size=self.pool_size,
                                  strides=self.strides,
                                  padding=self.padding)
        self._avgpool = AveragePooling1D(pool_size=self.pool_size,
                                         strides=self.strides,
                                         padding=self.padding)
        # one T per channel, clamped to [0,1]
        self._constraint = MinMaxNorm(min_value=0., max_value=1.)

    def build(self, input_shape):
        channels = input_shape[-1]
        self.T = self.add_weight(
            name='T',
            shape=(channels,),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            constraint=self._constraint
        )
        super().build(input_shape)

    def call(self, inputs):
        m = self._maxpool(inputs)
        a = self._avgpool(inputs)
        # broadcast T across batch & time dims
        T = tf.reshape(self.T, (1, 1, -1))
        return T * m + (1 - T) * a

    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size,
            "strides":   self.strides,
            "padding":   self.padding
        })
        return config

@register_keras_serializable(package="Custom", name="TMaxAvgPooling2D")
class TMaxAvgPooling2D(Layer):
    def __init__(self, pool_size=(2,2), strides=None, padding='valid', data_format='channels_last', **kwargs):
        super().__init__(**kwargs)
        self.pool_size   = pool_size
        self.strides     = strides or pool_size
        self.padding     = padding
        self.data_format = data_format

        # built-in layers for max/avg
        self._maxpool = MaxPool2D(pool_size=self.pool_size,
                                  strides=self.strides,
                                  padding=self.padding,
                                  data_format=self.data_format)
        self._avgpool = AveragePooling2D(pool_size=self.pool_size,
                                         strides=self.strides,
                                         padding=self.padding,
                                         data_format=self.data_format)
        # one T per channel, clamped to [0,1]
        self._constraint = MinMaxNorm(min_value=0., max_value=1.)

    def build(self, input_shape):
        # input_shape: (batch, H, W, C) if channels_last
        channels = input_shape[-1]
        self.T = self.add_weight(
            name='T',
            shape=(channels,),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            constraint=self._constraint
        )
        super().build(input_shape)

    def call(self, inputs):
        m = self._maxpool(inputs)
        a = self._avgpool(inputs)
        # broadcast T across batch, height & width dims
        # shape = (1, 1, 1, channels)
        T = tf.reshape(self.T, (1, 1, 1, -1))
        return T * m + (1. - T) * a

    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size":   self.pool_size,
            "strides":     self.strides,
            "padding":     self.padding,
            "data_format": self.data_format
        })
        return config

def base_scrnn(input_shape,classes = 11):
	
	inputs = Permute((2, 1))(Input(input_shape))
    #inputs = Permute((2, 1))(Input(input_shape))
	l = BatchNormalization()(inputs)
	# l = Lambda(lambda t: K.expand_dims(t, -2))(l)
	l = Conv1D(filters=128,kernel_size=5,activation='relu')(l)
	l = MaxPool1D(3)(l)
	l = Conv1D(filters=128,kernel_size=5,activation='relu')(l)
	# l = Lambda(lambda t: K.squeeze(t, -2))(l)
	l = LSTM(128,return_sequences=True,activation='relu',unroll=True)(l)    
	l = LSTM(128,return_sequences=True,activation='relu',unroll=True)(l)
	l = Dropout(0.8)(l)     
	l = Flatten()(l)    
	outputs = Dense(11,activation='softmax',kernel_constraint = max_norm(2.))(l)

	model = Model(inputs,outputs)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001) ,metrics=['accuracy'])
	return model

def tmaxavg_scrnn(input_shape,classes = 11):
	
	inputs = Permute((2, 1))(Input(input_shape))
    #inputs = Permute((2, 1))(Input(input_shape))
	l = BatchNormalization()(inputs)
	# l = Lambda(lambda t: K.expand_dims(t, -2))(l)
	l = Conv1D(filters=128,kernel_size=5,activation='relu')(l)
	l =TMaxAvgPooling1D(pool_size = 3)(l)
	l = Conv1D(filters=128,kernel_size=5,activation='relu')(l)
	# l = Lambda(lambda t: K.squeeze(t, -2))(l)
	l = LSTM(128,return_sequences=True,activation='relu',unroll=True)(l)    
	l = LSTM(128,return_sequences=True,activation='relu',unroll=True)(l)
	l = Dropout(0.8)(l)     
	l = Flatten()(l)    
	outputs = Dense(11,activation='softmax',kernel_constraint = max_norm(2.))(l)

	model = Model(inputs,outputs)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001) ,metrics=['accuracy'])
	return model

def freehand_tmaxavg(input_shape, dropout_rate=0.5, conv_1_filters=16, conv_1_kernel_size=4, conv_2_filters=32,
                conv_2_kernel_size=2, first_dense_units=256, second_dense_units=256, third_dense_units=128,
                activation_function="relu", classes=11):
    input = keras.Input(shape=input_shape)
    reshape = Reshape(input_shape + [1])(input)
    batch_normalization = BatchNormalization()(reshape)

    conv_1 = Convolution2D(conv_1_filters, conv_1_kernel_size, padding="same", activation=activation_function)(
        batch_normalization)
    T_max_pool = TMaxAvgPooling2D(pool_size = (2,2))(conv_1)
    batch_normalization_2 = BatchNormalization()(T_max_pool)
    fc1 = Dense(first_dense_units, activation=activation_function)(batch_normalization_2)
    conv_2 = Convolution2D(conv_2_filters, conv_2_kernel_size, padding="same", activation=activation_function)(fc1)
    batch_normalization_3 = BatchNormalization()(conv_2)
    T_max_pool = TMaxAvgPooling2D(pool_size = (2,2),padding = 'same')(batch_normalization_3)

    out_flatten = Flatten()(T_max_pool)
    dr = Dropout(dropout_rate)(out_flatten)
    fc2 = Dense(second_dense_units, activation=activation_function)(dr)
    batch_normalization_4 = BatchNormalization()(fc2)
    fc3 = Dense(third_dense_units, activation=activation_function)(batch_normalization_4)
    output = Dense(classes, name="output", activation="softmax")(fc3)

    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model