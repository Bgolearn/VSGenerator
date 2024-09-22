import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer,Add
from tensorflow.keras.models import Model
import numpy as np
import keras.backend as K

class SelfAttentionBlock(Layer):
    def __init__(self, units, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.units = units # dim of KQV

    def build(self, input_shape):
        self.W_q = self.add_weight(name="W_q", shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=True)
        self.W_k = self.add_weight(name="W_k", shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=True)
        self.W_v = self.add_weight(name="W_v", shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=True)
        super(SelfAttentionBlock, self).build(input_shape)

    def call(self, x):
        q = K.dot(x, self.W_q)
        k = K.dot(x, self.W_k)
        v = K.dot(x, self.W_v)

        attention_score = K.dot(q, K.transpose(k)) / K.sqrt(K.cast(self.units, dtype=K.floatx()))
        attention_weights = K.softmax(attention_score)
        output = K.dot(attention_weights, v)

        return output

# a function that samples from a normal distribution
def sampling(z_mean, z_log_var,num):
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_uniform(shape=(num, dim), minval=-1.0, maxval=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# model architecture
def build_vsg(input_dim, y_dim, latent_dim, gen_num):
    # Encoder
    _input = Input(shape=(input_dim,))
    y = Input(shape=(y_dim,))

    x = Dense(16, activation='relu')(_input)
    x = Dense(32, activation='relu')(x)
    h = SelfAttentionBlock(units=16)(x)  

    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = sampling(z_mean, z_log_var,gen_num)
    y_expanded = tf.tile(y, [int(gen_num), 1])
    z_cond = tf.concat([z, y_expanded], axis=1)

    # Decoder
    h_decoded = Dense(32, activation='relu')(z_cond)
    h_decoded = Dense(latent_dim+y_dim, activation='relu')(h_decoded)
    h_decoded =  Add()([h_decoded, z_cond])
    _h_decoded = Dense(latent_dim+y_dim, activation='relu')(h_decoded)
    h_decoded =  Add()([_h_decoded,h_decoded])
    x_decoded_mean = Dense(input_dim,)(h_decoded)

    # Predictor 
    y_pre = Dense(input_dim, activation='relu')(x_decoded_mean)
    y_pre = Dense(input_dim, activation='relu')(y_pre)
    y_pre =  Add()([y_pre, x_decoded_mean])
    y_pre = Dense(y_dim,)(y_pre)
    
    # do not trainable
    vsg_loss = tf.keras.losses.mse(y, y)  
    
    # Create vsg model
    vsg = Model(inputs=[_input, y], outputs=x_decoded_mean)

    # Add the custom loss to the model
    vsg.add_loss(vsg_loss)
    vsg.compile(optimizer='adam')

    return vsg


"""
# build model 
input_dim = 4  # dim of the input
y_dim = 1 # dim of the target
latent_dim = 15  # dim of the latent variables
gen_num = 100

vsg = build_svae(input_dim, y_dim, latent_dim,gen_num)
"""
