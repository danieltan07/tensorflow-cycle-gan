import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as tflayers
import numpy as np
from utils import forward

class ConvLayer():
    def __init__(self, num_filters=32, kernel_size=4, stride=2,
                 padding="SAME", normalizer=None, activation=None, weights_init=tf.truncated_normal_initializer):
        
        self.num_filters = num_filters
        self.padding = padding
        self.kernel_size=kernel_size
        self.stride = stride
        self.normalizer = normalizer
        self.activation = activation
        self.weights_init = weights_init

    def __call__(self,x):
        layer = slim.conv2d(x,self.num_filters,[self.kernel_size,self.kernel_size],[self.stride,self.stride],padding=self.padding ,weights_initializer=self.weights_init,
                                  activation_fn=self.activation)

        return layer

class ResBlock():
    def __init__(self, num_filters=32, kernel_size=4, stride=2,
                 padding_size=1,padding_type="REFLECT", normalizer=None, activation=None, weights_init=tf.truncated_normal_initializer):
        
        self.num_filters = num_filters
        self.padding_size = padding_size
        self.padding_type = padding_type
        self.kernel_size=kernel_size
        self.stride = stride
        self.normalizer = normalizer
        self.activation = activation
        self.weights_init = weights_init

    def __call__(self,x):
        layers = [
            PaddingLayer(self.padding_size,self.padding_type),
            ConvLayer(num_filters=self.num_filters,kernel_size=self.kernel_size,stride=self.stride,padding="VALID",
                      weights_init=self.weights_init,normalizer=self.normalizer,activation=self.activation),
            PaddingLayer(self.padding_size,self.padding_type),
            ConvLayer(num_filters=self.num_filters,kernel_size=self.kernel_size,stride=self.stride,padding="VALID",
                      weights_init=self.weights_init,normalizer=self.normalizer,activation=None)
        ]
        res = forward(layers)(x) + x
        return res

class PaddingLayer():
    def __init__(self, padding_size, padding_type):
        
        self.padding_size = padding_size
        self.padding_type = padding_type

    def __call__(self,x):
        p = int(self.padding_size)
        layer = tf.pad(x,[[0, 0], [p, p], [p, p], [0, 0]],self.padding_type)

        return layer

class ConvTransposeLayer():
    def __init__(self, num_outputs=32, kernel_size=3, stride=2,
                 padding="SAME", normalizer=None, activation=None):
        self.num_outputs = num_outputs
        self.padding = padding
        self.kernel_size=kernel_size
        self.stride = stride
        self.normalizer = normalizer
        self.activation = activation
        
    def __call__(self,x):
        
        return slim.conv2d_transpose(x,self.num_outputs,self.kernel_size,self.stride,padding=self.padding,
                                  normalizer_fn=self.normalizer,activation_fn=self.activation)

class FCLayer():
    def __init__(self, scope="fc_layer", size=None, dropout=None,
                 nonlinearity=None,reuse=True, normalizer=None):
        assert size, "Must specify layer size (num nodes)"
        self.scope = scope
        self.size = size
        self.dropout = dropout
        self.nonlinearity = nonlinearity
        self.reuse = reuse
        self.normalizer = normalizer

    def __call__(self,x):
        
        with tf.name_scope(self.scope):
            fc = slim.fully_connected(slim.flatten(x),self.size,activation_fn=self.nonlinearity,reuse=self.reuse,scope=self.scope)

            if self.dropout is not None:
                fc = tf.layers.dropout(inputs=fc,rate = self.dropout)
            if self.normalizer is not None:
                fc = self.normalizer(fc,scope=self.scope)
            
            return fc


            
##            if not (hasattr(self, 'w') and hasattr(self, 'b')):
##                std = tf.cast( (2 / x.get_shape()[1].value)**0.5, tf.float32)
##                w = tf.random_normal([x.get_shape()[1].value, self.size],stddev=std)
##                b = tf.zeros([self.size])
##
##                self.w = tf.Variable(w, trainable=True, name="weights")
##                self.b = tf.Variable(b, trainable=True, name="bias")
##
##            if self.dropout is not None:
##                self.w = tf.nn.dropout(self.w,self.dropout)
##
##            if self.nonlinearity is None:
##                return tf.matmul(x,self.w) + self.b
##
##            return self.nonlinearity(tf.matmul(x,self.w) + self.b)
        
