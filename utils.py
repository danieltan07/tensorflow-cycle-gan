import functools
from functools import partial
import tensorflow as tf
import numpy as np
import h5py
from scipy.misc import imread, imsave, imresize
from os import listdir
from os.path import isfile, join, isdir


def compose(func_1, func_2, unpack=False):
    """
    compose(func_1, func_2, unpack=False) -> function
    
    The function returned by compose is a composition of func_1 and func_2.
    That is, compose(func_1, func_2)(5) == func_1(func_2(5))
    """
    if not callable(func_1):
        raise TypeError("First argument to compose must be callable")
    if not callable(func_2):
        raise TypeError("Second argument to compose must be callable")
    
    if unpack:
        def composition(*args, **kwargs):
            return func_1(*func_2(*args, **kwargs))
    else:
        def composition(*args, **kwargs):
            return func_1(func_2(*args, **kwargs))
    return composition

def compose_all(*args):
    """
        Util for multiple function composition
        i.e. composed = composeAll([f, g, h])
             composed(x) # == f(g(h(x)))
    """
    return partial(functools.reduce, compose)(*args)

def forward(layers):
    return compose_all(reversed(layers))

def leakyrelu(x, leak=0.2, name="LeakyReLu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def instance_norm(x,eps=1e-7,name="instance_norm"):
    with tf.variable_scope(name):
        mean, std = tf.nn.moments(x, axes=(1,2), keep_dims=True)
        
        return (x - mean)/(std + eps)

def tohdf(path,filename,img_dim,outpath=""):
    imgs = [f for f in listdir(path) if isfile(join(path, f))]

    img_dataset = np.zeros((len(imgs),*img_dim))

    for i in range(len(imgs)):
        img_dataset[i,:] = imread(path+imgs[i])

    # Create the HDF5 file
    f = h5py.File(outpath + filename + ".h5", 'w')

    # Create the image and palette dataspaces
    dset = f.create_dataset(filename, data=img_dataset)

    dset.attrs['CLASS'] = 'IMAGE'
    dset.attrs['IMAGE_VERSION'] = '1.2'
    dset.attrs['IMAGE_SUBCLASS'] =  'IMAGE_TRUECOLOR'
    dset.attrs['IMAGE_MINMAXRANGE'] = np.array([0,255], dtype=np.uint8)

    f.close()
