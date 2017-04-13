from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc as sp
import time
import IPython.display
import tqdm # making loops prettier
import ipywidgets as widgets
import math
from layers import ConvLayer, PaddingLayer, ResBlock, ConvTransposeLayer
from ipywidgets import interact, interactive, fixed

from utils import leakyrelu, forward, instance_norm
import tensorflow.contrib.slim as slim

class CycleGAN():   
    DEFAULTS = {}
    def __init__(self,dataA,dataB, hyperparams = {}, img_dim=(256,256,3)):
        self.__dict__.update(CycleGAN.DEFAULTS, **hyperparams)
        self.dataA = dataA
        self.dataB = dataB
        self.img_dim = img_dim
        self.build_network()

    def build_network(self):
        self.real_label_A = 0.9 #0.9 for smoothing
        self.real_label_B = 0.9 #0.9 for smoothing
        self.fake_label_A = 0 
        self.fake_label_B = 0 
        
        self.disc_A = self.discriminator()
        self.gen_A = self.generator()
        self.disc_B = self.discriminator()
        self.gen_B = self.generator()

    def mse_loss(self, output, target):
        with tf.name_scope("MSE_loss"):
            return tf.reduce_sum(tf.reduce_mean((output - target)**2,[1,2,3]))
    def abs_loss(self, output,target):
        with tf.name_scope("Abs_loss"):
            return tf.reduce_sum(tf.reduce_mean(tf.abs(output - target),[1,2,3]))

    def fDx(self,netD, netG, real, fake, real_label,fake_label,scope_name="A"):
        # Real  log(D_A(B))
        with tf.variable_scope("Disc_"+scope_name,reuse=True):
            real_output = netD(real)
        errD_real = self.mse_loss(real_output,real_label)

        # Fake  + log(1 - D_A(G_A(A)))
        with tf.variable_scope("Disc_"+scope_name,reuse=True):
            fake_output = netD(fake)
        errD_fake = self.mse_loss(fake_output, fake_label)

        loss = (errD_real + errD_fake) / 2.0

        params = tf.trainable_variables()
        D_params = [i for i in params if "Disc_"+scope_name in i.name]

        return loss, D_params

    
    def fGx(self,netG, netD, netE,real,real2,real_label, lambda1, lambda2,scope_name="A"):
        #GAN loss : D_A(G_A(A))
        with tf.variable_scope("Gen_"+scope_name,reuse=True):
            fake = netG(real)
        with tf.variable_scope("Disc_"+scope_name,reuse=True):
            output = netD(fake)

        errG = self.mse_loss(output,real_label)

        if scope_name == "A":
            other_name = "B"
        else:
            other_name = "A"

        #forward cycle loss     
        with tf.variable_scope("Gen_"+other_name,reuse=True):
            rec = netE(fake)

        errRec = self.abs_loss(rec,real) * lambda1
        
        #backward cycle loss
        with tf.variable_scope("Gen_"+other_name,reuse=True):
            fake2 = netE(real2)
        with tf.variable_scope("Gen_"+scope_name,reuse=True):
            rec2 = netG(fake2)

        errAdapt = self.abs_loss(rec2,real2) * lambda2        


        params = tf.trainable_variables()
        G_params = [i for i in params if "Gen_"+scope_name in i.name]

        loss = errRec + errG + errAdapt

        return loss, G_params, fake, rec, errRec, errG, errAdapt


    def create_image(self,im):
        #scale the pixel values from [-1,1] to [0,1]
        return (im + 1) / 2
    
    def generator(self):
        f = 7
        p = (f-1)/2
        gen = [
            PaddingLayer(p,"reflect"),
            ConvLayer(num_filters=32,kernel_size=7,stride=1,padding="VALID",
                      weights_init=tf.truncated_normal_initializer(stddev=self.weight_stdev),
                      normalizer=self.normalizer,activation=tf.nn.relu),
            ConvLayer(num_filters=64,kernel_size=3,stride=2,padding="SAME",
                      weights_init=tf.truncated_normal_initializer(stddev=self.weight_stdev),
                      normalizer=self.normalizer,activation=tf.nn.relu),
            ConvLayer(num_filters=128,kernel_size=3,stride=2,padding="SAME",
                      weights_init=tf.truncated_normal_initializer(stddev=self.weight_stdev),
                      normalizer=self.normalizer,activation=tf.nn.relu)
        ]

        for i in range(self.n_res_blocks):
            gen.append(ResBlock(num_filters=128,kernel_size=3,stride=1,padding_size=1,padding_type="reflect",
                      weights_init=tf.truncated_normal_initializer(stddev=self.weight_stdev),
                                normalizer=self.normalizer,activation=tf.nn.relu))
        gen2 = [
            ConvTransposeLayer(num_outputs=64, kernel_size=3, stride=2,padding="SAME",
                               normalizer=self.normalizer, activation=tf.nn.relu),
            ConvTransposeLayer(num_outputs=32, kernel_size=3, stride=2,padding="SAME",
                               normalizer=self.normalizer, activation=tf.nn.relu),
            PaddingLayer(p,"reflect"),
            ConvLayer(num_filters=3,kernel_size=7,stride=1,padding="VALID",
                      weights_init=tf.truncated_normal_initializer(stddev=self.weight_stdev),
                      normalizer=self.normalizer,activation=tf.tanh)
            
        ]
        gen = gen + gen2
        return forward(gen)
        
        
                
    def discriminator(self):
        disc = [
            ConvLayer(num_filters=64,kernel_size=4,stride=2,padding="SAME",
                      weights_init=tf.truncated_normal_initializer(stddev=self.weight_stdev),activation=leakyrelu),
            ConvLayer(num_filters=128,kernel_size=4,stride=2,padding="SAME",
                      weights_init=tf.truncated_normal_initializer(stddev=self.weight_stdev),normalizer=slim.batch_norm,activation=leakyrelu),
            ConvLayer(num_filters=256,kernel_size=4,stride=2,padding="SAME",
                      weights_init=tf.truncated_normal_initializer(stddev=self.weight_stdev),normalizer=slim.batch_norm,activation=leakyrelu),
            ConvLayer(num_filters=512,kernel_size=4,stride=2,padding="SAME",
                      weights_init=tf.truncated_normal_initializer(stddev=self.weight_stdev),normalizer=slim.batch_norm,activation=leakyrelu),
            ConvLayer(num_filters=1,kernel_size=4,stride=1,padding="SAME",
                      weights_init=tf.truncated_normal_initializer(stddev=self.weight_stdev),normalizer=slim.batch_norm,activation=leakyrelu)
        ]

        return forward(disc)

    def plot_network_output(self,real_A,fake_B,rec_A,real_B,fake_A,rec_B):
        """
        Just plots the output of the network, error, reconstructions, etc
        """
        fig, ax = plt.subplots(nrows=2,ncols=3, figsize=(18,6))
        ax[(0,0)].imshow(self.create_image(np.squeeze(real_A)), cmap=plt.cm.gray, interpolation='nearest')
        ax[(0,1)].imshow(self.create_image(np.squeeze(fake_B)), cmap=plt.cm.gray, interpolation='nearest')
        ax[(0,2)].imshow(self.create_image(np.squeeze(rec_A)), cmap=plt.cm.gray, interpolation='nearest')
        ax[(0,0)].axis('off')
        ax[(0,1)].axis('off')
        ax[(0,2)].axis('off')

        ax[(1,0)].imshow(self.create_image(np.squeeze(real_B)), cmap=plt.cm.gray, interpolation='nearest')
        ax[(1,1)].imshow(self.create_image(np.squeeze(fake_A)), cmap=plt.cm.gray, interpolation='nearest')
        ax[(1,2)].imshow(self.create_image(np.squeeze(rec_B)), cmap=plt.cm.gray, interpolation='nearest')
        ax[(1,0)].axis('off')
        ax[(1,1)].axis('off')
        ax[(1,2)].axis('off')
        fig.suptitle('Input | Fake | Reconstructions')
        plt.show()
        #fig.savefig(''.join(['imgs/test_',str(epoch).zfill(4),'.png']),dpi=100)
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,10), linewidth = 4)
        D_A_plt, = plt.semilogy((self.D_A_loss_list), linewidth = 4, ls='-', color='r', alpha = .5, label='D_A')
        G_A_plt, = plt.semilogy((self.G_A_loss_list),linewidth = 4, ls='-', color='b',alpha = .5, label='G_A')
        D_B_plt, = plt.semilogy((self.D_B_loss_list),linewidth = 4, ls='-', color='k',alpha = .5, label='D_B')
        G_B_plt, = plt.semilogy((self.G_B_loss_list),linewidth = 4,ls='-', color='g',alpha = .5, label='G_B')
        
        axes = plt.gca()
        leg = plt.legend(handles=[D_A_plt, G_A_plt, D_B_plt, G_B_plt], fontsize=20)
        leg.get_frame().set_alpha(0.5)
        plt.show()


    def sample_fake(self,fake_pool, n, fake):
        """
        Image pool of the previous 50 fake images to sample from for training
        """
        n_fakes = fake_pool.shape[0]
        if n_fakes <= n:
            fake_pool = np.append(fake_pool, fake,axis=0)
            return fake
        else:
            if np.random.rand(1)[0] > 0.5:
                randIdx = np.random.randint(0,n_fakes,1)[0]
                tmp = fake_pool[randIdx,]
                fake_pool[randIdx,] = fake
                return tmp
            else:
                return fake
            

    def data_iterator(self,data,batch_size):
        """ A simple data iterator """
        batch_idx = 0
        while True:
            idxs = np.arange(0, len(data))
            np.random.shuffle(idxs)
            for batch_idx in range(0, len(data), batch_size):
                cur_idxs = idxs[batch_idx:batch_idx+batch_size]
                images_batch = data[cur_idxs]
                #images_batch = images_batch.astype("float32")
                yield images_batch
                
    def train(self, saved_session=None,epoch=0):
        self.real_A = tf.placeholder(tf.float32, [None, *self.img_dim])
        self.real_B = tf.placeholder(tf.float32, [None, *self.img_dim])
        self.fake_A = tf.placeholder(tf.float32, [None, *self.img_dim])
        self.fake_B = tf.placeholder(tf.float32, [None, *self.img_dim])
        self.fake_A_pool = np.empty((0,*self.img_dim))
        self.fake_B_pool = np.empty((0,*self.img_dim))
        
        self.G_A_loss_list = []
        self.G_B_loss_list = []
        self.D_A_loss_list = []
        self.D_B_loss_list = []

        real_A_norm = self.real_A
        real_B_norm = self.real_B

##        real_A_norm = instance_norm(self.real_A)
##        real_B_norm = instance_norm(self.real_B)
        
        #fDA
        disc_A_loss, D_A_params = self.fDx(self.disc_A, self.gen_A, real_B_norm, self.fake_B, self.real_label_A,self.fake_label_A, scope_name="A")

        #fGA
        gen_A_loss, G_A_params, fake_B, rec_A, errRec_A, errG_A, errAdapt_A = self.fGx(self.gen_A, self.disc_A, self.gen_B, real_A_norm, real_B_norm,
                                                                                       self.real_label_A, self.lambda_A, self.lambda_B,scope_name="A")
        
 
        #fDB
        disc_B_loss, D_B_params = self.fDx(self.disc_B, self.gen_B, real_A_norm, self.fake_A, self.real_label_B,self.fake_label_B, scope_name="B")

        #fGB
        gen_B_loss, G_B_params, fake_A, rec_B, errRec_B, errG_B, errAdapt_B = self.fGx(self.gen_B, self.disc_B, self.gen_A, real_B_norm, real_A_norm,
                                                                                       self.real_label_B, self.lambda_B, self.lambda_A,scope_name="B")

        
        lr = tf.placeholder(tf.float32, shape=[])

        optim = tf.train.AdamOptimizer(lr)

        grads_D_A = optim.compute_gradients(disc_A_loss, var_list = D_A_params)
        grads_G_A = optim.compute_gradients(gen_A_loss, var_list = G_A_params)
        grads_D_B = optim.compute_gradients(disc_B_loss, var_list = D_B_params)
        grads_G_B = optim.compute_gradients(gen_B_loss, var_list = G_B_params)

        train_D_A = optim.apply_gradients(grads_D_A)
        train_G_A = optim.apply_gradients(grads_G_A)
        train_D_B = optim.apply_gradients(grads_D_B)
        train_G_B = optim.apply_gradients(grads_G_B)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver() 
        self.sess = tf.Session()
        self.sess.run(init)
        tf.summary.FileWriter('logs', self.sess.graph)
        
        if saved_session is not None:
            tf.train.Saver.restore(saver, self.sess, saved_session)

        
        total_batch = int(np.floor(self.num_examples / self.batch_size)) # how many batches are in an epoch
        try:
            while epoch < self.num_epochs:    
                for i in tqdm.tqdm(range(total_batch)):
                    
                    self.iter_A = self.data_iterator(self.dataA, self.batch_size)
                    self.iter_B = self.data_iterator(self.dataB, self.batch_size)
                    next_batch_A = self.iter_A.__next__()
                    next_batch_B = self.iter_B.__next__()

                    if epoch <= 100:
                        curr_lr = self.learning_rate
                    else:
                        curr_lr = -(epoch-100)* self.learning_rate / 100 + self.learning_rate

                    
                    _, G_A_loss, fakeB, recA = self.sess.run([train_G_A, gen_A_loss, fake_B, rec_A],{lr: curr_lr,
                                                                                                     self.real_A: next_batch_A, self.real_B: next_batch_B})
                    sample_fake_B = self.sample_fake(self.fake_B_pool, self.pool_size, fakeB)
      
                    _, D_A_loss = self.sess.run([train_D_A, disc_A_loss],{lr: curr_lr, self.real_A: next_batch_A,
                                                                          self.real_B: next_batch_B, self.fake_B: sample_fake_B})

                                  
                    _, G_B_loss, fakeA, recB = self.sess.run([train_G_B, gen_B_loss, fake_A, rec_B],{lr: curr_lr,
                                                                                                     self.real_A: next_batch_A, self.real_B: next_batch_B})
                    
                    sample_fake_A = self.sample_fake(self.fake_A_pool, self.pool_size, fakeA)
                    _, D_B_loss = self.sess.run([train_D_B, disc_B_loss],{lr: curr_lr, self.real_A: next_batch_A,
                                                                          self.real_B: next_batch_B, self.fake_A: sample_fake_A})

                    self.G_A_loss_list.append(G_A_loss)
                    self.G_B_loss_list.append(G_B_loss)
                    self.D_A_loss_list.append(D_A_loss)
                    self.D_B_loss_list.append(D_B_loss)
                
                    if i%100 == 0:
                        # print display network output
                        IPython.display.clear_output()
                        print('Epoch: '+str(epoch))
                        print(G_A_loss,D_A_loss,G_B_loss,D_B_loss)
                        self.plot_network_output(next_batch_A,fakeB,recA,next_batch_B,fakeA,recB)
                        time.sleep(5)
              
                # save network
                saver.save(self.sess,''.join(['D:/Research Files/CyleGAN/models/summer_winter_',str(epoch).zfill(4),'.tfmod']))
                epoch +=1

        except(KeyboardInterrupt):
            print("(Epoch = {} ) err_d_a: {}, err_g_a: {}, err_d_b: {}, err_g_b ".format(epoch, D_A_loss, G_A_loss, D_B_loss, G_B_loss))











