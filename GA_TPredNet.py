#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 23:56:11 2020

@author: navin
"""

"""Traffic Congestion Prediction based on Generative Adversarial Network using Image dataset
 for city traffic congestion level """

import numpy as np
import pandas as pd
import cv2
import os

from keras.models import Input, Model
from keras.layers import Conv2D, Conv2DTranspose, LSTM, Activation, BatchNormalization, Flatten, Reshape, Dropout, Dense, Concatenate, TimeDistributed
from keras.optimizers import Adam
from keras.utils import to_categorical
#%%
data_path='/home/navin/GAN_thesis/dataset1/'
img_list=os.listdir(data_path)
img_list.sort()
len(img_list)
img_data_list=[]
test_holder=[]
#%%
for img in img_list:
    input_img=cv2.imread(data_path+img)
    test_img=cv2.imread(data_path+img,0)
    input_img=input_img.flatten()
    test_img=test_img.flatten()
    test_holder.append(test_img)
    img_data_list.append(input_img)
#%%
data3=np.array(test_holder)
test_holder = []
gray=np.zeros_like(data3)
gray[np.logical_and(data3>=0,data3<40)]=0
gray[np.logical_and(data3>=40,data3<140)]=1 
gray[np.logical_and(data3>=190, data3<255)]=2
gray[np.logical_and(data3>=140,data3<190)]=3
data3=pd.DataFrame(gray)

  #%%
row = 192
col = 448
channel = 3
data_per_day = 60
past_sequence = 12
prediction_horizon = 2
#split data into train and test based on number of day
train_upto_day = 40
total_day = 45
k = past_sequence + 1
#%%
y_train=[]
b=[i for i in range(0,train_upto_day)]
for j in range (0,len(b)-1):
    train1=data3.iloc[b[j]*data_per_day : b[j+1]*data_per_day,::]
    for i in range(past_sequence+prediction_horizon,train1.shape[0]+1):
        d=train1.iloc[i-past_sequence:i,:]
        d=np.array(d)
        y_train.append(d)   
y_train=np.array(y_train)
train1=[]
y_train= y_train.reshape(y_train.shape[0],y_train.shape[1], row, col, 1)

#%%
y_vali=[]
b=[i for i in range(train_upto_day-1,total_day)]
for j in range (0,len(b)-1):
    vali1=data3.iloc[b[j]*data_per_day : b[j+1]*data_per_day,::]
    for i in range(past_sequence+prediction_horizon,vali1.shape[0]+1):
        d=vali1.iloc[i-past_sequence:i,:]
        d=np.array(d)
        y_vali.append(d)   
y_vali=np.array(y_vali)
vali1 =[]
y_vali = y_vali.reshape(y_vali.shape[0], y_vali.shape[1],row, col,1)
#y_vali= y_vali.reshape(y_vali.shape[0], row, col, 4)
y_train1=to_categorical(y_train)
y_train1.shape
y_vali1=to_categorical(y_vali)
y_vali1.shape
data3 = []
y_train =[]
y_vali = []
#%%
data1=np.array(img_data_list)
img_data_list =[]
data=pd.DataFrame(data1)
x_train=[]
b=[i for i in range(0,train_upto_day)]
for j in range (0,len(b)-1):
    train=data.iloc[b[j]*data_per_day:b[j+1]*data_per_day-prediction_horizon,::]
    for i in range(past_sequence,train.shape[0]+1):
        d=train.iloc[i-past_sequence:i,:]
        d=np.array(d) 
        x_train.append(d)
x_train=np.array(x_train)
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],row,col,channel)
print(x_train.shape)
#%%
x_vali=[]
b=[i for i in range(train_upto_day-1, total_day)]
b
for j in range (0,len(b)-1):
    vali = data.iloc[b[j]*data_per_day:b[j+1]*data_per_day-prediction_horizon,::]
    for i in range(past_sequence,vali.shape[0]+1):
        d = vali.iloc[i-past_sequence:i,:]
        d = np.array(d) 
        x_vali.append(d)
x_vali=np.array(x_vali)
x_vali=x_vali.reshape(x_vali.shape[0],x_vali.shape[1],row,col,channel)
x_vali.shape
data = []
#%%
'''Generator'''
def generator(image_shape):
    f1=32
    f2=64
    f3=96
    f4=128
    f5=160
    f6=192
    input_img = Input(shape = image_shape) # image_shape [step, row, cols, channel]
    x1 = TimeDistributed(Conv2D(f1, (3, 3),strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(input_img)
    x1 = TimeDistributed(BatchNormalization())(x1)
    x1 = TimeDistributed(Dropout(0.1))(x1)
    
    x2 = TimeDistributed(Conv2D(f2, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(x1)
    x2 = TimeDistributed(BatchNormalization())(x2)
    x2 = TimeDistributed(Dropout(0.1))(x2)
    x2 = TimeDistributed(Conv2D(f2, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(x2)
    x2 = TimeDistributed(BatchNormalization())(x2)
    x2 = TimeDistributed(Dropout(0.1))(x2)
    
    x3 = TimeDistributed(Conv2D(f3, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(x2)
    x3 = TimeDistributed(BatchNormalization())(x3)
    x3 = TimeDistributed(Dropout(0.1))(x3)
    x3 = TimeDistributed(Conv2D(f3, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(x3)
    x3 = TimeDistributed(BatchNormalization())(x3)
    x3 = TimeDistributed(Dropout(0.1))(x3)
    
    x4 = TimeDistributed(Conv2D(f4, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(x3)
    x4 = TimeDistributed(BatchNormalization())(x4)
    x4 = TimeDistributed(Dropout(0.1))(x4)
    x4 = TimeDistributed(Conv2D(f4, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(x4)
    x4 = TimeDistributed(BatchNormalization())(x4)
    x4 = TimeDistributed(Dropout(0.1))(x4)
    
    x5 = TimeDistributed(Conv2D(f5, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(x4)
    x5 = TimeDistributed(BatchNormalization())(x5)
    x5 = TimeDistributed(Dropout(0.1))(x5)
    x5 = TimeDistributed(Conv2D(f5, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(x5)
    x5 = TimeDistributed(BatchNormalization())(x5)
    x5 = TimeDistributed(Dropout(0.1))(x5)
    
    x6 = TimeDistributed(Conv2D(f6, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(x5)
    x6 = TimeDistributed(BatchNormalization())(x6)
    x6 = TimeDistributed(Dropout(0.1))(x6)
    x6 = TimeDistributed(Conv2D(8, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(x6)
    x6 = TimeDistributed(Flatten())(x6)
    encoded = TimeDistributed(Dropout(0.1))(x6)
    
    
    lstm = LSTM(672,return_sequences=True, activation='relu', kernel_initializer='he_uniform')(encoded)
    lstm = LSTM(672, return_sequences=True, activation='relu', kernel_initializer='he_uniform')(lstm)
    lstm = LSTM(672,return_sequences=True, activation='relu', kernel_initializer='he_uniform')(lstm)
    lstm = LSTM(672,return_sequences=True, activation='relu', kernel_initializer='he_uniform')(lstm)
    lstm = LSTM(672, activation='relu',return_sequences=True, kernel_initializer='he_uniform')(lstm)
    
    
    d1 = Reshape((12,6,14,8))(lstm)
    d1 = TimeDistributed(Conv2DTranspose(f5, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(d1)
    d1 = TimeDistributed(BatchNormalization())(d1)
    d1 = Concatenate()([d1,x5])
    d1 = TimeDistributed(Dropout(0.1))(d1)
    d1 = TimeDistributed(Conv2D(f5, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(d1)
    d1 = TimeDistributed(BatchNormalization())(d1)
    d1 = TimeDistributed(Dropout(0.1))(d1)
 
    d2 = TimeDistributed(Conv2DTranspose(f4, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(d1)
    d2 = TimeDistributed(BatchNormalization())(d2)
    d2 = Concatenate()([d2,x4])
    d2 = TimeDistributed(Dropout(0.1))(d2)
    d2 = TimeDistributed(Conv2D(f4, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(d2)
    d2 = TimeDistributed(BatchNormalization())(d2)
    d2 = TimeDistributed(Dropout(0.1))(d2)


    d3 = TimeDistributed(Conv2DTranspose(f3, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(d2)
    d3 = TimeDistributed(BatchNormalization())(d3)
    d3 = Concatenate()([d3,x3])
    d3 = TimeDistributed(Dropout(0.1))(d3)
    d3 = TimeDistributed(Conv2D(f3, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(d3)
    d3 = TimeDistributed(BatchNormalization())(d3)
    d3 = TimeDistributed(Dropout(0.1))(d3)


    d4 = TimeDistributed(Conv2DTranspose(f2, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(d3)
    d4 = TimeDistributed(BatchNormalization())(d4)
    d4 = Concatenate()([d4,x2])
    d4 = TimeDistributed(Dropout(0.1))(d4)
    d4 = TimeDistributed(Conv2D(f2, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(d4)
    d4 = TimeDistributed(BatchNormalization())(d4)
    d4 = TimeDistributed(Dropout(0.1))(d4)



    d5 = TimeDistributed(Conv2DTranspose(f1, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(d4)
    d5 = TimeDistributed(BatchNormalization())(d5)
    d5 = Concatenate()([d5,x1])
    d5 = TimeDistributed(Dropout(0.1))(d5)

    d5 = TimeDistributed(Conv2D(f1, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(d5)
    d5 = TimeDistributed(BatchNormalization())(d5)
    d5 = TimeDistributed(Dropout(0.1))(d5)

    output_image = TimeDistributed(Conv2D(4, (3, 3),activation='softmax', padding='same',kernel_initializer='he_uniform'))(d5)


     
    generator = Model(input_img, output_image)
    generator.summary()
    
    return generator   
#%%

# define a generator block of GAN generator with real and target image
#def discriminator():
##    in_source_image = Input(shape = (192,448,36))
#    in_target_image = Input(shape = (192,448,48))
##    merged = Concatenate()([in_source_image, in_target_image])
#    d = Conv2D(64, (2,2), strides = (1), padding = 'valid', activation = 'relu', kernel_initializer = 'he_uniform')(in_target_image)
#    d = BatchNormalization()(d)
#    d = Dropout(0.1)(d)
#       
#    d = Conv2D(128, (2,2), strides = (2,2), padding = 'valid', activation = 'relu', kernel_initializer = 'he_uniform')(d)
#    d = BatchNormalization()(d)
#    d = Dropout(0.1)(d)
#    
#    d = Conv2D(256, (2,2), strides = (2,2), padding = 'valid', activation = 'relu', kernel_initializer = 'he_uniform')(d)
#    d = BatchNormalization()(d)
#    d = Dropout(0.1)(d)
#    
#    d = Conv2D(128, (2,2), strides = (2,2), padding = 'valid', activation = 'relu', kernel_initializer = 'he_uniform')(d)
#    d = BatchNormalization()(d)
#    d = Dropout(0.1)(d)
#    
#    d = Conv2D(64, (2,2),strides = (2,2), padding = 'valid', activation = 'relu', kernel_initializer = 'he_uniform')(d)
#    d = BatchNormalization()(d)
#    d = Dropout(0.1)(d)
#    d = Conv2D(32, (2,2),strides = (2,2,), padding = 'valid', activation = 'relu', kernel_initializer = 'he_uniform')(d)
#    d = BatchNormalization()(d)
##    d = Flatten()(d)
##    
##    d = Dense(100, activation = 'relu')(d)
##    output = Dense(1)(d)
#    output = Conv2D(1, (2,2), strides = (2,2), padding = 'valid', activation = 'relu', kernel_initializer = 'he_uniform')(d)
#    discriminator = Model( in_target_image, output)
#    discriminator.summary()
#    return(discriminator)
#%%
'''Discriminator'''
def discriminator():
    in_target_image = Input(shape = (192,448,48))
    d = Conv2D(64, (2,2), strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(in_target_image)
    d = BatchNormalization()(d)
    d = Dropout(0.1)(d)
       
    d = Conv2D(128, (2,2), strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(d)
    d = BatchNormalization()(d)
    d = Dropout(0.1)(d)
    
    d = Conv2D(256, (2,2), strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(d)
    d = BatchNormalization()(d)
    d = Dropout(0.1)(d)
    
    d = Conv2D(128, (2,2), strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(d)
    d = BatchNormalization()(d)
    d = Dropout(0.1)(d)
    
    d = Conv2D(64, (2,2),strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(d)
    d = BatchNormalization()(d)
    d = Dropout(0.1)(d)
    d = Conv2D(32, (2,2),strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(d)
    d = BatchNormalization()(d)

    output = Conv2D(1, (2,2), strides = (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(d)
    discriminator = Model( in_target_image, output)
    discriminator.summary()
    return(discriminator)
#%%
def define_gan(g_model, d_model, source_image_shape):
    '''make weights in the discriminator not trainable'''
    d_model.trainable = False
    
    '''define the source image'''
    in_source_image = Input(shape = source_image_shape)
    
    '''connect the source input to the generator input'''
    generator_output= g_model(in_source_image)
    generator_output1 = Reshape((192,448,48))(generator_output)
    
    '''Connect the target_input and generator output to the discriminator input'''
    discriminator_output = d_model(generator_output1)
    
    ''' source image sequence, target image, generator image, and discriminator output'''
    gan_model = Model(in_source_image, discriminator_output)
    
    return gan_model
#%%
def generate_real_samples(x_train, y_train1, order, samps):
    x_train_g = x_train[order]
    x_train_g = x_train_g.reshape(samps, x_train_g.shape[0], x_train_g.shape[1], x_train_g.shape[2], x_train_g.shape[3])
    x_train_d = y_train1[order]
    x_train_d = x_train_d.reshape(samps, x_train_d.shape[0], x_train_d.shape[1], x_train_d.shape[2], x_train_d.shape[3])
    return [x_train_g, x_train_d]
#%%
def summarize_performance(g_model, x_train,y_train, order,name, n_samples= 1):
    [x_train_g, x_train_d]= generate_real_samples(x_train, y_train1, order, samps=1)
    x_gen = g_model.predict(x_train_g)
    pred1=x_gen[:,11,:,:,:]
    pred=pred1.argmax(axis=-1)
    pred.shape
    pred_n=np.zeros(shape=(pred.shape[0],192,448,3))
    pred_n[np.logical_and(pred>=0,pred<1)]=[0,0,0]
    pred_n[np.logical_and(pred>=1,pred<2)]=[75,90,255]
    pred_n[np.logical_and(pred>=2,pred<3)]=[75,225,250]
    pred_n[np.logical_and(pred>=3,pred<=4)]= [100,195,140]
    target1 = x_train_d[:,11,:,:,:]
    target = target1.argmax(axis = -1)
    target_n=np.zeros(shape=(target.shape[0],192,448,3))
    target_n[np.logical_and(target>=0,target<1)]=[0,0,0]
    target_n[np.logical_and(target>=1,target<2)]=[75,90,255]
    target_n[np.logical_and(target>=2,target<3)]=[75,225,250]
    target_n[np.logical_and(target>=3,target<=4)]= [100,195,140] 
    print(target_n.shape)
    out = cv2.vconcat([target_n[0], pred_n[0]])
    cv2.imwrite('/home/navin/GAN_thesis/kri4/gan{}.png'.format(name),out)
#%%
from sklearn.metrics import accuracy_score
def accuracy(true,pred):
    return accuracy_score(true,pred)
#%%

def accuracy_check(g_model, x_vali,y_vali1):
    x_gen = g_model.predict(x_vali, batch_size = 1)
    road=pd.read_csv('/home/navin/road1.csv')
    road=road.iloc[::,1:]
    road=np.array(road)
    pred_n = x_gen[:,11,:,:,:]
    pred_n=pred_n.argmax(axis=-1)
    pred_n.shape
    
    """to check the accuracy of the model """
    pred_n=pred_n.reshape(x_vali.shape[0],192,448)
    pred_all=[]
    for a in range (0,pred_n.shape[0]):
        val=[]
        input_img=pred_n[a]
        for i in range(0, len(road)):
            level=input_img[road[i][0]][road[i][1]]
            val.append(level)
        val=np.array(val)
        pred_all.append(val)
    pred_all=np.array(pred_all)
    pred_all.shape
    pred_all=pred_all.reshape(x_vali.shape[0], 349)

    true_n = y_vali1[:,11,:,:,:]
    true_n = true_n.argmax(axis = -1)
    true_n=true_n.reshape(x_vali.shape[0],192,448)
    true_all=[]
    for a in range (0,true_n.shape[0]):
        val=[]
        input_img=true_n[a]
        for i in range(0, len(road)):
            level=input_img[road[i][0]][road[i][1]]
            val.append(level)
        val=np.array(val)
        true_all.append(val)
    true_all=np.array(true_all)
    true_all=true_all.reshape(x_vali.shape[0],349)
    s = 0
    s1 = 0
    for i in range(0,x_vali.shape[0]):
        s1  += 1
        print(accuracy_score(true_all[i],pred_all[i]))
        s += accuracy_score(true_all[i],pred_all[i])
    print('Accuracy')
    print(s/s1)
    #%%
def train(d_model, g_model, gan_model, x_train, y_train1,x_vali, y_vali1, epochs = 10000, n_batch = 1):
    
    '''determine the output shape of the discriminator''' 
    batch_per_epochs = int(len(x_train) / n_batch)
    y_real = np.ones((1,192,448,1))
    y_fake = np.zeros((1,192,448,1))

    for epoch in range (0, epochs):
        for i in range(0, batch_per_epochs):
            samps = 1
            '''train discrimintor based on real samples'''
            [x_train_g, x_train_d] = generate_real_samples(x_train, y_train1, i, samps)
            x_generator = g_model.predict(x_train_g)
            x_train_d1 = x_train_d.reshape(1,192, 448, 48)
            x_generator = x_generator.reshape(1,192, 448, 48)
#            x_train_in = x_train_g.reshape(1,192,448,36)

            '''Train discriminator in real and fake data'''
            d_loss_real = d_model.train_on_batch(x_train_d1, y_real)

            d_loss_fake = d_model.train_on_batch(x_generator, y_fake)

#            d_loss = 0.5* np.add(d_loss_real, d_loss_fake)
            
            #update generator model
            g_loss = gan_model.train_on_batch(x_train_g,y_real)   #%% look 
#            print('done_gan')
            print("%d  %d [D loss:  %f %f, acc.: %.2f%%   %.2f%%] [G loss: %f %f ]" % (epoch, i, d_loss_real[0], d_loss_fake[0], d_loss_real[1],d_loss_fake[1],g_loss[0],g_loss[1]))
            if (i+1) % 100 == 0:
                summarize_performance(g_model, x_train, y_train,i, epoch*len(x_train) + i)
            if (i+1) % 500 == 0:
                accuracy_check(g_model, x_vali, y_vali1)
#%%
optimizer = Adam(lr = 0.00001, beta_1 = 0.9)
source_image_shape = [12, 192, 448, 3]
source_f_shape = [192,448,36]
target_image_shape = [192, 448, 4]
d_model = discriminator()
d_model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
g_model = generator(source_image_shape) 
g_model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)
gan_model = define_gan(g_model, d_model, source_image_shape) 
gan_model.compile(loss = ['binary_crossentropy'], optimizer = optimizer, metrics = ['mse'])
train(d_model, g_model, gan_model, x_train, y_train1, x_vali, y_vali1) 
#%%
from keras.models import save_model, load_model
gan_model.save('/home/navin/t1.h5')
d_model.save('/home/navin/t_d.h5')
g_model.save('/home/navin/tg.h5')
##%%

#%%