import os
import util
import model
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools as itr
from PIL import Image
from random import randrange
from collections import Counter

#cleverhans
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model, KerasModelWrapper
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, LBFGS, BasicIterativeMethod
from cleverhans.utils import AccuracyReport

#keras
import keras
from keras import __version__
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import mnist
print("Finished Import")

# Set arguments:  Save_Dir Structure Learning_Rate Earling_Stoping Batch_Size Data_Dir    
epochs = 10
data_dir = ''
save_dir = ''
learning_rate = 0.001
early_stop = 20
batch_size = 128
reg_coeff = 0.001
split = 0.90
master = ''
checkpoint_path = 'nips-2017-adversarial-learning-development-set/inception_v3.ckpt'
input_dir = 'nips-2017-adversarial-learning-development-set/images/'
output_dir = ''
image_width = 28
image_height = 28
num_classes = 10
eps = 2.0 * 16.0 / 255.0
batch_shape = [batch_size, image_height, image_width, 3]
num_ens = 10

tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.set_verbosity(tf.logging.ERROR)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, image_width, image_height)
    x_test = x_test.reshape(x_test.shape[0], 1, image_width, image_height)
    input_shape = (1, image_width, image_height)
else:
    x_train = x_train.reshape(x_train.shape[0], image_width, image_height, 1)
    x_test = x_test.reshape(x_test.shape[0], image_width, image_height, 1)
    input_shape = (image_width, image_height, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#Our model architecture for MNIST dataset
def model_arch():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))    #(?,image_width,image_height,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
    return model

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
sess = tf.Session()
keras.backend.set_session(sess)

#-----------------------------------Adversarial Training--------------------------------------------------------------
#first adversarial examples are generated using train_data, then the model is trained on train_data+adv_train_data.
#Then the model is tested on normal test_data, then the model is tested on adversarial_test_data.
#So, we are generating the adversarial examples twice both on train and test data.

model = load_model("models/mnist/fgsm_model.h5")
wrap = KerasModelWrapper(model)

#CLEAN TESTING
print("CLEAN TESTING")
#build a fresh model for fgsm training
fgsm_acc_train = model.evaluate(x_test,y_test,verbose=0)
fgsm_acc_train[1] #Accuracy of adversarially trained model on clean examples
print("Accuracy : " + str(fgsm_acc_train[1]))

#FGSM TESING
print("FGSM TESTING")
#generate adversarial examples for adversarially trained model on test_data
adv_fgsm_test = util.fgsm_attack(x_test,model,sess)
fgsm_adv_acc_train = model.evaluate(adv_fgsm_test,y_test,verbose=0)
fgsm_adv_acc_train[1] #Accuracy of adversarially trained model on adv_test images
print("Accuracy : " + str(fgsm_adv_acc_train[1]))

#BIM TESTING
print("BIM TESTING")
#generate adversarial examples for adversarially trained model on test_data
adv_bim_test = util.bim_attack(x_test,model,sess)
bim_adv_acc_train = model.evaluate(adv_bim_test,y_test,verbose=0)
bim_adv_acc_train[1] #Accuracy of adversarially trained model on adv_test images
print("Accuracy : " + str(bim_adv_acc_train[1]))

# #LBFGS TESTING
# print("LBFGS TESTING")
# #generate adversarial examples for adversarially trained model on test_data
# adv_lbfgs_test = util.lbfgs_attack(x_test,model,sess,6)
# lbfgs_adv_acc_train = model.evaluate(adv_lbfgs_test,y_test,verbose=0)
# lbfgs_adv_acc_train[1] #Accuracy of adversarially trained model on adv_test images
# print("Accuracy : " + str(lbfgs_adv_acc_train[1]))

del model