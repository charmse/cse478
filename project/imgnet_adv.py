import os
import util
import model
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools as itr
from scipy.misc import imread
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

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', '', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 16, '')
flags.DEFINE_float('lr', 0.001, '')
flags.DEFINE_integer('early_stop', 20, '')
flags.DEFINE_string('db', 'emodb', '')
flags.DEFINE_integer('epochs', 100, '')
flags.DEFINE_float('reg_coeff', 0.001, '')
flags.DEFINE_float('split', 0.90, '')
flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')
flags.DEFINE_string('checkpoint_path', 'nips-2017-adversarial-learning-development-set/inception_v3.ckpt', 'Path to checkpoint for inception network.')
flags.DEFINE_string('input_dir', 'nips-2017-adversarial-learning-development-set/images/', 'Input directory with images.')
flags.DEFINE_string('output_dir', '', 'Output directory with images.')
flags.DEFINE_float('max_epsilon', 4.0, 'Maximum size of adversarial perturbation.')
flags.DEFINE_integer('image_width', 299, 'Width of each input images.')
flags.DEFINE_integer('image_height', 299, 'Height of each input images.')
flags.DEFINE_float('eps', 2.0 * 16.0 / 255.0, '')
flags.DEFINE_integer('num_classes', 1001, '')
flags.DEFINE_integer('num_ens', 10, '')
FLAGS = flags.FLAGS

def main(argv):

    print("Start Main")
    # Set arguments:  Save_Dir Structure Learning_Rate Earling_Stoping Batch_Size Data_Dir    
    data_dir = FLAGS.data_dir
    save_dir = FLAGS.save_dir
    learning_rate = FLAGS.lr
    early_stop = FLAGS.early_stop
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    reg_coeff = FLAGS.reg_coeff
    split = FLAGS.split
    master = FLAGS.master
    checkpoint_path = FLAGS.checkpoint_path
    input_dir = FLAGS.input_dir
    output_dir = FLAGS.output_dir
    image_width = FLAGS.image_width
    image_height = FLAGS.eps
    num_classes = FLAGS.num_classes
    eps = FLAGS.eps
    batch_shape = [batch_size, image_height, image_width, 3]
    input_shape = [image_height, image_width, 3]
    num_ens = FLAGS.num_ens

    tf.logging.set_verbosity(tf.logging.INFO)

    def model_arch():
        model = Sequential()
        model.add(Conv2D(50, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(100, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(200, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(400, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
        return model

    model = model_arch

    #load training data
    x_train,y_train,train_names = util.load_training_images('tiny-imagenet-200/train/')
    print("Training Images Loaded")
    
    x_test,y_test,test_names = util.load_training_images('tiny-imagenet-200/test/')
    print("Testing Images Loaded")

    #retrype and resize training data
    x_train = x_train[0:100]
    y_train = y_train[0:100]
    train_names = train_names[0:100]
    x_train_large = np.ndarray(shape= [x_train.shape[0],299,299,3])
    for i in range(x_train.shape[0]):
        x_train_large[i,:,:,:] = util.rescale(x_train[i])
    x_train_large=x_train_large.astype('uint8')
    x_train_noisy = np.ndarray(shape= x_train_large.shape)
    for i in range(x_train_large.shape[0]):
        x_train_noisy[i,:,:,:] = util.noisy(1,x_train_large[i])
    x_train_noisy=x_train_noisy.astype('uint8')
    x_train_sub,y_train_sub = util.subsample(x_train_noisy,y_train)
    batch_shape = [20, 299, 299, 3]
    num_classes = 200

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    sess = tf.Session()
    keras.backend.set_session(sess)

    #-----------------------------------Adversarial Training--------------------------------------------------------------
    #first adversarial examples are generated using train_data, then the model is trained on train_data+adv_train_data.
    #Then the model is tested on normal test_data, then the model is tested on adversarial_test_data.
    #So, we are generating the adversarial examples twice both on train and test data.

    model = load_model("models/imgnet/original_model.h5")
    wrap = KerasModelWrapper(model)

    #generate adversarial examples on train data.
    adv_fgsm_train = util.fgsm_attack(x_train,model,sess)
    adv_bim_train = util.bim_attack(x_train,model,sess)
    adv_lbfgs_train = util.lbfgs_attack(x_train,model,sess,6)
    train_plus_adv_fgsm = np.concatenate([x_train,adv_fgsm_train])
    y_train_plus_adv_fgsm = np.concatenate([y_train,y_train])
    train_plus_adv_bim = np.concatenate([x_train,adv_bim_train])
    y_train_plus_adv_bim = np.concatenate([y_train,y_train])
    train_plus_adv_lbfgs = np.concatenate([x_train,adv_lbfgs_train])
    y_train_plus_adv_lbfgs = np.concatenate([y_train,y_train])
    del model

    print("FGSM TRAINING")
    #build a fresh model for fgsm training
    model = model_arch()
    wrap = KerasModelWrapper(model)
    model.fit(train_plus_adv_fgsm, y_train_plus_adv_fgsm, batch_size=batch_size, epochs=epochs, verbose=1)
    model.save("models/imgnet/fgsm_model.h5")
    fgsm_acc_train = model.evaluate(x_test,y_test,verbose=0)
    fgsm_acc_train[1] #Accuracy of adversarially trained model on clean examples

    #generate adversarial examples for adversarially trained model on test_data
    adv_fgsm_test = util.fgsm_attack(x_test,model,sess)
    fgsm_adv_acc_train = model.evaluate(adv_fgsm_test,y_test,verbose=0)
    fgsm_adv_acc_train[1] #Accuracy of adversarially trained model on adv_test images

    del model
    
    print("BIM TRAINING")#BIM TRAINING
    #build a fresh model for bim training
    model = model_arch()
    wrap = KerasModelWrapper(model)
    model.fit(train_plus_adv_bim, y_train_plus_adv_bim, batch_size=batch_size, epochs=epochs, verbose=1)
    bim_acc_train = model.evaluate(x_test,y_test,verbose=0)
    print("Accuracy of adversarially trained model on clean examples\n" + str(bim_acc_train[1]))

    #generate adversarial examples for adversarially trained model on test_data
    adv_bim_test = util.bim_attack(x_test,model,sess)
    bim_adv_acc_train = model.evaluate(adv_bim_test,y_test,verbose=0)
    print("Accuracy of adversarially trained model on adv_test images\n" + str(bim_adv_acc_train[1]))

    del model

    print("LBFGS TRAINING")
    #build a fresh model for lbfgs training
    model = model_arch()
    wrap = KerasModelWrapper(model)
    model.fit(train_plus_adv_lbfgs, y_train_plus_adv_lbfgs, batch_size=batch_size, epochs=epochs, verbose=1)
    print("Accuracy of adversarially trained model on clean examples")
    lbfgs_acc_train = model.evaluate(x_test,y_test,verbose=0)
    print(str(lbfgs_acc_train[1]))

    print("Accuracy of adversarially trained model on lbfgs examples")
    lbfgs_acc_train[1]
    adv_lbfgs_test = util.lbfgs_attack(x_test,model,sess,6)
    lbfgs_adv_acc_train = model.evaluate(adv_lbfgs_test,y_test,verbose=0)
    print(str(lbfgs_adv_acc_train[1]) #Accuracy of adversarially trained model on adv_test images
    
    del model

if __name__ == "__main__":
    tf.app.run()