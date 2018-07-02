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
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_float('lr', 0.001, '')
flags.DEFINE_integer('early_stop', 20, '')
flags.DEFINE_string('db', 'emodb', '')
flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_float('reg_coeff', 0.001, '')
flags.DEFINE_float('split', 0.90, '')
flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')
flags.DEFINE_string('checkpoint_path', 'nips-2017-adversarial-learning-development-set/inception_v3.ckpt', 'Path to checkpoint for inception network.')
flags.DEFINE_string('input_dir', 'nips-2017-adversarial-learning-development-set/images/', 'Input directory with images.')
flags.DEFINE_string('output_dir', '', 'Output directory with images.')
flags.DEFINE_float('max_epsilon', 4.0, 'Maximum size of adversarial perturbation.')
flags.DEFINE_integer('image_width', 28, 'Width of each input images.')
flags.DEFINE_integer('image_height', 28, 'Height of each input images.')
flags.DEFINE_float('eps', 2.0 * 16.0 / 255.0, '')
flags.DEFINE_integer('num_classes', 10, '')
flags.DEFINE_integer('num_ens', 10, '')
FLAGS = flags.FLAGS

def main(argv):

    print("Start Main")
    # Set arguments:  Save_Dir Structure Learning_Rate Earling_Stoping Batch_Size Data_Dir    
    epochs = FLAGS.epochs
    data_dir = FLAGS.data_dir
    save_dir = FLAGS.save_dir
    learning_rate = FLAGS.lr
    early_stop = FLAGS.early_stop
    batch_size = FLAGS.batch_size
    reg_coeff = FLAGS.reg_coeff
    split = FLAGS.split
    master = FLAGS.master
    checkpoint_path = FLAGS.checkpoint_path
    input_dir = FLAGS.input_dir
    output_dir = FLAGS.output_dir
    image_width = FLAGS.image_width
    image_height = FLAGS.image_height
    num_classes = FLAGS.num_classes
    eps = FLAGS.eps
    batch_shape = [batch_size, image_height, image_width, 3]
    num_ens = FLAGS.num_ens

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
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_width,image_height,1)))
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

    model = load_model("models/mnist/original_model.h5")
    wrap = KerasModelWrapper(model)

    print("Generate Examples")
    #generate adversarial examples on train data.
    adv_fgsm_train = util.fgsm_attack(x_train,model,sess)
    print("Done Generating")
    #adv_bim_train = util.bim_attack(x_train,model,sess)
    #adv_lbfgs_train = util.lbfgs_attack(x_train,model,sess,6)
    train_plus_adv_fgsm = np.concatenate([x_train,adv_fgsm_train])
    y_train_plus_adv_fgsm = np.concatenate([y_train,y_train])
    #train_plus_adv_bim = np.concatenate([x_train,adv_bim_train])
    #y_train_plus_adv_bim = np.concatenate([y_train,y_train])
    #train_plus_adv_lbfgs = np.concatenate([x_train,adv_lbfgs_train])
    #y_train_plus_adv_lbfgs = np.concatenate([y_train,y_train])
    del model
    
    #FGSM TRAINING
    #build a fresh model for fgsm training
    print("FGSM TRAINING")
    model = model_arch()
    wrap = KerasModelWrapper(model)
    model.fit(train_plus_adv_fgsm, y_train_plus_adv_fgsm, batch_size=batch_size, epochs=epochs, verbose=1)
    model.save("models/mnist/fgsm_model.h5")
    #fgsm_acc_train = model.evaluate(x_test,y_test,verbose=0)
    #fgsm_acc_train[1] #Accuracy of adversarially trained model on clean examples

    #generate adversarial examples for adversarially trained model on test_data
    #adv_fgsm_test = util.fgsm_attack(x_test,model,sess)
    #fgsm_adv_acc_train = model.evaluate(adv_fgsm_test,y_test,verbose=0)
    #fgsm_adv_acc_train[1] #Accuracy of adversarially trained model on adv_test images

    del model

if __name__ == "__main__":
    tf.app.run()