# -*- coding: utf-8 -*-

# @Author    : Tatiana Tyukina 
# @Project   : Arch_I_Scan
# @Software  : 


import tensorflow as tf
from tensorflow import keras

import os
import io
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image


import matplotlib.pyplot as plt
from time import time

##https://keras.io/examples/vision/image_classification_from_scratch/
#cascade
#from tensorflow.keras import layers 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,load_model
    
import argparse
import pickle

from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import GeneratorEnqueuer, Sequence, OrderedEnqueuer


#initialization of global parameters

TRAIN_DIR ="./train"
VAL_DIR = "./val"
TEST_DIR = "./test"

NUM_CLASSES=5
MODEL_FILE="./Model/I_V3_Train.h5"#
MODEL_FILE_CKPT="Model/I_V3_Train_ckpt_{epoch}_{val_loss:.4f}_{val_accuracy:.4f}.h5"

LOG_DIR_FIGS="./logs/Figs"
LOG_DIR="./logs/"
dEPOCHS=5# Epochs total


batch_size = 32
img_height = 224#input size
img_width = 224 #
img_dim=(img_width,img_height)

#auxilliary functions, classes

def get_init_epoch(checkpoint_path): #get epoch number for writing model file
    filename = os.path.basename(checkpoint_path)
    filename = os.path.splitext(filename)[0]
    init_epoch = filename.split("_")[-3]
    score=  filename.split("_")[-1]
    return int(init_epoch), float(score)
    
#######################     

class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):

    def __init__(self, total_images=1, batch_size=1,classes=[], csv_file="tmp.csv"):
        #super(EarlyStoppingAtMinLoss, self).__init__()
        self.total_batches = np.ceil(total_images/batch_size)+1
        self.batch_size=batch_size
        self.classes=classes
        self.running_average_accuracy=0.0
        self.average_accuracy=0.0
        self.total_accuracy=0.0
        self.file_name=csv_file
        self.update=0.0
        
        
    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        indx=[self.batch_size*batch,min(self.batch_size*(batch+1),len(self.classes))]

        self.total_accuracy+=np.sum(1-np.abs(np.argmax(logs["outputs"],axis=-1)-np.array(self.classes[indx[0]:indx[-1]])))
        
        if ((batch+1)/self.total_batches*100)-self.update>1:
            self.update=int((batch+1)/self.total_batches*100)
            print("For batch  - {} ({:7.5f}%) , total_accuracy - {} over {} images".format(batch+1, (batch+1)/self.total_batches*100,
                               int(self.total_accuracy), min(len(self.classes),self.batch_size*(batch+1))), end="\r")
            
            
    def on_predict_end(self, logs=None):
        self.average_accuracy=self.total_accuracy/(len(self.classes))
        print("\n","average accuraccy - {:9.7f}".format(self.average_accuracy))
        return
        
    def on_epoch_end(self, epoch, logs=None):
        print("The average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )
        
####################### 
def time_decay(epoch, lr):
    decay_rate = 0.01
    new_lrate = lr/(1+decay_rate*epoch)
    return new_lrate

#######################    
class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
#######################


def make_image_tensor(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Adapted from https://github.com/lanpa/tensorboard-pytorch/
    """
    #print('make tensor', tensor.shape)
    if len(tensor.shape) == 3:
        (height, width, channel) = tensor.shape
    else:
        (height, width) = tensor.shape
        channel = 1
    tensor = tensor.astype(np.uint8)
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()

    return  tensor 

#######################    

class TensorBoardWriter:

    def __init__(self, outdir):
        assert (os.path.isdir(outdir))
        self.outdir = outdir

        self.writer = tf.summary.create_file_writer(self.outdir)
       

    def save_image(self, tag, image, global_step=None):
        
        image_tensor = make_image_tensor(image)
        
        if len(image_tensor.shape) == 3:
            image_tensor= np.reshape(image_tensor, (-1, *image_tensor.shape))
        else:
            image_tensor= np.reshape(image_tensor, (-1, *image_tensor.shape, 1))

        with self.writer.as_default():
            tf.summary.image(name=tag, data=image_tensor,
                                step=global_step)
        
    def save_text(self, tag, text_array, global_step=None):
        
        with self.writer.as_default():
            if np.isscalar(text_array):
                tf.summary.text(name=tag, data=str(text_array),
                                step=global_step)
            else:
                tf.summary.text(name=tag, data= np.array2string(text_array, separator=','),
                                step=global_step)


    def close(self):

        self.writer.close()

#######################    

class ModelDiagnoser(Callback):

    def __init__(self,
                 batch_size,
                 number_images,
                 data_generator,
                 output_dir,
                 normalization_mean):
        self.batch_size = batch_size
        self.n_imgs= number_images
        
        self.tensorboard_writer = TensorBoardWriter(output_dir)
        self.normalization_mean = normalization_mean
                
        is_sequence = isinstance(data_generator, Sequence)
        if is_sequence:
            self.enqueuer = OrderedEnqueuer(data_generator,
                                            use_multiprocessing=False,
                                            shuffle=False)
        else:
            self.enqueuer = GeneratorEnqueuer(data_generator,
                                              use_multiprocessing=False)
        self.enqueuer.start(workers=1, max_queue_size=64)

    def on_epoch_end(self, epoch, logs=None):
        output_generator = self.enqueuer.get()
        steps_done = 0
        for i in range(self.n_imgs):
            generator_output = next(output_generator)
            x, y = generator_output[0], generator_output[1]
            
            
            img = np.squeeze(x[i, :, :])#x[i, :, :, :])
            img = 255. * (img + self.normalization_mean)  # mean is the training images normalization mean
            img = img[:, :, [0, 1, 2]]  # reordering of channels
            ground_truth = np.squeeze(y[i, :])
                    
            class_index = np.argmax(ground_truth, axis=-1)
            
            y_pred = self.model.predict(x)
            pred = np.squeeze(y_pred[i, :])
            class_pred= np.argmax(pred, axis=-1)
                    
            self.tensorboard_writer.save_image("Epoch-{}/{}/{}/x"
                        .format(epoch,class_index,class_pred), img, i)
            self.tensorboard_writer.save_text("Epoch-{}/{}/{}/prediction"
                        .format(epoch,class_index,class_pred), pred, i)
                    

    def on_train_end(self, logs=None):
        self.enqueuer.stop()
        self.tensorboard_writer.close()

#######################    
 
 
def make_modelZZ(input_shape, num_classes):
   
    net =InceptionV3(weights='imagenet', include_top=False,  input_shape=(img_height,img_width,3))#VGG19(include_top=False,  input_shape=(img_height,img_width,3),pooling='max') 
    
    net.trainable = True

    
    model = Sequential([
       net,
       # here are  custom prediction layer 
       Dropout(0.50),
       GlobalAveragePooling2D(),
       Dense(512, activation='relu'),   
       Dense(NUM_CLASSES, activation='softmax')
    ])
     
    model.summary()
    
    opt = Adam(learning_rate=0.0001)
   # opt = SGD(lr=0.01, momentum=0.5)
    model.compile(optimizer = opt, 
              loss = 'categorical_crossentropy', 
              metrics=['accuracy'])
             
    return model
   
#################     
    
    
def main_part():
    global MODEL_FILE

    
    image_size = img_dim
    eval_batch_size=32
    
    train_generator_with_aug = ImageDataGenerator(
                                               horizontal_flip = True,
                                               rescale=1.0/255.0,
                                               zoom_range = 0.3, 
                                               vertical_flip = True,
                                               rotation_range=180
                                                )
    val_generator_no_aug = ImageDataGenerator(rescale=1.0/255.0)
    
    train_ds = train_generator_with_aug.flow_from_directory(
                                            directory=TRAIN_DIR,
                                            target_size=image_size,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            class_mode='categorical')

    val_ds = val_generator_no_aug.flow_from_directory(
                                            directory=VAL_DIR,
                                            target_size=image_size,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            class_mode='categorical')
                                            
    eval_ds =val_generator_no_aug.flow_from_directory(
                                            directory=VAL_DIR,
                                            target_size=image_size,
                                            batch_size=eval_batch_size,
                                            shuffle=False,
                                            class_mode='categorical')
    
    target_names=(train_ds.class_indices)
    labels=dict((v,k) for k,v in target_names.items())
    #print(labels)
    
    nb_train = len(train_ds.filenames)
    nb_val = len(val_ds.filenames)
    nb_eval = len(eval_ds.filenames)
    #print(nb_train, nb_val, nb_eval)
          
    if not os.path.exists(MODEL_FILE):
        model = make_modelZZ(input_shape=image_size + (3,), num_classes=NUM_CLASSES)
        initial_epch=0
        epoch=1
        Epochs = dEPOCHS
        bestSCORE=100
    else:
        model = load_model(MODEL_FILE)
        # Finding the epoch index from which we are resuming
        initial_epch, bestSCORE = get_init_epoch(MODEL_FILE)
        Epochs=initial_epch+dEPOCHS
        
     
    # Define Tensorboard as a Keras callback
    tensorboard = LRTensorBoard(log_dir=LOG_DIR,
                                histogram_freq=0,
                                write_images=True
                                ) 
                                
    lrate = LearningRateScheduler(time_decay,verbose=1)
                                
    diagnoser=ModelDiagnoser(batch_size,
                            6,
                            val_ds,
                            output_dir=LOG_DIR_FIGS,
                            normalization_mean=0)               
                                    
    callbacks = [
        ModelCheckpoint(MODEL_FILE_CKPT, monitor='val_accuracy'),
        tensorboard,
        lrate,
        diagnoser
    ]
    
    train_steps = train_ds.n // batch_size
    val_steps = val_ds.n // batch_size

    hist=model.fit(
            train_ds,steps_per_epoch=train_steps,
            workers=2,
            max_queue_size=64,
            epochs=Epochs, callbacks=callbacks, 
            validation_data=val_ds, validation_steps=val_steps,
            initial_epoch=initial_epch
        )
        
        MODEL_FILE=MODEL_FILE_CKPT #use the latest model for evaluation
  
 ###############   
def confusion_matrix_func(): 
    global MODEL_FILE, img_width, img_height
    eval_batch_size = 32
    #img_height = 224
    #img_width = 224 
    image_size=(img_width,img_height)     
    train_generator_no_aug = ImageDataGenerator(rescale=1.0/255.0)
    val_generator_no_aug = ImageDataGenerator(rescale=1.0/255.0)
    
    train_generator = train_generator_no_aug.flow_from_directory(
                                        directory=TRAIN_DIR,
                                        target_size=image_size,
                                        batch_size=eval_batch_size,
                                        class_mode='categorical',#'binary',
                                        shuffle=False)

    val_generator = val_generator_no_aug.flow_from_directory(
                                        directory=VAL_DIR,
                                        target_size=image_size,
                                        batch_size=eval_batch_size,
                                        class_mode='categorical',#'binary',
                                        shuffle=False)
    nb_train = len(train_generator.filenames)
    nb_val = len(val_generator.filenames)
    
    target_names = (train_generator.class_indices)
    labels=dict((v,k) for k,v in target_names.items())
    print(labels)
    
    val_target_names = (train_generator.class_indices)
    val_labels=dict((v,k) for k,v in val_target_names.items())
    print(val_labels)
    
    
    model = load_model(MODEL_FILE)
    print("train predict")

    T_pred = model.predict_generator(train_generator, steps=np.ceil(nb_train/eval_batch_size),workers=4, 
                callbacks=[LossAndErrorPrintingCallback(nb_train, eval_batch_size, train_generator.classes, train_csv)]) 

    print('Training Confusion Matrix')
    confusion_matrix=np.array(np.zeros((NUM_CLASSES,NUM_CLASSES)))
    k=list(labels.keys())
    v=list(labels.values())
    for index, probability in enumerate(T_pred):

        image_class = train_generator.filenames[index].split('/')[0]
        row=k[v.index(image_class)]
        col=np.argmax(probability)

        confusion_matrix[row,col]+=1
    print(confusion_matrix.astype(int))
    
    print('*'*15)
    print(classification_report(train_generator.classes, np.argmax(T_pred, axis=-1)))
    print('*'*15)
   

    print("val predict")
    V_pred = model.predict_generator(val_generator,steps=np.ceil(nb_val/eval_batch_size), workers=12, 
                    callbacks=[LossAndErrorPrintingCallback(nb_val, eval_batch_size, val_generator.classes, val_csv)])

    print('Validation Confusion Matrix')
    confusion_matrix=np.array(np.zeros((NUM_CLASSES,NUM_CLASSES)))
    k=list(val_labels.keys())
    v=list(val_labels.values())

    for index, probability in enumerate(V_pred):       
        image_class = val_generator.filenames[index].split('/')[0]
        row=k[v.index(image_class)]
        col=np.argmax(probability)
        confusion_matrix[row,col]+=1
        
    print(confusion_matrix.astype(int))
    
    print('*'*15)
    print(classification_report(val_generator.classes, np.argmax(V_pred, axis=-1)))
   
    print('*'*15)


#################

def eval_checkpoint():
    global MODEL_FILE, img_width, img_height 
    
    #MODEL_FILE="./Model/I_V3_.h5"#specify the model file
    model = load_model(MODEL_FILE)
    
    batch_size = 1
    #img_height = 224
    #img_width = 224 
    image_size=(img_width,img_height)
    
    ####################
    test_data_generator = ImageDataGenerator(rescale=1./255)
    test_generator = test_data_generator.flow_from_directory(
                    TEST_DIR,
                    target_size=image_size,                                            
                    batch_size=1,
                    class_mode="categorical", 
                    shuffle=False)
    filenames = test_generator.filenames

    nb_samples = len(filenames)
    print("number of samples after generator", nb_samples)
    T_prob = model.predict(test_generator, nb_samples)#TEST_SIZE
    target_names = (test_generator.class_indices)
    labels=dict((v,k) for k,v in target_names.items())
    print(labels)
    
    print('Test Confusion Matrix')
    confusion_matrix=np.array(np.zeros((NUM_CLASSES,NUM_CLASSES)))
    k=list(labels.keys())
    v=list(labels.values())
    for index, probability in enumerate(T_prob):
        image_class = test_generator.filenames[index].split('/')[0]
        row=k[v.index(image_class)]
        col=np.argmax(probability)
        
        confusion_matrix[row,col]+=1
        
    print(confusion_matrix.astype(int))
    
    
##########################
    
if __name__=="__main__":

    K.clear_session()
 
    main_part(args)
    confusion_matrix_func() # for training and validation sets
    eval_checkpoint()# for test set
