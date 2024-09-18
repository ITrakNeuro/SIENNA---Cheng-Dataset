from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dropout
from sklearn.utils import shuffle
from keras import backend as K
import cv2
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from intensity_redistribution import adaptive_MRI_enhancement
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from adversarial_training import AdversarialGenerator

# SIENNA Hyperparameter Tuning and CNN Training Script
# Key Functions:
# 1. create_model: Defines the CNN architecture and trains the model using hyperparameter tuning.
# 2. data: Loads and preprocesses the medical image data for training, testing, and cross-validation.
# 3. run_hyperparameter_tuning: Initiates hyperparameter tuning to find the optimal model configuration.
# 4. round_int: Utility function to round a floating-point number to an integer.
# 5. recall: Custom metric function for computing recall during model training.
# reference - Hyperas: Simple Hyperparameter Tuning for Keras Models, Max Pumperla, https://github.com/maxpumperla/hyperas/tree/master

dense_layer=3
def create_model(x_train, y_train, x_test, y_test):
    """
    Create and train SIENNA's convolutional neural network (CNN) model using hyperparameter tuning.
    Args:
        x_train: Training data input features (images)
        y_train: Training data labels (corresponding classes for each image)
        x_test: Testing data input features (images)
        y_test: Testing data labels (corresponding classes for each image)
    Returns:
        Dictionary with loss, status, and trained model
    """
    gpus = tf.config.experimental.list_physical_devices('GPU') #GPU configuration
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    #dynamic model initiation
    model = models.Sequential()
    #first conv block, with pre-defined parameter search spaces to be tuned by HYPERAS
    model.add(layers.Conv2D(filters={{choice([10, 12, 16, 24, 33, 39, 48, 64,74, 87, 96,110, 115,120, 128, 200])}}, kernel_size={{choice([(2,2),(3,3), (4,4), (5,5),(6,6), (7,7), (8,8)])}} , strides=(4,4),padding='same',
                            kernel_regularizer =tf.keras.regularizers.l2( l={{choice([(0.01),(0.001), (0.0001), (0.00001),(0.005), (0.0005), (0.5)])}}),
                            input_shape=(240,240, dense_layer),name='cv1'))
    model.add(layers.Activation({{choice(['relu','selu','elu', 'tanh', 'exponential', 'LeakyReLU'])}},name='activ1'))
    model.add(layers.MaxPooling2D((2, 2), padding='same',name='mpool1'))
    model.add(Dropout({{choice([0.1,0.2,0.3,0.4,0.5])}},name='drp1'))

    #second conv block, with pre-defined parameter spaces to be tuned by HYPERAS
    model.add(layers.Conv2D(filters={{choice([10, 12, 16, 24, 33, 39, 48, 64,74, 87, 96,110, 115,120, 128, 200])}}, kernel_size={{choice([(2,2),(3,3), (4,4), (5,5),(6,6), (7,7), (8,8)])}} , strides=(4,4),padding='same',
                            kernel_regularizer =tf.keras.regularizers.l2( l={{choice([(0.01),(0.001), (0.0001), (0.00001),(0.005), (0.0005), (0.5)])}}),
                            input_shape=(240,240, dense_layer),name='cv2'))
    model.add(layers.Activation({{choice(['relu','selu','elu', 'tanh', 'exponential', 'LeakyReLU'])}},name='activ2'))
    model.add(layers.MaxPooling2D((2, 2), padding='same',name='mpool2'))
    model.add(Dropout({{choice([0.1,0.2,0.3,0.4,0.5])}},name='drp2'))

    #third conv block, with pre-defined parameter spaces to be tuned by HYPERAS
    model.add(layers.Conv2D({{choice([10, 12, 16, 24, 33, 39, 48, 64,74, 87, 96,110, 115,120, 128, 200])}}, kernel_size={{choice([(2,2),(3,3), (4,4), (5,5),(6,6), (7,7), (8,8)])}} , strides=(4,4),padding='same',
                            kernel_regularizer =tf.keras.regularizers.l2( l={{choice([(0.01),(0.001), (0.0001), (0.00001),(0.005), (0.0005), (0.5)])}}),
                            input_shape=(240,240, dense_layer),name='cv3'))
    model.add(layers.Activation({{choice(['relu','selu','elu', 'tanh', 'exponential', 'LeakyReLU'])}},name='activ3'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'),name='mpool3')
    model.add(Dropout({{choice([0.1,0.2,0.3,0.4,0.5])}},name='drp2'))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, {{choice(['relu', 'sigmoid'])}},name='fc'))
    model.add(Dropout({{choice([0.1,0.2,0.3,0.4,0.5])}}))
    model.add(layers.Dense(dense_layer,name='2fc'))
    model.add(layers.Activation("softmax"))

    model.compile(loss='categorical_crossentropy', optimizer= 'Adam', metrics=['acc',tf.keras.metrics.FalseNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.Recall()])
    #callback for early stopping
    callbacks = [EarlyStopping(monitor='val_loss', patience=2), ModelCheckpoint(filepath=f'model/optimised_model.h5', monitor='val_loss', save_best_only=True)]
    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    adv_generator = AdversarialGenerator(model=model, x=x_train, y=y_train, loss_function=loss_function, eps=0.1,stabilization_factor=1e-8, plot_perturbation=False, plot_example=False, training_iterations=1)
    x_train_adv, y_train_adv=adv_generator.generate()
    x_new_train = np.concatenate((x_train, x_train_adv), axis=0)
    y_new_train = np.concatenate((y_train, y_train_adv), axis=0)
    #model training
    history = model.fit(x_new_train,y_new_train,
                batch_size={{choice([16, 32, 64])}},
                epochs={{choice([15,25])}},
                verbose=2,
                validation_data=(x_test, y_test),callbacks=callbacks)
    validation_acc = max(history.history['val_acc'])
    score,acc,fn,fp,recall = model.evaluate(x_test, y_test) 
    return {'loss': score+fp+fn, 'status': STATUS_OK, 'model': model}

def data():
    """
    Load and preprocess data for model training.
    Returns:
        x_train, y_train, x_test, y_test, x_cv, y_cv
    """
    global dense_layer
    #data path
    directory = '/content/SIENNA---CD2/Axial'
    x = []
    y = []
    label=0
    class_count = {0: 0, 1: 0, 2: 0}
    max_per_class=200

    for i in os.listdir(directory):
            if 'meningioma' in i:
                label = 1
            elif 'glioma' in i:
                label = 2
            else:
                label = 0
            for p in os.listdir(os.path.join(directory, i)):
                if class_count[label] < max_per_class:
                    image_path = os.path.join(directory, i, p)
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Failed to load image {image_path}")
                        continue
                    img = cv2.resize(img, (240, 240))
                    he = adaptive_MRI_enhancement(img)  
                    img2 = he.PREMO()  # PREMO
                    img = img2 / 255.0  # Normalization
                    x.append(img)
                    y.append(label)
                    class_count[label] += 1
                    print(image_path)
    y_train = to_categorical(y, dense_layer)
    x_shuffle, y_shuffle = shuffle(x, y_train)
    #training-vaalidation-test split as 60-20-20
    xa, x_test, ya, y_test = train_test_split(x_shuffle,y_shuffle,test_size=0.2,train_size=0.8)
    x_train, x_cv, y_train, y_cv = train_test_split(xa,ya,test_size = 0.25,train_size =0.75)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    x_cv = np.array(x_cv)
    y_cv = np.array(y_cv)
    #SMOTE data augementation to balance minority classes MET and Non-Tumor
    sm = SMOTE()
    train_rows=len(x_train)
    x_train = x_train.reshape(train_rows,-1)
    x_train, y_train = sm.fit_resample(x_train, y_train)
    x_train = x_train.reshape(-1,240,240,3)
    return x_train,y_train,x_test,y_test,x_cv,y_cv

def round_int(x):
    """
    Round a floating-point number to an integer.
    Args:
        x: Floating-point number
    Returns:
        Rounded integer or NaN if input is infinity
    """
    if x in [float("-inf"),float("inf")]: 
        return float("nan")
    return int(round(x))

def recall(y_true, y_pred):
    """
    Compute recall metric.
    Args:
        y_true: True labels
        y_pred: Predicted labels
    Returns:
        Recall value
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fn(y_true, y_pred):
    """
    Compute false negative metric.
    Args:
        y_true: True labels
        y_pred: Predicted labels
    Returns:
        False negative value
    """
    return K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

def fp(y_true, y_pred):
    """
    Compute false positive metric.
    Args:
        y_true: True labels
        y_pred: Predicted labels
    Returns:
        false positive value
    """
    return K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

def run_hyperparameter_tunning(type):
    """
    Run hyperparameter tuning for the SIENNA model.
    Args:
        type: Number of dense layers for the model
    Returns:
        Trials object, best_model, best_run, x_train, y_train, x_test, y_test, x_cv, y_cv
    """
    global dense_layer
    dense_layer=type
    trials = Trials()
    x_train, y_train, x_test, y_test,x_cv,y_cv= data()
    #running parameter optiimization
    best_run, best_model = optim.minimize(model=create_model,
                                        data=data,
                                        functions=[recall],
                                        algo=tpe.suggest,
                                        max_evals=30,
                                        trials=trials)
    
    return trials,best_model,best_run,x_train, y_train, x_test, y_test,x_cv,y_cv
        