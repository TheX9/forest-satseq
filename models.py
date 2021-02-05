import argparse

import callbacks_config

import os
# Set environment variables

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ConvLSTM2D, Dense, Conv3D, Conv2D, BatchNormalization, Input, Reshape, TimeDistributed, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.layers import ZeroPadding3D, Dropout, SpatialDropout2D
from tensorflow.python.keras import backend as K
from tensorboard.plugins.hparams import api as hp

import tensorflow_addons as tfa


import datetime
import numpy as np

from load_dataset import LoadDatasetHelper
from tensorflow.keras.layers.experimental import preprocessing

def get_metrics():
    metrics = [
               tf.keras.metrics.TruePositives(name='tp'),
               tf.keras.metrics.FalsePositives(name='fp'),
               tf.keras.metrics.TrueNegatives(name='tn'),
               tf.keras.metrics.FalseNegatives(name='fn'), 
               tf.keras.metrics.BinaryAccuracy(name='accuracy'),
               tf.keras.metrics.Precision(name='precision'),
               tf.keras.metrics.Recall(name='recall'),
               jacard_coef_bool,
               jacard_coef,
               tf.keras.metrics.AUC(name='auc_pr', curve='PR')
               ]

    return metrics

def jacard_coef_bool(y_true, y_pred):
    y_true_f = K.cast(K.flatten(y_true) >= 0.5, dtype='float32')
    y_pred_f = K.cast(K.flatten(y_pred) >= 0.5, dtype='float32')
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def get_class_weights_and_bias(beta):
    samples_per_cls = np.array([64*64*0.991, 64*64*0.009])
    n_classes = 2
    total_n = (64*64)

    # Calculate class weights
    if beta == 1:
        weights = (total_n)/samples_per_cls
    else:
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta)/np.array(effective_num)
    class_weights = weights / np.sum(weights) * n_classes

    # Set initial bias
    initial_bias = np.log([samples_per_cls[1]/samples_per_cls[0]])
    initial_bias = tf.keras.initializers.Constant(initial_bias)

    # Define weighted bce
    def weighted_bce(y_true, y_pred):
        weights = (y_true * class_weights[1]) + class_weights[0]*(1-y_true)
        bce = K.binary_crossentropy(y_true, y_pred)
        weighted_bce = K.mean(bce * weights)
        return weighted_bce

    return weighted_bce, initial_bias

    
def get_dropout(model_name, input_layer, name, dropout_rate, time_dist = True):

    if dropout_rate > 0:
        if 'spatial' in model_name:
            if time_dist:
                output_layer = TimeDistributed(SpatialDropout2D(dropout_rate), name = name)(input_layer)
            else:
                output_layer = SpatialDropout2D(dropout_rate, name = name)(input_layer)
        else:
            if time_dist:
                output_layer = TimeDistributed(Dropout(dropout_rate), name = name)(input_layer)
            else:
                output_layer = Dropout(dropout_rate, name = name)(input_layer)
    else:
        output_layer = input_layer
    return output_layer

def get_reduced_model(model_name, input_layer, dropout_rate, tile_size):
    # ---Encoder path---
    encode_1 = get_encode_conv2d_bn(input_layer, 1, 64, reps = 2)
    enc_dropout_1 = get_dropout(model_name, encode_1, 'encode_dropout_1', dropout_rate)

    max_pool1 = TimeDistributed(MaxPooling2D(2), name = f'maxpool_1')(enc_dropout_1)
    encode_2 = get_encode_conv2d_bn(max_pool1, 2, 128, reps = 2)
    enc_dropout_2 = get_dropout(model_name, encode_2, 'encode_dropout_2', dropout_rate)
    
    max_pool2 = TimeDistributed(MaxPooling2D(2), name = f'maxpool_2')(enc_dropout_2)
    encode_3 = get_encode_conv2d_bn(max_pool2, 3, 256, reps = 4)
    enc_dropout_3 = get_dropout(model_name, encode_3, 'encode_dropout_3', dropout_rate)
    
    max_pool3 = TimeDistributed(MaxPooling2D(2), name = f'maxpool_3')(enc_dropout_3)
    encode_4 = get_encode_conv2d_bn(max_pool3, 4, 256, reps = 4)
    enc_dropout_4 = get_dropout(model_name, encode_4, 'encode_dropout_4', dropout_rate)

    # --- Forecasting Module ---
    forecasting_module = get_forecasting_module(model_name, enc_dropout_4, 256, dropout_rate, tile_size)

    # --- Decoder path --- 
    up2 = up_conv_and_concat_last_frame(forecasting_module, 256, encode_3, 2)   
    dec2 = get_decode_conv(up2, 2, 128, reps = 4)
    dec_dropout_2 = get_dropout(model_name, dec2, 'decode_dropout_2', dropout_rate, time_dist = False)

    up3 = up_conv_and_concat_last_frame(dec_dropout_2, 128, encode_2, 3)
    dec3 = get_decode_conv(up3, 3, 64, reps = 4)
    dec_dropout_3 = get_dropout(model_name, dec3, 'decode_dropout_3', dropout_rate, time_dist = False)

    up4 = up_conv_and_concat_last_frame(dec_dropout_3, 64, encode_1, 4)
    dec4 = get_decode_conv(up4, 4, 64, reps = 1)
    dec_dropout_4 = get_dropout(model_name, dec4, 'decode_dropout_4', dropout_rate, time_dist = False)

    return dec_dropout_4


def get_double_reduced_model(model_name, input_layer, dropout_rate, tile_size):
    # ---Encoder path---
    encode_2 = get_encode_conv2d_bn(input_layer, 2, 64, reps = 2)
    enc_dropout_2 = get_dropout(model_name, encode_2, 'encode_dropout_2', dropout_rate)
    
    max_pool2 = TimeDistributed(MaxPooling2D(2), name = f'maxpool_2')(enc_dropout_2)
    encode_3 = get_encode_conv2d_bn(max_pool2, 3, 128, reps = 4)
    enc_dropout_3 = get_dropout(model_name, encode_3, 'encode_dropout_3', dropout_rate)
    
    max_pool3 = TimeDistributed(MaxPooling2D(2), name = f'maxpool_3')(enc_dropout_3)
    encode_4 = get_encode_conv2d_bn(max_pool3, 4, 128, reps = 4)
    enc_dropout_4 = get_dropout(model_name, encode_4, 'encode_dropout_4', dropout_rate)

    # --- Forecasting Module ---
    forecasting_module = get_forecasting_module(model_name, enc_dropout_4, 128, dropout_rate, tile_size)

    # --- Decoder path --- 
    up2 = up_conv_and_concat_last_frame(forecasting_module, 256, encode_3, 2)   
    dec2 = get_decode_conv(up2, 2, 128, reps = 4)
    dec_dropout_2 = get_dropout(model_name, dec2, 'decode_dropout_2', dropout_rate, time_dist = False)

    up3 = up_conv_and_concat_last_frame(dec_dropout_2, 128, encode_2, 3)
    dec3 = get_decode_conv(up3, 3, 64, reps = 4)
    dec_dropout_3 = get_dropout(model_name, dec3, 'decode_dropout_3', dropout_rate, time_dist = False)

    return dec_dropout_3

# --- Forecasting Module ---
def get_forecasting_module(model_name, input_layer, num_units, dropout_rate, tile_size):
    if 'conv3d' in model_name:
        # Padding
        zero_padding = ZeroPadding3D(padding = (0,1,1))(input_layer)

        # Forecasting Module
        conv_3d = Conv3D(num_units, kernel_size = (6, 3, 3), padding='valid', name='forecasting_conv3d')(zero_padding)
        
        # Batch norm
        batch_norm = BatchNormalization(name = 'batchnorm_forecasting')(conv_3d)

        # Reshape
        if 'double_reduced' in model_name:
            reshape = Reshape(target_shape = (int(tile_size/4), int(tile_size/4), num_units))(batch_norm)
        elif 'reduced' in model_name:
            reshape = Reshape(target_shape = (int(tile_size/8), int(tile_size/8), num_units))(batch_norm)

        # Dropout
        dropout = get_dropout(model_name, reshape, 'dropout_forecasting', dropout_rate, time_dist = False)

        # Forecasting module
        forecasting_module = dropout

    else:
        # Forcasting module
        convlstm2d = ConvLSTM2D(filters = num_units,
                                kernel_size=(3, 3),
                                padding="same",
                                return_sequences=False,
                                dropout=dropout_rate,
                                recurrent_dropout=dropout_rate,
                                name='forecasting_convlstm2d')(input_layer)
        # Batch norm
        batch_norm = BatchNormalization(name = 'batchnorm_forecasting')(convlstm2d)

        # Forecasting module
        forecasting_module = batch_norm

    return forecasting_module

# Encoder path 
def get_encode_conv2d_bn(prev_layer, encode_i, feature_channels, reps = 1):     
    for rep in range(reps):
        conv = TimeDistributed(Conv2D(feature_channels,
                                        kernel_size = 3,
                                        activation='relu',
                                        padding = 'same'),
                                name = f'encode_{encode_i}_conv2_{rep+1}')(prev_layer)
        layer = TimeDistributed(BatchNormalization(), name = f'encode_{encode_i}_bn_{rep+1}')(conv)
        prev_layer = layer
    return layer

# Decoder path
def up_conv_and_concat_last_frame(prev_layer, num_units, concat_layer, up_id):
    # Up convolution
    upconv = Conv2DTranspose(num_units,
                                kernel_size = 2,
                                strides = 2,
                                activation='relu',
                                padding = 'same',
                                name = f'upconv_{up_id}')(prev_layer)
    
    # take last encode frame of encode layer
    last_enc_frame_encode= tf.keras.layers.Lambda(lambda x: x[:, -1, :, :, :], name = f'slice_layer_{up_id}')(concat_layer)
    concat = Concatenate(axis = -1, name = f'concat_layer_{up_id}')([last_enc_frame_encode, upconv])
    
    return concat

def get_decode_conv(prev_layer, decode_i, feature_channels, reps = 1):
    for rep in range(reps):
        conv = Conv2D(feature_channels,
                        kernel_size = 3,
                        activation='relu',
                        padding = 'same',
                        name = f'decode_{decode_i}_conv2_{rep+1}')(prev_layer)
        layer = BatchNormalization(name = f'decode_{decode_i}_bn_{rep+1}')(conv)
        prev_layer = layer
    return layer

def get_model(model_name, keras_norm_params, dataset, hparams, tile_size):
 
    if 'comb_img_only' in dataset:
        input_size = (6, tile_size, tile_size, 6)
    else:
        input_size = (6, tile_size, tile_size, 8)

    # Input layer
    input_layer = Input(shape=input_size, name = 'in_seq')

    if keras_norm_params:
        print('Using keras normalization')
        x = preprocessing.Normalization(mean=keras_norm_params[0], variance=keras_norm_params[1])(input_layer)
    else:
        x = input_layer

    # Hyperparameters
    # dropout_rate = 0.2
    # beta = 0.9
    # learning_rate = 0.01

    dropout_rate = hparams[HP_DROPOUT]
    beta = hparams[HP_BETA]
    learning_rate = hparams[HP_LEARNING_RATE]

    # Select model architecture
    if 'double_reduced' in model_name:
        last_layer = get_double_reduced_model(model_name, x, dropout_rate, tile_size)
    elif 'reduced' in model_name:
        last_layer = get_reduced_model(model_name, x, dropout_rate, tile_size)

    # Get loss function and initial bias
    weighted_bce, initial_bias = get_class_weights_and_bias(beta)

    # Output
    output_layer = Conv2D(1, kernel_size = 3, activation='sigmoid', bias_initializer=initial_bias, padding = 'same', name = 'out')(last_layer)

    # Get model metrics
    metrics = get_metrics()

    #  Define model
    model = Model(inputs=input_layer, outputs=output_layer)

    #model.summary(line_length=150) 

    model.compile(loss=weighted_bce,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=metrics)

    return model


### Run Configuration ###
# Configure hyperparameter search
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]))
HP_BETA= hp.HParam('beta', hp.Discrete([0.9, 0.99, 0.999, 0.9999, 1.0]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.1, 0.01, 0.001]))

# Train metrics
METRIC_TRAIN_TP='Train_tp'
METRIC_TRAIN_FP='Train_fp'
METRIC_TRAIN_TN='Train_tn'
METRIC_TRAIN_FN='Train_fn'
METRIC_TRAIN_ACCURACY='Train_Accuracy'
METRIC_TRAIN_PRECISION='Train_Precision'
METRIC_TRAIN_RECALL='Train_Recall'
METRIC_TRAIN_F1 = 'Train_F1'
METRIC_TRAIN_JAC_COEF_BOOL='Train_jac_coef_bool'
METRIC_TRAIN_JAC_COEF='Train_jac_coef'
METRIC_TRAIN_AUC_PR='Train_AUC_PR'

# Val metrics
METRIC_VAL_TP = 'val_tp'
METRIC_VAL_FP = 'val_fp'
METRIC_VAL_TN = 'val_tn'
METRIC_VAL_FN = 'val_fn'
METRIC_VAL_ACCURACY = 'val_accuracy'
METRIC_VAL_PRECISION = 'val_precision'
METRIC_VAL_RECALL = 'val_recall'
METRIC_VAL_F1 = 'val_f1'
METRIC_VAL_JAC_COEF_BOOL = 'val_jac_coef_bool'
METRIC_VAL_JAC_COEF = 'val_jac_coef'
METRIC_VAL_AUC_PR = 'val_auc_pr'

# Test metrics
METRIC_TEST_TP = 'test_tp'
METRIC_TEST_FP = 'test_fp'
METRIC_TEST_TN = 'test_tn'
METRIC_TEST_FN = 'test_fn'
METRIC_TEST_ACCURACY = 'test_accuracy'
METRIC_TEST_PRECISION = 'test_precision'
METRIC_TEST_RECALL = 'test_recall'
METRIC_TEST_F1 = 'test_f1'
METRIC_TEST_JAC_COEF_BOOL = 'test_jac_coef_bool'
METRIC_TEST_JAC_COEF = 'test_jac_coef'
METRIC_TEST_AUC_PR = 'test_auc_pr'

class RunHandler:
    def __init__(self,
                model_name,
                mode = '64px_comb_img_only',
                tile_size = 64,
                batch_size = 64,
                dataset_file_path = 'data/datasets/main/2013_2019/64px_comb',
                no_early_stopping = False,
                lr_scheduler = False):

        self.model_name = model_name
        self.mode = mode
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.dataset_file_path = dataset_file_path
        self.no_early_stopping = no_early_stopping
        self.lr_scheduler = lr_scheduler

        # Init dataset
        self.datasets = self.init_dataset(mode)
        
        if 'keras_normalization' in self.model_name:
            self.keras_norm_params = self.init_dataset_norm_layer()
        else:
            self.keras_norm_params = None

    def init_dataset_norm_layer(self):
        feature_ds = self.datasets['train'].map(lambda x, y: x)
        normalizer = preprocessing.Normalization()
        normalizer.adapt(feature_ds)
        return normalizer.mean.numpy(), normalizer.variance.numpy()
        
    def init_dataset(self, mode):
        print(mode)

        load_dataset_helper = LoadDatasetHelper(file_path=self.dataset_file_path,
                                            baseline=False,
                                            check_dataset = False,
                                            train_batch_size = self.batch_size,
                                            val_test_batch_size= self.batch_size,
                                            mode = mode
                                            )
        load_dataset_helper.sample_weights = []

        if 'keras_normalization' in self.model_name:
            load_dataset_helper.normalization = False
        else:
            load_dataset_helper.normalization = True
        
        load_dataset_helper.complete_label = False
        load_dataset_helper.kernel_size = self.tile_size
        datasets = load_dataset_helper.get_datasets()
        return datasets

    def run(self, run_dir, hparams, model_name, dataset, file_dir):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            
            train_metrics, val_metrics, test_metrics = self.train_model(hparams, model_name, dataset, file_dir)

            # train metrics
            train_tp, train_fp, train_tn, train_fn, train_accuracy, train_precision, train_recall, train_jac_coef_bool, train_jac_coef, train_pr_auc = train_metrics

            tf.summary.scalar(METRIC_TRAIN_TP, train_tp, step=1) 
            tf.summary.scalar(METRIC_TRAIN_FP, train_fp, step=1)
            tf.summary.scalar(METRIC_TRAIN_TN, train_tn, step=1)
            tf.summary.scalar(METRIC_TRAIN_FN, train_fn, step=1)
            tf.summary.scalar(METRIC_TRAIN_ACCURACY, train_accuracy, step=1)
            tf.summary.scalar(METRIC_TRAIN_PRECISION, train_precision, step=1)
            tf.summary.scalar(METRIC_TRAIN_RECALL, train_recall, step=1)
            if train_precision+train_recall == 0:
                train_f1 = 0
            else:
                train_f1 = 2*((train_precision*train_recall)/(train_precision+train_recall))
            tf.summary.scalar(METRIC_TRAIN_F1, train_f1, step=1)
            tf.summary.scalar(METRIC_TRAIN_JAC_COEF_BOOL, train_jac_coef_bool, step=1)
            tf.summary.scalar(METRIC_TRAIN_JAC_COEF, train_jac_coef, step=1)
            tf.summary.scalar(METRIC_TRAIN_AUC_PR, train_pr_auc, step=1)

            # val metrics
            val_tp, val_fp, val_tn, val_fn, val_accuracy, val_precision, val_recall, val_jac_coef_bool, val_jac_coef, val_pr_auc = val_metrics

            tf.summary.scalar(METRIC_VAL_TP, val_tp, step=1) 
            tf.summary.scalar(METRIC_VAL_FP, val_fp, step=1)
            tf.summary.scalar(METRIC_VAL_TN, val_tn, step=1)
            tf.summary.scalar(METRIC_VAL_FN, val_fn, step=1)
            tf.summary.scalar(METRIC_VAL_ACCURACY, val_accuracy, step=1)
            tf.summary.scalar(METRIC_VAL_PRECISION, val_precision, step=1)
            tf.summary.scalar(METRIC_VAL_RECALL, val_recall, step=1)
            if val_precision+val_recall == 0:
                val_f1 = 0
            else:
                val_f1 = 2*((val_precision*val_recall)/(val_precision+val_recall))
            tf.summary.scalar(METRIC_VAL_F1, val_f1, step=1)
            tf.summary.scalar(METRIC_VAL_JAC_COEF_BOOL, val_jac_coef_bool, step=1)
            tf.summary.scalar(METRIC_VAL_JAC_COEF, val_jac_coef, step=1)
            tf.summary.scalar(METRIC_VAL_AUC_PR, val_pr_auc, step=1)

            # test metrics
            test_tp, test_fp, test_tn, test_fn, test_accuracy, test_precision, test_recall, test_jac_coef_bool, test_jac_coef, test_pr_auc = test_metrics
            
            tf.summary.scalar(METRIC_TEST_TP, test_tp, step=1) 
            tf.summary.scalar(METRIC_TEST_FP, test_fp, step=1)
            tf.summary.scalar(METRIC_TEST_TN, test_tn, step=1)
            tf.summary.scalar(METRIC_TEST_FN, test_fn, step=1)
            tf.summary.scalar(METRIC_TEST_ACCURACY, test_accuracy, step=1)
            tf.summary.scalar(METRIC_TEST_PRECISION, test_precision, step=1)
            tf.summary.scalar(METRIC_TEST_RECALL, test_recall, step=1)
            if test_precision+test_recall == 0:
                test_f1 = 0
            else:
                test_f1 = 2*((test_precision*test_recall)/(test_precision+test_recall))
            tf.summary.scalar(METRIC_TEST_F1, test_f1, step=1)
            tf.summary.scalar(METRIC_TEST_JAC_COEF_BOOL, test_jac_coef_bool, step=1)
            tf.summary.scalar(METRIC_TEST_JAC_COEF, test_jac_coef, step=1)
            tf.summary.scalar(METRIC_TEST_AUC_PR, test_pr_auc, step=1)



    def start_trial(self, model_name, dataset, file_dir):
        session_num = 0
        for dropout in HP_DROPOUT.domain.values:
            for beta in HP_BETA.domain.values:
                for learning_rate in HP_LEARNING_RATE.domain.values:
                    print('weighted_bce', dropout, beta, learning_rate)
                    hparams = {
                                HP_DROPOUT: dropout,
                                HP_BETA: beta,
                                HP_LEARNING_RATE: learning_rate
                                }

                    run_name = "run-%d" % session_num
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    self.run(f'{file_dir}/hparam_tuning/' + run_name, hparams, model_name, dataset, file_dir)
                    session_num += 1

    def train_model(self, hparams, model_name, dataset_name, file_dir):
        dataset = self.datasets
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            # Get model
            model = get_model(model_name, self.keras_norm_params, dataset_name, hparams, self.tile_size)

            # Get callbacks
            dropout = hparams[HP_DROPOUT]
            beta = hparams[HP_BETA]
            learning_rate = hparams[HP_LEARNING_RATE]

            params = f'dropout_{dropout}_beta_{beta}_learning_rate_{learning_rate}'
            log_dir = f"{file_dir}/logs/fit_{params}"
            hyper_params_dir = f'{params}'
            # Create folder if not existing 
            weight_dir = f'{file_dir}/weights/{hyper_params_dir}'
            if not os.path.exists(weight_dir):
                os.makedirs(weight_dir)
            callbacks = callbacks_config.configure_callbacks(hparams, log_dir, weight_dir, no_early_stopping = self.no_early_stopping, lr_reducer= self.lr_scheduler, histogram_freq = 0)

            # Fit model
            hist = model.fit(dataset['train'],
                    epochs=EPOCHS,
                    validation_data = dataset['val'],
                    callbacks=callbacks)

            # Save final weights
            model.save_weights(f'{weight_dir}/best_model.h5')

            print(hist.history.keys())
            # Get train metrics, to check for overfitting
            train_tp = hist.history['tp'][-1]
            train_fp = hist.history['fp'][-1]
            train_tn = hist.history['tn'][-1]
            train_fn = hist.history['fn'][-1]
            train_accuracy = hist.history['accuracy'][-1]
            train_precision = hist.history['precision'][-1]
            train_recall = hist.history['recall'][-1] 
            train_jac_coef_bool = hist.history['jacard_coef_bool'][-1]
            train_jac_coef = hist.history['jacard_coef'][-1]
            train_pr_auc = hist.history['auc_pr'][-1]

            # Get val metrics
            _, val_tp, val_fp, val_tn, val_fn, val_accuracy, val_precision, val_recall, val_jac_coef_bool, val_jac_coef, val_pr_auc = model.evaluate(dataset['val'])

            # Get test metrics
            _, test_tp, test_fp, test_tn, test_fn, test_accuracy, test_precision, test_recall, test_jac_coef_bool, test_jac_coef, test_pr_auc = model.evaluate(dataset['test'])

            train_metrics = train_tp, train_fp, train_tn, train_fn, train_accuracy, train_precision, train_recall, train_jac_coef_bool, train_jac_coef, train_pr_auc
            val_metrics = val_tp, val_fp, val_tn, val_fn, val_accuracy, val_precision, val_recall, val_jac_coef_bool, val_jac_coef, val_pr_auc
            test_metrics = test_tp, test_fp, test_tn, test_fn, test_accuracy, test_precision, test_recall, test_jac_coef_bool, test_jac_coef, test_pr_auc

            return (train_metrics, val_metrics, test_metrics)


def main(model_name, dataset, file_dir, tile_size, batch_size, dataset_file_path, no_early_stopping, lr_reducer):
    run_handler = RunHandler(model_name, dataset, tile_size, batch_size, dataset_file_path, no_early_stopping, lr_reducer)
    run_handler.start_trial(model_name, dataset, file_dir)
if __name__ == "__main__":

    # Create parser
    parser = argparse.ArgumentParser(description='Model training')

    # Epochs
    parser.add_argument('--epochs', default=250, type=int)

    # model name
    parser.add_argument('--model_name', type=str)

    # dataset file path
    parser.add_argument('--dataset_file_path', type=str, default='data/datasets/main/2013_2019/64px_comb')

    # dataset
    parser.add_argument('--dataset', type=str, default = '64px_comb_img_only')
    
    # dataset
    parser.add_argument('--tile_size', type=int, default = 64)

    # dataset
    parser.add_argument('--batch_size', type=int, default = 64)

    # dropout
    parser.add_argument('--hp_dropout', nargs='+', type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    # beta
    parser.add_argument('--hp_beta', nargs='+', type=float, default=[0.0, 0.99, 0.995, 0.999, 0.9995, 1.0])

    # learning rate
    parser.add_argument('--hp_learning_rate', nargs='+', type=float, default=[0.1, 0.01, 0.001])

    # determine early stopping
    parser.add_argument('--no_early_stopping', action="store_true")

    # determine lr scheduler
    parser.add_argument('--lr_reducer', action="store_true")

    # Evaluate argparse object
    args = parser.parse_args()

    EPOCHS = args.epochs
    model_name = args.model_name
    dataset = args.dataset
    tile_size = args.tile_size
    batch_size = args.batch_size
    dataset_file_path = args.dataset_file_path
    no_early_stopping = args.no_early_stopping
    lr_reducer = args.lr_reducer

    file_dir = f'models/{model_name}/{dataset}'
    
    hp_dropout = args.hp_dropout
    hp_beta = args.hp_beta
    hp_learning_rate = args.hp_learning_rate

    HP_DROPOUT = hp.HParam('dropout', hp.Discrete(hp_dropout))
    HP_BETA= hp.HParam('beta', hp.Discrete(hp_beta))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete(hp_learning_rate))

    with tf.summary.create_file_writer(f'{file_dir}/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_DROPOUT,
                     HP_BETA,
                     HP_LEARNING_RATE
                     ],
            metrics=[
                     # Train metrics
                     hp.Metric(METRIC_TRAIN_TP, display_name='Train_tp'),
                     hp.Metric(METRIC_TRAIN_FP, display_name='Train_fp'),
                     hp.Metric(METRIC_TRAIN_TN, display_name='Train_tn'),
                     hp.Metric(METRIC_TRAIN_FN, display_name='Train_fn'),
                     hp.Metric(METRIC_TRAIN_ACCURACY, display_name='Train_Accuracy'),
                     hp.Metric(METRIC_TRAIN_PRECISION, display_name='Train_Precision'),
                     hp.Metric(METRIC_TRAIN_RECALL, display_name='Train_Recall'),
                     hp.Metric(METRIC_TRAIN_F1, display_name = 'Train_F1'),
                     hp.Metric(METRIC_TRAIN_JAC_COEF_BOOL, display_name='Train_jac_coef_bool'),
                     hp.Metric(METRIC_TRAIN_JAC_COEF, display_name='Train_jac_coef'),
                     hp.Metric(METRIC_TRAIN_AUC_PR, display_name='Train_AUC_PR'),
                     # Val metrics
                     hp.Metric(METRIC_VAL_TP, display_name='Val_tp'),
                     hp.Metric(METRIC_VAL_FP, display_name='Val_fp'),
                     hp.Metric(METRIC_VAL_TN, display_name='Val_tn'),
                     hp.Metric(METRIC_VAL_FN, display_name='Val_fn'),
                     hp.Metric(METRIC_VAL_ACCURACY, display_name='Val_Accuracy'),
                     hp.Metric(METRIC_VAL_PRECISION, display_name='Val_Precision'),
                     hp.Metric(METRIC_VAL_RECALL, display_name='Val_Recall'),
                     hp.Metric(METRIC_VAL_F1, display_name = 'Val_F1'),
                     hp.Metric(METRIC_VAL_JAC_COEF_BOOL, display_name='Val_jac_coef_bool'),
                     hp.Metric(METRIC_VAL_JAC_COEF, display_name='Val_jac_coef'),
                     hp.Metric(METRIC_VAL_AUC_PR, display_name='Val_AUC_PR'),
                     # Test metrics
                     hp.Metric(METRIC_TEST_TP, display_name='Test_tp'),
                     hp.Metric(METRIC_TEST_FP, display_name='Test_fp'),
                     hp.Metric(METRIC_TEST_TN, display_name='Test_tn'),
                     hp.Metric(METRIC_TEST_FN, display_name='Test_fn'),
                     hp.Metric(METRIC_TEST_ACCURACY, display_name='Test_Accuracy'),
                     hp.Metric(METRIC_TEST_PRECISION, display_name='Test_Precision'),
                     hp.Metric(METRIC_TEST_RECALL, display_name='Test_Recall'),
                     hp.Metric(METRIC_TEST_F1, display_name = 'Test_F1'),
                     hp.Metric(METRIC_TEST_JAC_COEF_BOOL, display_name='Test_jac_coef_bool'),
                     hp.Metric(METRIC_TEST_JAC_COEF, display_name='Test_jac_coef'),
                     ],)
    print(dataset)
    main(model_name, dataset, file_dir, tile_size, batch_size, dataset_file_path, no_early_stopping, lr_reducer)