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
import threading
import time


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
    if beta > 1:
        weights = (total_n)/samples_per_cls*beta
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


def get_model(model_name, keras_norm_params, dataset, hparams, tile_size):
 
    beta = hparams[HP_BETA]
    learning_rate = hparams[HP_LEARNING_RATE]

    # Input layer
    if 'add_feat' in dataset:
        print('used additional features')
        input_layer= Input(shape=(11), name = 'in_bas')
    else:
        input_layer= Input(shape=(9), name = 'in_bas')
    
    #point_id = Input(shape=(1), name = 'point_id')
    #px_id = Input(shape=(1), name = 'px_id')

    # Get loss function and initial bias
    weighted_bce, initial_bias = get_class_weights_and_bias(beta)

    # Output layer
    output_layer = Dense(1, activation='sigmoid', bias_initializer=initial_bias, name = 'out')(input_layer)

    # Get model metrics
    metrics = get_metrics()

    #  Define model
    model = Model(inputs=input_layer, outputs=output_layer)

    model.summary(line_length=150) 

    model.compile(loss=weighted_bce,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=metrics)

    return model

### Run Configuration ###
# Configure hyperparameter search
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
    def __init__(self, model_name, mode, tile_size, batch_size, dataset_file_path):
        self.model_name = model_name
        self.mode = mode
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.dataset_file_path = dataset_file_path

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
        load_dataset_helper = LoadDatasetHelper(file_path=self.dataset_file_path,
                                                baseline=True, check_dataset = False,
                                                train_batch_size = self.batch_size, val_test_batch_size = self.batch_size*10,
                                                mode = mode)
        load_dataset_helper.sample_weights = []
        load_dataset_helper.normalization = True        
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
        self.thread_list = []

        session_num = 0
        for learning_rate in HP_LEARNING_RATE.domain.values:
            for beta in HP_BETA.domain.values:
                    print('weighted_bce', beta, learning_rate)
                    hparams = {
                                HP_BETA: beta,
                                HP_LEARNING_RATE: learning_rate
                                }

                    run_name = f'run-beta_{beta}-lr_{learning_rate}'
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})

                    # Start run in threads
                    self.run(f'{file_dir}/hparam_tuning/' + run_name, hparams, model_name, dataset, file_dir)
                    
                    session_num += 1
        

    def train_model(self, hparams, model_name, dataset_name, file_dir):
        dataset = self.datasets
        # Get model
        model = get_model(model_name, self.keras_norm_params, dataset_name, hparams, self.tile_size)

        # Get callbacks
        beta = hparams[HP_BETA]
        learning_rate = hparams[HP_LEARNING_RATE]

        params = f'beta_{beta}_learning_rate_{learning_rate}'
        log_dir = f"{file_dir}/logs/fit_{params}"
        hyper_params_dir = f'{params}'
        # Create folder if not existing 
        weight_dir = f'{file_dir}/weights/{hyper_params_dir}'
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        callbacks = callbacks_config.configure_callbacks(hparams, log_dir, weight_dir, no_early_stopping = False, histogram_freq = 0, early_stopping_patience = 5)

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

def main(model_name, dataset, file_dir, tile_size, batch_size, dataset_file_path):
    run_handler = RunHandler(model_name, dataset, tile_size, batch_size, dataset_file_path)
    run_handler.start_trial(model_name, dataset, file_dir)
if __name__ == "__main__":

    # Create parser
    parser = argparse.ArgumentParser(description='Model training')

    # Epochs
    parser.add_argument('--epochs', default=250, type=int)

    # model name
    parser.add_argument('--model_name', type=str)

    # dataset file path
    parser.add_argument('--dataset_file_path', type=str, default='data/datasets/baseline/2013_2019/2013_2019_full_64px')

    # dataset
    parser.add_argument('--dataset', type=str, default = '64px')
    
    # dataset
    parser.add_argument('--tile_size', type=int, default = 64)

    # dataset
    parser.add_argument('--batch_size', type=int, default = 2048)

    # beta
    parser.add_argument('--hp_beta', nargs='+', type=float, default=[0.995, 0.999, 0.9995, 0.9999, 1.0, 0.0, 0.9, 0.99, ])

    # learning rate
    parser.add_argument('--hp_learning_rate', nargs='+', type=float, default=[0.01,0.1,0.001])


    # Evaluate argparse object
    args = parser.parse_args()

    EPOCHS = args.epochs
    model_name = args.model_name
    dataset = args.dataset
    tile_size = args.tile_size
    batch_size = args.batch_size
    dataset_file_path = args.dataset_file_path

    file_dir = f'models/{model_name}/{dataset}'
    
    hp_beta = args.hp_beta
    hp_learning_rate = args.hp_learning_rate

    HP_BETA= hp.HParam('beta', hp.Discrete(hp_beta))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete(hp_learning_rate))

    hparam_path = f'{file_dir}/hparam_tuning/'
    if not os.path.exists(hparam_path):
        with tf.summary.create_file_writer(hparam_path).as_default():
            hp.hparams_config(
                hparams=[HP_BETA,
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
    main(model_name, dataset, file_dir, tile_size, batch_size, dataset_file_path)