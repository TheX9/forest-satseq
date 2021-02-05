
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

def configure_callbacks(hparams, log_dir, weight_dir, no_early_stopping = False, lr_reducer = True, histogram_freq = 1, early_stopping_patience = 25):
    callbacks = []
    # Initialize model checkpoint
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(weight_dir + "/weights.{epoch:02d}-{val_loss:.2f}-{val_precision:.2f}-{val_recall:.2f}.hdf5",
                                                        monitor='val_loss',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        mode='auto')



    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                            histogram_freq=histogram_freq,
                                                            write_grads=True,
                                                            write_images=True)

    # Learning rate scheduler
    #def scheduler(epoch, lr):
    #    if epoch < 10:
    #        return lr
    #    else:
    #        return lr * tf.math.exp(-0.1)
    #
    # learning_rate_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    # Hyperparameter search callback
    hparams_callback = hp.KerasCallback(log_dir, hparams)

    callbacks = [model_checkpoint, tensorboard_callback, hparams_callback]

    if not no_early_stopping:
        print('early stopping')
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stopping_patience, verbose=0,
                                                        mode='auto', baseline=None, restore_best_weights=True
                                                        )

        callbacks.append(early_stopping_callback)

    if lr_reducer:
        print('Using lr reducer')
        # Control the learning rate on plateau
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                    factor=0.1,
                                                                    patience=5,
                                                                    min_lr=0.001)

        callbacks.append(reduce_lr_callback)

    return callbacks