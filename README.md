# Future Forest Loss Prediction

This repository contains code used for my master thesis "Predicting Future Deforestation Using Satellite Images".

The figure shows prediction results of different models. Each row represents a different test patch, while the first column shows the ground truth and the other columns show the prediction of the baseline and the proposed models. Yellow pixels denote forest loss.
![alt text](https://github.com/thex9/forest-satseq/blob/master/images/top_pred.png)
# Files
## Configuration
**config.py:** is used to specify general settings.
 <br/>**logging_config.py:** logging configuration.
## Data Sampling
**Sample Year Statistics.ipynb:** is used to determine the amount of satellite imagery for different months.
 <br/>**create_non_overlapping_samples.py:** creates non-overlapping train, validation, test splits for an input of patches.
 <br/>**Stratify samples.ipynb:** is used to stratify the train, validation, test splits to have the same average forest loss proportion per patch.
 <br/>**sample_tiles.py:** samples statistics, baseline features and satellite images from patches using Google Earth Engine and saves them to a Google Drive folder specified in config.py.
 <br/>**download_dataset.py:** is used to download the from Google Earth Engine sampled patches from Google Drive.
 <br/>**process_statistic_sample.py:** processes sampled statistic patches from Google Earth Engine samples to get forest loss statistics for each patch.
 <br/>**process_sampled_tiles.py:** processes the sampled patches from Google Earth Engine to the final dataset.

## Data Visualization
**GE - Visualize Features.ipynb:** is used to visualize the different features.

## Datasets
Raw data sampled from Google Earth Engine is removed from the repository due to size limitations.
The following datasets can be downloaded from https://drive.google.com/drive/folders/1ODuUNBAGtQdCcVp8dfO5CFyWRzEBkS6t?usp=sharing:
<br/>64px, 256px dataset sampled from the deforestation fronts of the Amazon
<br/>64px, 256px dataset sampled from the whole Amazon region
<br/>
<br/>64px baseline dataset

<br/>**/data/shapefiles:** contains the shapefiles used for sampling.
<br/>**/data/stats:** contains statistics about satellite image availability and forest loss per point.

 <br/>**load_dataset.py:** is used to load the datasets.

Example for loading the 256x256px dataset placed in data/datasets/main/2013_2019/256px_comb.
"datasets" contains the train, validation and test split in TFRecordDataset format.
```python
from load_dataset import LoadDatasetHelper
load_dataset_helper = LoadDatasetHelper(file_path='data/datasets/main/2013_2019/256px_comb/',
										baseline=False,
										check_dataset = False,
										train_batch_size = 64,
										val_test_batch_size= 64,
										mode = '256px_comb'
										)
load_dataset_helper.normalization = True
load_dataset_helper.kernel_size = 256
datasets = load_dataset_helper.get_datasets()
```

Example for loading the 64x64px dataset placed in data/datasets/main/2013_2019/64px_comb.
```python
from load_dataset import LoadDatasetHelper
load_dataset_helper = LoadDatasetHelper(file_path='data/datasets/main/2013_2019/64px_comb/',
										baseline=False,
										check_dataset = False,
										train_batch_size = 64,
										val_test_batch_size= 64,
										mode = '64px_comb'
										)
load_dataset_helper.normalization = True
load_dataset_helper.kernel_size = 64
datasets = load_dataset_helper.get_datasets()
```

Example for loading the 64x64px baseline dataset placed in data/datasets/baseline/2013_2019/64px_baseline/.
```python
from load_dataset import LoadDatasetHelper
load_dataset_helper = LoadDatasetHelper(file_path='data/datasets/baseline/2013_2019/64px_baseline/',
										baseline=False,
										check_dataset = False,
										train_batch_size = 64,
										val_test_batch_size= 64,
										mode = '64px'
										)
load_dataset_helper.normalization = True
load_dataset_helper.kernel_size = 64
datasets = load_dataset_helper.get_datasets()
```


## Experiments
**callbacks_config.py:** is used to specify the Tensorflow training callbacks.
 <br/>**models.py:** handles the hyperparameter optimization of the deep learning models.
 <br/>**model_baseline:** handles the hyperparameter optimization of the baseline model.

<br/>
<br/>
The best deep learning models alongside the dataset can be loaded using:

```python
import models
def load_model(model_name, dataset, beta, lr, dropout, dataset_file_path):
    dl_runhandler = models.RunHandler(model_name = model_name, mode = dataset, dataset_file_path = dataset_file_path)
    hparams = {dl
            models.HP_DROPOUT: dropout,
            models.HP_BETA: beta,
            models.HP_LEARNING_RATE: lr,
            }
    model = models.get_model(dl_runhandler.model_name, dl_runhandler.keras_norm_params, dl_runhandler.mode, hparams, dl_runhandler.tile_size)
    weight_file = f'models/{model_name}/{dataset}/weights/dropout_{dropout}_beta_{beta}_learning_rate_{lr}/best_model.h5'
    model.load_weights(weight_file)
    return dl_runhandler.datasets, model
```
Best model for Conv3D with satellite bands and forest loss (overall best model)

```python
# Model conv3d comb
model_name = 'segmenting_future_double_reduced_conv3d'
dataset_file_path = 'data/datasets/main/2013_2019/64px_comb/'
dataset = '64px_comb'
beta = 0.995
lr = 0.001
dropout = 0.1

# Init model
datasets, model_dl = load_model(model_name, dataset, beta, lr, dropout, dataset_file_path)
```

Best model for Conv3D with satellite bands only
```python
# Model conv3d img only
model_name = 'segmenting_future_double_reduced_conv3d'
dataset_file_path = 'data/datasets/main/2013_2019/64px_comb/'
dataset = '64px_comb_img_only'
beta = 0.999
lr = 0.001
dropout = 0.1
# Init model
datasets, model_dl = load_model(model_name, dataset, beta, lr, dropout, dataset_file_path)
```

Best model for ConvLSTM with satellite bands and forest loss
```python
# Model convlstm comb
model_name = 'segmenting_future_double_reduced_convlstm2d'
dataset_file_path = 'data/datasets/main/2013_2019/64px_comb/' 
dataset = '64px_comb'
beta = 0.999
lr = 0.1
dropout = 0.2

# Init model
datasets, model_dl = load_model(model_name, dataset, beta, lr, dropout, dataset_file_path)
```

Best model for ConvLSTM with satellite bands only
```python
# Model convlstm img only
model_name = 'segmenting_future_double_reduced_convlstm2d'
dataset_file_path = 'data/datasets/main/2013_2019/64px_comb/'
dataset = '64px_comb_img_only'
beta = 0.999
lr = 0.001
dropout = 0.0

# Init model
datasets, model_dl = load_model(model_name, dataset, beta, lr, dropout, dataset_file_path)
```

Loading the baseline model and dataset can be done as follows:
```python
import model_baseline
model_name = 'baseline'
dataset = '64px'
beta = 0.9995
lr = 0.1
dataset_file_path = 'data/datasets/baseline/2013_2019_fixed/64px_baseline'
def load_baseline_model(model_name, dataset, beta, lr, dataset_file_path):
    baseline_runhandler = model_baseline.RunHandler(model_name = model_name, mode = dataset, tile_size = None, batch_size = 64, dataset_file_path = dataset_file_path)
    hparams = {
            model_baseline.HP_BETA: beta,
            model_baseline.HP_LEARNING_RATE: lr,
            }
    model = model_baseline.get_model(baseline_runhandler.model_name, baseline_runhandler.keras_norm_params, baseline_runhandler.mode, hparams, baseline_runhandler.tile_size)

    weight_file = f'models/{model_name}/{dataset}/weights/beta_{beta}_learning_rate_{lr}/best_model.h5'
    model.load_weights(weight_file)
    return baseline_runhandler, model
```

## Result Analysis
**Analyse Results.ipynb:** is used to compare predictions of the deep learning and baseline models.
