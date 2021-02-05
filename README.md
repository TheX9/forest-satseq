# Future Forest Loss Prediction

This repository contains code used for my master thesis "Predicting Future Deforestation Using Satellite Images".
![alt text](https://github.com/thex/forest-satseq/blob/master/top_pred.png?raw=true)
# Files
## Configuration
config.py: is used to specify general settings.
logging_config.py: logging configuration.
## Data Sampling
Sample Year Statistics.ipynb : is used to determine the amount of satellite imagery for different months.
 <br/>create_non_overlapping_samples.py: creates non-overlapping train, validation, test splits for an input of patches.
 <br/>Stratify samples.ipynb: is used to stratify the train, validation, test splits to have the same average forest loss proportion per patch.
 <br/>sample_tiles.py : samples statistics, baseline features and satellite images from patches using Google Earth Engine and saves them to a Google Drive folder specified in config.py.
 <br/>download_dataset.py: is used to download the from Google Earth Engine sampled patches from Google Drive.
 <br/>process_statistic_sample.py: processes sampled statistic patches from Google Earth Engine samples to get forest loss statistics for each patch.
 <br/>process_sampled_tiles.py: processes the sampled patches from Google Earth Engine to the final dataset.

## Data Visualization
GE - Visualize Features.ipynb: is used to visualize the different features.

## Dataset
Raw data is removed from the repository due to size limitations.
Links to download the 64px, 256px and baseline dataset will follow.
/data/shapefiles: contains the shapefiles used for sampling.
 <br/>/data/stats: contains statistics about satellite image availability and forest loss per point.

 <br/>load_dataset.py: is used to load the datasets.

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
callbacks_config.py: is used to specify the Tensorflow training callbacks.
models.py: handles the hyperparameter optimization of the deep learning models.
model_baseline: handles the hyperparameter optimization of the baseline model.

Model weights of the best performing models will follow.

## Result Analysis
Analyse Results.ipynb: is used to compare predictions of the deep learning and baseline models.
