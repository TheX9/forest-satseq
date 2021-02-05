import math
import numpy as np
import tensorflow as tf

import ee
ee.Initialize()

# Google earth engine
DRIVE_FOLDER = 'Amazonas Deforestation fronts' # Google drive folder for exports

#DRIVE_DATA_FILE_PATH_ID = '' % Specifiy to be able to sample tile
RELATIVE_DATA_FILE_PATH = 'data/gee_datasets'
RELATIVE_STATS_FILE_PATH = 'data/stats'
FOREST_LOSS_STAST_FILE_PATH = f'{RELATIVE_STATS_FILE_PATH}/forest_loss_stats'

LOGGING_PATH = 'logs/'

MULT_TASK_ALLOWANCE = 2

# PATCH sizes
PATCH_SIZE_PX = 256
SCALE = 30
PATCH_SIZE_METERS = PATCH_SIZE_PX*SCALE

KERNEL_SIZE = 256

LANDSAT5_BAND_MAPPING = {
                        'B1':'BLUE',
                        'B2':'GREEN',	
                        'B3':'RED',	
                        'B4':'NIR',	
                        'B5':'SWIR1',
                        'B7':'SWIR2'
                        }


LANDSAT8_BAND_MAPPING = {
                        'B2':'BLUE',
                        'B3':'GREEN',	
                        'B4':'RED',	
                        'B5':'NIR',	
                        'B6':'SWIR1',
                        'B7':'SWIR2'
                        }

ELEV_BAND = 'elevation'
SLOPE_BAND = 'slope'

BANDS = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
PROPERTIES = ['point_id'] # ['time_slice',
FOREST_BANDS = ['lossyear','treecover2000']
FEATURES = BANDS + FOREST_BANDS + [ELEV_BAND] + PROPERTIES
YEARS = None # should be set externally

DATASET_TYPES = ['train','test','val']

TIME_INVARIANT = 'time_invariant'
TIME_SLICES = None # should be set externally
SCALE = 30


# Baseline bands
# Population
POPULATION_DENSITY_YEARS = [2000, 2005, 2010, 2015, 2020]
POP_IMAGE = ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density")
POP_IMAGE_BAND = 'population_density'
DIST_TO_CITY_CENTRE_BAND = 'dist_to_city_centre'
DIST_TO_ROADS_BAND = 'dist_to_roads'

def get_pop_years(years):
    # Get first and last year
    first_year = years[0]
    last_year = years[-1]

    # Population density and distance to city centres
    # Select first year
    select_years = np.array(POPULATION_DENSITY_YEARS)-first_year
    select_years[select_years>0] = -1000
    first_year_index = select_years.argmax()
    first_year = POPULATION_DENSITY_YEARS[first_year_index]

    # Select last years
    select_years = np.array(POPULATION_DENSITY_YEARS)-last_year
    select_years[select_years<0] = 1000
    last_year_index = select_years.argmin()
    last_year = POPULATION_DENSITY_YEARS[last_year_index]

    # Get the population density images
    return POPULATION_DENSITY_YEARS[POPULATION_DENSITY_YEARS.index(first_year):POPULATION_DENSITY_YEARS.index(last_year)+1]


def _get_closest_years(year, search_years):
    # Get closest years
    search_years = np.array(search_years)
    selection_years = abs(search_years - year)
    first_year_index = np.argmin(selection_years)
    selection_years[first_year_index] = 1000                 
    second_year_index = np.argmin(selection_years)
    return np.sort(np.array([search_years[first_year_index], search_years[second_year_index]]))

# Distance to forest
FOREST_IMAGE_BAND = 'lossyear'
DIST_TO_FOREST_BAND = 'dist_to_forest_loss'
FOREST_LOSS_LABEL = 'label_forest_loss'


# Specify the size and shape of patches expected by the model.
KERNEL_SIZE = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]

# GEE_FEATURE_LIMIT = 3e7
GEE_FEATURE_LIMIT = 1e7

SHAPEFILES_PATH = f'data/shapefiles'

## TFrecord parsing
def prepare_GEE_feature_dict(years):
    """Get feature dict for GEE TFrecords files.

    Args:
        years (range): Years to consider.

    Returns:
        dict: Feature dict with TF features.
    """
    # Functions to read input tf files from GEE
    # tf record structure of the files, that come from the Google Earth Engine export
    return  {**{
                # Parse yearly bands
                f'{band}_{year}': tf.io.FixedLenFeature(shape=[KERNEL_SIZE, KERNEL_SIZE],
                    dtype=tf.float32,
                    default_value=None)
                for band in BANDS for year in years
                },
                # Parse forest bands and elev band
                **{band: tf.io.FixedLenFeature(shape=[KERNEL_SIZE, KERNEL_SIZE],
                            dtype=tf.float32,
                            default_value=None)
                    for band in FOREST_BANDS + [ELEV_BAND]
                    },
                **{prop: tf.io.FixedLenFeature(shape=1, dtype=tf.string, default_value=None)
                for prop in PROPERTIES }
                }

## TFrecord parsing baseline
# Feature dict to parse the data coming from GEE
def prepare_baseline_GEE_feature_dict(years, mode = 'Parsing', features_per_part = None):
    """Get feature dict for GEE baseline TFrecords files.

    Args:
        years (range): Years to consider.
        features_per_part (int): If set, returns a partwise feature dictionary.
                                  This is needed since per patch several parts where needed to obtain features.
        mode (str): Allow to either get parsing or serializing features.
    Returns:
        dict: Feature dict with TF features.
    """
    _feature_dict_concatenated_bands = {}

    # Complex list
    complex_sampling = []

    # Properties - point id
    band_name = PROPERTIES[0]
    _feature_dict_concatenated_bands[band_name] = prepare_feature_serializing_or_parsing(mode, 'str', shape = 1)

    # Get pop years
    pop_years = _get_closest_years(years[-1], POPULATION_DENSITY_YEARS)
    first_pop_year = pop_years[0]

    # Pop density
    for pop_year in pop_years:
        # Population density
        band_name = f'{POP_IMAGE_BAND}_{pop_year}'
        _feature_dict_concatenated_bands[band_name] = prepare_feature_serializing_or_parsing(mode, 'float')
        complex_sampling.append(False)
        # Only get distance to city centre from first pop year
        if first_pop_year == pop_year:
            # Distance to city centre
            band_name = f'{DIST_TO_CITY_CENTRE_BAND}_{pop_year}'
            _feature_dict_concatenated_bands[band_name] = prepare_feature_serializing_or_parsing(mode, 'float')
            complex_sampling.append(True)

    # Forest loss image
    for band_name in FOREST_BANDS:
        _feature_dict_concatenated_bands[band_name] = prepare_feature_serializing_or_parsing(mode, 'float')
        complex_sampling.append(False)

    # Distance to forest loss
    _feature_dict_concatenated_bands[DIST_TO_FOREST_BAND] = prepare_feature_serializing_or_parsing(mode, 'float')
    complex_sampling.append(True)
    
    # Distance to roads
    _feature_dict_concatenated_bands[DIST_TO_ROADS_BAND] = prepare_feature_serializing_or_parsing(mode, 'float')
    complex_sampling.append(True)
    
    # Elevation
    band_name = ELEV_BAND
    _feature_dict_concatenated_bands[band_name] = prepare_feature_serializing_or_parsing(mode, 'float')
    complex_sampling.append(False)

    # Slope
    band_name = SLOPE_BAND
    _feature_dict_concatenated_bands[band_name] = prepare_feature_serializing_or_parsing(mode, 'float')
    complex_sampling.append(False)

    if features_per_part:
        # Ignore first feature point_id and add it manually later to every dict
        del _feature_dict_concatenated_bands[PROPERTIES[0]]

        _feature_dict_parts = []
        # Create several feature stacks for sampling
        i = 0
        ii = 0
        feature_dict_part = {}
        for band_name, feature in _feature_dict_concatenated_bands.items():
            if not complex_sampling[i]:
                # Add band and feature to dict
                feature_dict_part[band_name] = feature
                ii+=1
                # Create part dict and go to next dict
                if (ii)%features_per_part == 0 or (i+1)==len(_feature_dict_concatenated_bands):
                    # add point_id as feature
                    feature_dict_part[PROPERTIES[0]] = prepare_feature_serializing_or_parsing(mode, 'str', shape = 1)
                    _feature_dict_parts.append(feature_dict_part)
                    feature_dict_part = {}
                    ii = 0
            else:
                # add point_id as feature
                feature_dict_part_complex = {}
                feature_dict_part_complex[PROPERTIES[0]] = prepare_feature_serializing_or_parsing(mode, 'str', shape = 1)
                feature_dict_part_complex[band_name] = feature
                _feature_dict_parts.append(feature_dict_part_complex)
            i+=1
        return _feature_dict_parts
    else:
        return _feature_dict_concatenated_bands

FOREST_LOSS_PROPORTION_ALL = 'forest_loss_proportion_all'
FOREST_LOSS_PROPORTION_LAST_YEARS = 'forest_loss_proportion_last_years'
FOREST_LOSS_PROPORTION_DIRECT = 'forest_loss_direct_direct'
FOREST_LOSS_ON_THIS_PX_BEFORE = 'forest_loss_on_this_px_before'

def get_features_baseline_serializing_or_parsing(mode):
    feature_dict = {}

    # Properties 
    # Point id
    feature_dict[PROPERTIES[0]] = prepare_feature_serializing_or_parsing(mode, 'str', shape = 1)
    # Pixel id
    feature_dict['px_id'] = prepare_feature_serializing_or_parsing(mode, 'float', shape = 1)
    # Coords
    feature_dict['coords'] = prepare_feature_serializing_or_parsing(mode, 'float', shape = 2)

    # Distance to forest loss
    feature_dict[DIST_TO_FOREST_BAND] = prepare_feature_serializing_or_parsing(mode, 'float', shape = 1)

    # Distance to roads
    feature_dict[DIST_TO_ROADS_BAND] = prepare_feature_serializing_or_parsing(mode, 'float', shape = 1)

    # Population density
    band_name = f'{POP_IMAGE_BAND}'
    feature_dict[band_name] = prepare_feature_serializing_or_parsing(mode, 'float', shape = 1)

    # Distance to city centre
    band_name = f'{DIST_TO_CITY_CENTRE_BAND}'
    feature_dict[band_name] = prepare_feature_serializing_or_parsing(mode, 'float', shape = 1)

    # Forest loss in 256*256 patch - for all years
    feature_dict[FOREST_LOSS_PROPORTION_ALL] = prepare_feature_serializing_or_parsing(mode, 'float', shape = 1)

    # Forest loss in 256*256 patch - for the last years
    feature_dict[FOREST_LOSS_PROPORTION_LAST_YEARS] = prepare_feature_serializing_or_parsing(mode, 'float', shape = 1)

    # Forest loss in direct vicinity
    feature_dict[FOREST_LOSS_PROPORTION_DIRECT] = prepare_feature_serializing_or_parsing(mode, 'float', shape = 1)

    # Forest loss occured before
    feature_dict[FOREST_LOSS_ON_THIS_PX_BEFORE] = prepare_feature_serializing_or_parsing(mode, 'float', shape = 1)

    # Forest loss as label (last year)
    feature_dict[FOREST_LOSS_LABEL] = prepare_feature_serializing_or_parsing(mode, 'float', shape = 1)

    # Treecover
    feature_dict[FOREST_BANDS[1]] = prepare_feature_serializing_or_parsing(mode, 'float', shape = 1)

    # Elevation
    feature_dict[ELEV_BAND] = prepare_feature_serializing_or_parsing(mode, 'float', shape = 1)

    # Slope
    feature_dict[SLOPE_BAND] = prepare_feature_serializing_or_parsing(mode, 'float', shape = 1)

    return feature_dict

def prepare_feature_serializing_or_parsing(mode, feature_type, shape=[KERNEL_SIZE, KERNEL_SIZE]):
    # Dependent on mode return different variables
    if mode == 'Serializing':
        if feature_type == 'float':
            return 'float'
        elif feature_type == 'str':
            return 'bytes'
        elif feature_type == 'int64':
            return 'int64'
    elif mode == 'Parsing':
        if feature_type == 'float':
            return tf.io.FixedLenFeature(shape=shape, dtype=tf.float32, default_value=None)
        elif feature_type == 'str':
            return tf.io.FixedLenFeature(shape=shape, dtype=tf.string, default_value=None)
        elif feature_type == 'int64':
            return tf.io.FixedLenFeature(shape=shape, dtype=tf.int64, default_value=None)
  

def prepare_sequence_feature_serializing_or_parsing(mode, feature_type, shape=[KERNEL_SIZE, KERNEL_SIZE]):
    # Dependent on mode return different variables
    if mode == 'Serializing':
        if feature_type == 'float':
            return 'float'
        elif feature_type == 'str':
            return 'bytes'
        elif feature_type == 'int64':
            return 'int64'
    elif mode == 'Parsing':
        if feature_type == 'float':
            return tf.io.FixedLenSequenceFeature(shape=shape, dtype=tf.float32, default_value=None)
        elif feature_type == 'str':
            return tf.io.FixedLenSequenceFeature(shape=shape, dtype=tf.string, default_value=None)
        elif feature_type == 'int64':
            return tf.io.FixedLenSequenceFeature(shape=shape, dtype=tf.int64, default_value=None)

def get_various_dataset_features_serializing_or_parsing(mode, dataset):
    context_feature_dict = {}
    sequence_feature_dict = {}

    # Point id
    context_feature_dict[PROPERTIES[0]] = prepare_feature_serializing_or_parsing(mode, 'str', shape = 1)
    # Define adjusted kernel size
    if '64' in dataset:
        adj_kernel_size = 64
    elif '128' in dataset:
        adj_kernel_size = 128

    if (dataset =='256px_loss_only' 
    or dataset =='256px_loss_only_undersampling'
    or dataset =='256px_loss_only_oversampling'
    or dataset =='256px_cum_loss_only'
    or dataset == '256px_cum_seg_only'
    or dataset == '256px_seg_only'):
        # Properties 
        # Coords
        context_feature_dict['coords'] = prepare_feature_serializing_or_parsing(mode, 'float', shape = 2)
        # Loss label
        context_feature_dict[LABEL] = prepare_feature_serializing_or_parsing(mode, 'float', shape = [KERNEL_SIZE, KERNEL_SIZE])
        
        # loss per years
        sequence_feature_dict['forest_loss'] = prepare_sequence_feature_serializing_or_parsing(mode, 'float', shape = [KERNEL_SIZE, KERNEL_SIZE])
    elif dataset == '64px' or dataset == '128px' or dataset=='64px_comb' or dataset=='128px_comb' or dataset=='128px_comb_img_only' or dataset=='64px_comb_img_only' or dataset =='64px_comb_img_only_context':
        
        context_feature_dict = {}
        context_feature_dict['point_id'] = prepare_feature_serializing_or_parsing(mode, 'str', shape = 1)
        context_feature_dict['coords'] = prepare_feature_serializing_or_parsing(mode, 'float', shape = 2)
        context_feature_dict['elevation'] = prepare_feature_serializing_or_parsing(mode, 'float', shape = [adj_kernel_size, adj_kernel_size])
        context_feature_dict['treecover2000'] = prepare_feature_serializing_or_parsing(mode, 'float', shape = [adj_kernel_size, adj_kernel_size])
        context_feature_dict['forest_loss_label'] = prepare_feature_serializing_or_parsing(mode, 'float', shape = [adj_kernel_size, adj_kernel_size])

        sequence_feature_dict = {**{band:prepare_sequence_feature_serializing_or_parsing(mode, 'float', shape = [adj_kernel_size, adj_kernel_size])
                                    for band in BANDS},
                                 **{'forest_loss':prepare_sequence_feature_serializing_or_parsing(mode, 'float', shape = [adj_kernel_size, adj_kernel_size]),
                                    'year':prepare_sequence_feature_serializing_or_parsing(mode, 'int64', shape = 1)}
                                 }

        if 'comb' in dataset:
            sequence_feature_dict['forest_loss_cumu'] = prepare_sequence_feature_serializing_or_parsing(mode, 'float', shape = [adj_kernel_size, adj_kernel_size])

    # Concatenated dataset
    elif 'normal_concat_baseline' in dataset:
        # Context features
        context_feature_dict = {}
        context_feature_dict['point_id'] = prepare_feature_serializing_or_parsing(mode, 'str', shape = 1)
        context_feature_dict['coords'] = prepare_feature_serializing_or_parsing(mode, 'float', shape = 2)
        context_feature_dict['elevation'] = prepare_feature_serializing_or_parsing(mode, 'float', shape = [adj_kernel_size, adj_kernel_size])
        context_feature_dict['treecover2000'] = prepare_feature_serializing_or_parsing(mode, 'float', shape = [adj_kernel_size, adj_kernel_size])
        context_feature_dict['forest_loss_label'] = prepare_feature_serializing_or_parsing(mode, 'float', shape = [adj_kernel_size, adj_kernel_size])

        # Sequence features
        sequence_feature_dict = {**{band:prepare_sequence_feature_serializing_or_parsing(mode, 'float', shape = [adj_kernel_size, adj_kernel_size])
                                    for band in BANDS},
                                 **{'forest_loss':prepare_sequence_feature_serializing_or_parsing(mode, 'float', shape = [adj_kernel_size, adj_kernel_size]),
                                    'year':prepare_sequence_feature_serializing_or_parsing(mode, 'int64', shape = 1)}
                                 }
        sequence_feature_dict['forest_loss_cumu'] = prepare_sequence_feature_serializing_or_parsing(mode, 'float', shape = [adj_kernel_size, adj_kernel_size])

        # Context features from baseline
        # Distance to forest loss
        context_feature_dict[DIST_TO_FOREST_BAND] = prepare_feature_serializing_or_parsing(mode, 'float', shape = [adj_kernel_size, adj_kernel_size])

        # Distance to roads
        context_feature_dict[DIST_TO_ROADS_BAND] = prepare_feature_serializing_or_parsing(mode, 'float', shape = [adj_kernel_size, adj_kernel_size])

        # Population density
        band_name = f'{POP_IMAGE_BAND}'
        context_feature_dict[band_name] = prepare_feature_serializing_or_parsing(mode, 'float', shape = [adj_kernel_size, adj_kernel_size])

        # Distance to city centre
        band_name = f'{DIST_TO_CITY_CENTRE_BAND}'
        context_feature_dict[band_name] = prepare_feature_serializing_or_parsing(mode, 'float', shape = [adj_kernel_size, adj_kernel_size])

        # Slope
        context_feature_dict[SLOPE_BAND] = prepare_feature_serializing_or_parsing(mode, 'float', shape = [adj_kernel_size, adj_kernel_size])

    else:
        if mode =='Serializing':
            context_feature_dict = CONTEXT_FEATURES_SERIALIZING
            sequence_feature_dict = SEQUENCE_FEATURES_SERIALIZING
        else:
            context_feature_dict = CONTEXT_FEATURES_PARSING
            sequence_feature_dict = SEQUENCE_FEATURES_PARSING
    return context_feature_dict, sequence_feature_dict


BASELINE_NORMALIZATION = {}
BASELINE_NORMALIZATION[DIST_TO_FOREST_BAND] = ('min_max', 0,6210.428)
BASELINE_NORMALIZATION[DIST_TO_ROADS_BAND] = ('min_max', 0,15651.276)
BASELINE_NORMALIZATION[POP_IMAGE_BAND] = ('min_max',0,12.448312)
BASELINE_NORMALIZATION[DIST_TO_CITY_CENTRE_BAND] = ('min_max', 0, 72197.8)
BASELINE_NORMALIZATION[FOREST_BANDS[1]] = ('min_max', 0,100)
BASELINE_NORMALIZATION[ELEV_BAND] = ('min_max', 0,657.0)
BASELINE_NORMALIZATION[SLOPE_BAND] = ('min_max', 0,51.964115)
BASELINE_NORMALIZATION['coord_lat'] =('min_max', 0,)
BASELINE_NORMALIZATION['coord_lon'] = ('min_max', 0,)

NORMALIZATION = {}
NORMALIZATION['BLUE'] = ('min_max', -0.1042, 1.0634)
NORMALIZATION['GREEN'] = ('min_max', -0.00225, 1.1804)
NORMALIZATION['RED'] = ('min_max', -0.00045, 1.3717)
NORMALIZATION['NIR'] = ('min_max', -0.0156, 1.5863)
NORMALIZATION['SWIR1'] = ('min_max', -0.0101, 1.6)
NORMALIZATION['SWIR2'] = ('min_max', -0.0052, 1.6)
NORMALIZATION['forest_loss'] = ('min_max', 0, 1)
NORMALIZATION['forest_loss_cumu'] = ('min_max', 0, 1)
NORMALIZATION[ELEV_BAND] = ('min_max', 0,657.0)
NORMALIZATION[FOREST_BANDS[1]] = ('min_max', 0,100)
NORMALIZATION[DIST_TO_FOREST_BAND] = ('min_max', 0,6210.428)
NORMALIZATION[DIST_TO_ROADS_BAND] = ('min_max', 0,15651.276)
NORMALIZATION[POP_IMAGE_BAND] = ('min_max',0,12.448312)
NORMALIZATION[DIST_TO_CITY_CENTRE_BAND] = ('min_max', 0, 72197.8)
NORMALIZATION[FOREST_BANDS[1]] = ('min_max', 0,100)
NORMALIZATION[ELEV_BAND] = ('min_max', 0,657.0)
NORMALIZATION[SLOPE_BAND] = ('min_max', 0,51.964115)
NORMALIZATION['coord_lat'] =('min_max', 0,)
NORMALIZATION['coord_lon'] = ('min_max', 0,)

## TFrecords parsing non-baseline
# Functions to serialize/parse the final dataset
CONTEXT_FEATURES = ['point_id', 'elevation', 'treecover2000']

# Serializing - used to serialize the features
CONTEXT_FEATURES_SERIALIZING = {
    'point_id' : 'bytes',
    'coords' : 'float',
    'elevation' : 'float',
    'treecover2000': 'float',
    'forest_loss_label': 'float'
}

# Parsing - used to parse the features
CONTEXT_FEATURES_PARSING = {
    'point_id': tf.io.FixedLenFeature([1], tf.string),
    'coords': tf.io.FixedLenFeature([2], tf.float32),
    'elevation': tf.io.FixedLenFeature([KERNEL_SIZE,KERNEL_SIZE], tf.float32),
    'treecover2000': tf.io.FixedLenFeature([KERNEL_SIZE,KERNEL_SIZE], tf.float32),
    'forest_loss_label': tf.io.FixedLenFeature([KERNEL_SIZE,KERNEL_SIZE], tf.float32)
}

LABEL = 'forest_loss_label'
BASELINE_LABEL = FOREST_LOSS_LABEL

# Serializing - used to serialize the features
SEQUENCE_FEATURES_SERIALIZING = {**{band:'float' for band in BANDS},
                                 **{'forest_loss':'float',
                                    'year':'int64'}
                                }

# Parsing - used to parse the features
SEQUENCE_FEATURES_PARSING = {**{band:tf.io.FixedLenSequenceFeature([KERNEL_SIZE,KERNEL_SIZE], tf.float32) for band in BANDS},
                             **{'forest_loss':tf.io.FixedLenSequenceFeature([KERNEL_SIZE,KERNEL_SIZE], tf.float32),
                               'year':tf.io.FixedLenSequenceFeature([1], tf.int64)}
                            }


# Training related
BATCH_SIZE = 32
SHUFFLE_SIZE_FILES = 1000
SHUFFLE_SIZE = 1000


# Shard size
SHARD_SIZE_TF_RECORD = 25 # select such that ~100mb

SHARD_SIZE_TF_RECORD_BASELINE = 60 #25*8 # select such ~100mb

# Maximum number of files in dir
MAX_FILES_IN_DIR = 200

# Samples
TRAIN_SIZE = 30000
TEST_SIZE = 10000
VAL_SIZE = 10000
TOTAL_SAMPLE_SIZE = TRAIN_SIZE+TEST_SIZE+VAL_SIZE
TRAIN_TEST_VAL_SPLIT = np.array([TRAIN_SIZE/TOTAL_SAMPLE_SIZE, TEST_SIZE/TOTAL_SAMPLE_SIZE, VAL_SIZE/TOTAL_SAMPLE_SIZE])

###### For random point buffer sampling
# Sample split, defines how many PATCH buffer points will be sampled in one sampling round for train/val/test 
# It will be sampled until the total sample size is reached
SAMPLING_SPLIT = [3,1,1]

#### Datasets
# Prepare google earth engine assets
#deforest_hotspots = ee.FeatureCollection('users/thex/rainforest_deforestation_hotspots')
DEFOREST_HOTSPOTS = ee.FeatureCollection('users/thex/deforestation_amazon_hotspots')
DEFOREST_HOTSPOTS_BUFFERED = ee.FeatureCollection(DEFOREST_HOTSPOTS.map(lambda feature: feature.buffer(PATCH_SIZE_METERS)))

AMAZON_ECOREGION = ee.FeatureCollection('users/thex/amazon_ecoregion')
AMAZON_ECOREGION_BUFFERED = ee.FeatureCollection(AMAZON_ECOREGION.map(lambda feature: feature.buffer(PATCH_SIZE_METERS)))

ROADS_AMAZON = ee.FeatureCollection('users/thex/roads_amazon')

# Cloud masking function.
def maskL8sr(image):
    cloudShadowBitMask = ee.Number(2).pow(3).int()
    cloudsBitMask = ee.Number(2).pow(5).int()
    qa = image.select('pixel_qa')
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
      qa.bitwiseAnd(cloudsBitMask).eq(0))
    return image.updateMask(mask)

# Use Landsat 8 surface reflectance data.
def get_landsat8_image(year, month = 7, cloud_cover = 5):
    l8sr= ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
    image = (l8sr
            .filter(ee.Filter.calendarRange(year, year, 'year'))
            .filter(ee.Filter.calendarRange(month, month, 'month')) # sample from Juli only
            .filterMetadata("CLOUD_COVER","less_than", cloud_cover) # filter cloudy images
            .map(maskL8sr) # Mask clouds
            .median() # Get median composite
            .select(list(LANDSAT8_BAND_MAPPING)) # Select bands
            .rename(list(LANDSAT8_BAND_MAPPING.values())    ) # Rename bands
            .multiply(0.0001) # Scale bands
            )
    return image


def cloudMaskL457(image):
    """Function to mask clouds based on the pixel_qa band of Landsat SR data.
    
    Source:  https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C01_T1_SR
    adapated to python

    Args:
        image (ee.Image): image Input Landsat SR image

    Returns:
        [ee.Image]: Cloudmasked Landsat image
    """

    qa = image.select('pixel_qa')
    # If the cloud bit (5) is set and the cloud confidence (7) is high
    # or the cloud shadow bit is set (3), then it's a bad pixel.
    cloud = (qa.bitwiseAnd(1 << 5)
                    .And(qa.bitwiseAnd(1 << 7))
                    .Or(qa.bitwiseAnd(1 << 3))
            )
    # Remove edge pixels that don't occur in all bands
    mask2 = image.mask().reduce(ee.Reducer.min())
    return image.updateMask(cloud.Not()).updateMask(mask2)

def get_landsat5_image(year, month = 7, cloud_cover = 5):
    l5sr = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')
    image = (l5sr
            .filter(ee.Filter.calendarRange(year, year, 'year'))
            .filter(ee.Filter.calendarRange(month, month, 'month')) # sample from Juli only
            .filterMetadata("CLOUD_COVER","less_than", cloud_cover) # filter cloudy images
            .map(cloudMaskL457) # Mask clouds
            .median() # Get median composite
            .select(list(LANDSAT5_BAND_MAPPING)) # Select bands
            .rename(list(LANDSAT5_BAND_MAPPING.values())) # Rename bands
            .multiply(0.0001) # Scale bands
            )
    return image

# Load forest loss and elevation image
FOREST_LOSS_IMAGE = ee.Image("UMD/hansen/global_forest_change_2019_v1_7").unmask(0)
FOREST_LOSS_IMAGE_MASKED = ee.Image("UMD/hansen/global_forest_change_2019_v1_7")

DIGITAL_ELEVATION_IMAGE = ee.Image("USGS/SRTMGL1_003")

