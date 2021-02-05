# Imports
import argparse
import os

from glob import glob

import pandas as pd
import geopandas as gpd
import tensorflow as tf
import numpy as np

from tqdm import tqdm

import config 

feature_dict_statistic_sample = {
 'lossyear': tf.io.FixedLenFeature(shape=[config.KERNEL_SIZE, config.KERNEL_SIZE], dtype=tf.float32, default_value=None),
 'point_id': tf.io.FixedLenFeature(shape=1, dtype=tf.string, default_value=None)
}

def parse_loss_years_dataset(fileNames):
  # Read `TFRecordDatasets` 
  dataset = tf.data.TFRecordDataset(fileNames, compression_type='GZIP')
    
  # Make a parsing functionmai
  def parse(example_proto):
    parsed_features = tf.io.parse_single_example(example_proto, feature_dict_statistic_sample)    
    return parsed_features
  
  # Map the function over the dataset
  dataset = dataset.map(parse, num_parallel_calls=5)
  
  return dataset


if os.name == 'nt':
    import win32api
    import pywintypes
    
    def transform_path(path):
        dir_name = os.path.dirname(path)
        base_name = os.path.basename(path)
        try:
            dir_name = win32api.GetShortPathName(dir_name)
        except pywintypes.error:
            dir_name = dir_name
        path = f'{dir_name}/{base_name}'
        return path
else:
    def transform_path(path):
        return path


def main(file_dir, year_range, random_buffer_shapefile):        
    file_dir = os.path.normpath(file_dir)
    file_name = file_dir.split(os.sep)[-1]
    
    # Init examples
    tf_record_files = glob(f'{file_dir}/*/*.tfrecord.gz')
    tf_record_files = [transform_path(file) for file in tf_record_files]

    np.random.shuffle(tf_record_files)
    stats_sample_dataset = parse_loss_years_dataset(tf_record_files)

    data_points_stats = []

    # Loop over all data points
    for data_point in tqdm(stats_sample_dataset):
        data_point_stat = {}
        data_point_stat['point_id'] = data_point['point_id'].numpy()[0].decode('UTF-8')
    
        lossyear = data_point['lossyear']
        # get losses for years
        for year in year_range:
            # Get binary loss mask for year, year is indicated by two digit number (e.g. 19 == 2019)
            binary_forest_loss_mask =  (lossyear == int(str(year)[-2:]))
            # Assign to dictionary
            data_point_stat[f'loss_sum_{year}'] = binary_forest_loss_mask.numpy().sum()
            
        data_points_stats.append(data_point_stat)
    
    # Create point stats dataframe
    df_point_stats = pd.DataFrame(data_points_stats)

    # Output file path
    output_file_dir = f'{config.FOREST_LOSS_STAST_FILE_PATH}/{file_name}'
    # Check if file dir exists
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)

    # Save stats dataframe
    df_point_stats.to_csv(f'{output_file_dir}/df_point_stats_{file_name}.csv')

    # Load random points shapefile from which the stats were sampled
    gdf_rand_points = gpd.read_file(random_buffer_shapefile)

    # Get coordinates and additional information about the sampled points
    gdf_sampled_points = gdf_rand_points.merge(df_point_stats, left_on=['FID'], right_on=['point_id'])

    # Save shapefile
    gdf_sampled_points.to_file(f'{output_file_dir}/sampled_points_{file_name}.shp')

if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser(description='Process statistic sample.')

    # Input directory
    parser.add_argument('--file_dir', default=None)

    # Sample years
    parser.add_argument('--year_range',  nargs="+", default = [2013, 2020], type=int)

    # Shapefiles path
    parser.add_argument('--shapefile', default=None)

    # Evaluate argparse object
    args = parser.parse_args()

    # Call main 
    main(args.file_dir,
         range(*args.year_range),
         args.shapefile,
         )