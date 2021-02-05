# Imports


import config

import time
import datetime 
import argparse
import rtree
import os
from os import path

import geopandas as gpd
import pandas as pd
import numpy as np
import tqdm

def _create_fid_col(gdf):
    """Helper function to change the feature id (FID) column to the index.

    Args:
        gdf (GeoDataFrame): GeoDataFrame.
    """
    if 'FID' not in gdf.columns:
        gdf.index.rename('FID', inplace=True)
        gdf.reset_index(inplace=True)

def create_gdf_that_does_not_intersect(gdf_gen, gdf_all, max_n, index = None):
    """ Create maximal max_n buffer points from dataset gdf_all that do not overlapp with buffer points from gdf_gen.

    Args:
        gdf_gen (GeoDataFrame): GeoDataFrame that holds all so far sampled buffer points.
        gdf_all (GeoDataFrame): GeoDataFrame that holds all available buffer points.
        max_n (int): Maximum amount of buffer points that should be sampled.
        index: rtree index of gdf_gen.

    Returns:
        GeoDataFrame, GeoDataFrame: Holds up to max_n sampled buffer points, that do not overlap with points in gdf_gen, Holds all points that do overlapp with gdf_gen.
    """
    # List to collect potential new points and failed points
    new_rows = []
    failed_points = []

    # Instantiate index class
    #idx = rtree.index.Index()
    #for pos, row in gdf_gen.iterrows():
    #    idx.insert(pos, row['geometry'].bounds)
   
    for _, row in gdf_all.iterrows():
        # Test for potential intersection with each feature of the other feature collection  
        intersect = np.array([row['geometry'].intersects(gdf_gen.loc[intersect_maybe,'geometry'])
                         for intersect_maybe in index.intersection(row['geometry'].bounds)]).any()

        # Append only if no intersection was found
        if not(intersect):
            new_rows.append(row)
            if len(new_rows) == max_n:
                break
        else:
          failed_points.append(row)
    
    return gpd.GeoDataFrame(new_rows), gpd.GeoDataFrame(failed_points)

def create_non_overlapping_gdfs(gdf_sample, total_sample_size = None, init_gdfs = {}):
    # Initialize gdf_all
    gdf_all = gdf_sample

    # Create empty dataframes
    gdf_train = gpd.GeoDataFrame()
    gdf_val = gpd.GeoDataFrame()
    gdf_test = gpd.GeoDataFrame()

    # Initialize empty indexes for the different datasets
    test_index = rtree.index.Index()
    train_index = rtree.index.Index()
    val_index = rtree.index.Index()

    # Init dataframes
    if len(init_gdfs):
        for dataset_type, init_gdf in init_gdfs.items():
            # Flag as init dataframe
            init_gdf['init'] = True
            init_gdf['type'] = dataset_type
            if dataset_type == 'train':
                gdf_train = init_gdf
            elif dataset_type == 'val':
                gdf_val = init_gdf
            elif dataset_type == 'test':
                gdf_test = init_gdf
        
        # Concat all
        gdf_all = gpd.GeoDataFrame(pd.concat([gdf_sample, gdf_train, gdf_val, gdf_test], ignore_index=True))
        gdf_train = gdf_all[gdf_all['type'] == 'train']
        gdf_val = gdf_all[gdf_all['type'] == 'val']
        gdf_test = gdf_all[gdf_all['type'] == 'test']

        # Remove datasets from all dataset
        gdf_all = gdf_all.loc[gdf_all.index.difference(gdf_train.index)]
        gdf_all = gdf_all.loc[gdf_all.index.difference(gdf_val.index)]
        gdf_all = gdf_all.loc[gdf_all.index.difference(gdf_test.index)]

        # Init rtrees
        for pos, row in gdf_train.iterrows():
            val_index.insert(pos, row['geometry'].bounds)
            test_index.insert(pos, row['geometry'].bounds)

        for pos, row in gdf_val.iterrows():
            train_index.insert(pos, row['geometry'].bounds)
            test_index.insert(pos, row['geometry'].bounds)
        
        for pos, row in gdf_test.iterrows():
            train_index.insert(pos, row['geometry'].bounds)
            val_index.insert(pos, row['geometry'].bounds)
        
    # Keep failed points in dataframe to not check them again
    gdf_train_failed = gpd.GeoDataFrame()
    gdf_val_failed = gpd.GeoDataFrame()
    gdf_test_failed = gpd.GeoDataFrame()

    # if total sample size is none, sample until gdf_all is empty
    if not total_sample_size:
        total_sample_size = len(gdf_all)

    # Amount of times that need to be sampled at least
    sample_times = int((total_sample_size)/np.sum(config.SAMPLING_SPLIT))

    # Loop for sample times
    for _ in tqdm.tqdm(range(sample_times)):

        #Test
        # Get so far used tiles + init files
        gdf_gen = gpd.GeoDataFrame(pd.concat([gdf_train, gdf_val], ignore_index=False))
        # Remove already tested points from tile set
        gdf_all_test = gdf_all.loc[gdf_all.index.difference(gdf_test_failed.index)]
        # Abort sampling if no more points are available to choose from
        if len(gdf_all_test) == 0:
            break
        # Generate new test tiles
        gdf_test_new,gdf_test_failed_new = create_gdf_that_does_not_intersect(gdf_gen, gdf_all_test, config.SAMPLING_SPLIT[2], index = test_index)
        # Add new test tiles to indexes
        for pos, row in gdf_test_new.iterrows():
            train_index.insert(pos, row['geometry'].bounds)
            val_index.insert(pos, row['geometry'].bounds)
        # Concatenate new and old
        gdf_test = gpd.GeoDataFrame(pd.concat([gdf_test, gdf_test_new]))
        gdf_test_failed = gpd.GeoDataFrame(pd.concat([gdf_test_failed, gdf_test_failed_new]))
        # Remove test dataset from all dataset
        gdf_all = gdf_all.loc[gdf_all.index.difference(gdf_test.index)]
        
        # Val
        # Get so far used tiles + init files
        gdf_gen = gpd.GeoDataFrame(pd.concat([gdf_train,gdf_test], ignore_index=False))
        # Remove already tested points from tile set
        gdf_all_val = gdf_all.loc[gdf_all.index.difference(gdf_val_failed.index)]
        # Abort sampling if no more points are available to choose from
        if len(gdf_all_val) == 0:
            break
        # Generate new test tiles
        gdf_val_new, gdf_val_failed_new = create_gdf_that_does_not_intersect(gdf_gen, gdf_all_val, config.SAMPLING_SPLIT[1], index = val_index)
        # Add new val tiles to indexes
        for pos, row in gdf_val_new.iterrows():
            train_index.insert(pos, row['geometry'].bounds)
            test_index.insert(pos, row['geometry'].bounds)
        # Concatenate new and old
        gdf_val = gpd.GeoDataFrame(pd.concat([gdf_val, gdf_val_new]))
        gdf_val_failed = gpd.GeoDataFrame(pd.concat([gdf_val_failed, gdf_val_failed_new]))
        # Remove test dataset from all dataset
        gdf_all = gdf_all.loc[gdf_all.index.difference(gdf_val.index)] 
        
        # Train
        # Get so far used tiles + init files
        gdf_gen = gpd.GeoDataFrame(pd.concat([gdf_test,gdf_val], ignore_index=False))
        # Remove already tested points from tile set
        gdf_all_train = gdf_all.loc[gdf_all.index.difference(gdf_train_failed.index)]
        # Abort sampling if no more points are available to choose from
        if len(gdf_all_train) == 0:
            break
        # Generate new test tiles
        gdf_train_new, gdf_train_failed_new = create_gdf_that_does_not_intersect(gdf_gen, gdf_all_train, config.SAMPLING_SPLIT[0], index = train_index)
        # Add new train tiles to indexes
        for pos, row in gdf_train_new.iterrows():
            val_index.insert(pos, row['geometry'].bounds)
            test_index.insert(pos, row['geometry'].bounds)
        # Concatenate new and old
        gdf_train = gpd.GeoDataFrame(pd.concat([gdf_train, gdf_train_new]))
        gdf_train_failed = gpd.GeoDataFrame(pd.concat([gdf_train_failed, gdf_train_failed_new]))
        # Remove test dataset from all dataset
        gdf_all = gdf_all.loc[gdf_all.index.difference(gdf_train.index)] 


    # Remove initial gdfs from result
    if init_gdfs:
        gdf_test = gdf_test[gdf_test['init'] != True]
        gdf_val = gdf_val[gdf_val['init'] != True]
        gdf_train = gdf_train[gdf_train['init'] != True]

    # Set crs type
    gdf_test.crs = gdf_sample.crs
    gdf_val.crs = gdf_sample.crs
    gdf_train.crs = gdf_sample.crs

    # Create id column
    _create_fid_col(gdf_test)
    _create_fid_col(gdf_val)
    _create_fid_col(gdf_train)

    # Maintain train_val_test_split
    sample_sizes = np.min(np.array([len(gdf_train), len(gdf_val), len(gdf_test)])/np.array(config.SAMPLING_SPLIT))*np.array(config.SAMPLING_SPLIT)

    # Adjust sampling sizes
    gdf_train = gdf_train.sample(int(sample_sizes[0]))
    gdf_val = gdf_val.sample(int(sample_sizes[1]))
    gdf_test = gdf_test.sample(int(sample_sizes[2]))

    # Return the points
    return gdf_test, gdf_val, gdf_train


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create non overlapping patches.')

    # Shapefiles
    parser.add_argument('--shapefiles', nargs='+')

    # Random points
    parser.add_argument('--n_rand_points', type = int, default = 100)

    # Output file name
    parser.add_argument('--output_file_path', type=str)

    # Init shapefile
    parser.add_argument('--init_shapefile', default='')

    # Parse args
    args = parser.parse_args()

    import random
    # Check if shapefile is given for sampling
    random.seed(0)
    random_states = random.sample(range(1, 1000000), 20)
    best_random_state = 0
    best_amount = 0

    if args.init_shapefile:
        init_gdfs = {}
        for dataset_type in config.DATASET_TYPES:
            # Load shapefiles
            init_gdfs[dataset_type] = gpd.read_file(args.init_shapefile%(dataset_type))

    if args.shapefiles:
        gdfs_shapefile = []
        for key_value in args.shapefiles:
            shapefile_name, shapefile = key_value.split('=', 1)

            gdf = gpd.read_file(shapefile)
            gdf['FID']  +=f'_{shapefile_name}'
            gdfs_shapefile.append(gdf)

        # Ignore the index, since it is not meaningful
        gdf_all = pd.concat(gdfs_shapefile, ignore_index = True)

        gdf_test, gdf_val, gdf_train = create_non_overlapping_gdfs(gdf_all, init_gdfs = init_gdfs)
        file_path = os.path.normpath(args.output_file_path)

    file_dir, file_name = os.path.split(file_path)
    file_name = f'%s_{file_name.replace(".shp","")}.shp'
    file_path = f'{file_dir}/{file_name}'

    # Export sampled sets
    gdf_test.to_file(file_path%('test'))
    gdf_val.to_file(file_path%('val'))
    gdf_train.to_file(file_path%('train'))

    # Patch sampling shapefile
    gdf_sampling = gpd.GeoDataFrame(pd.concat([gdf_test, gdf_val, gdf_train], ignore_index=False))
    gdf_sampling.crs = gdf_test.crs
    gdf_sampling.to_file(file_path%('sampling'))