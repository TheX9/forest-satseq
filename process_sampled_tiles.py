# Imports
import argparse
import time
import threading
import math
import multiprocessing
import gc
import os
import re
from glob import glob
import datetime

import pandas as pd
import geopandas as gpd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import config
import logging_config

import skimage.transform

import pathos.multiprocessing

class TFRecordHelper:
    def __init__(self,
                 years,
                 baseline=False,
                 context_features_serializing=None,
                 sequence_features_serializing=None,
                 mode = None):
        self.years = years
        self.baseline = baseline

        if baseline:
            # GEE parsing bands
            self._initial_parsing_feature_dict = config.prepare_baseline_GEE_feature_dict(years)
            # Get serializing
            self.context_features_serializing = config.get_features_baseline_serializing_or_parsing('Serializing')
        else:
            # GEE parsing bands
            self._initial_parsing_feature_dict = config.prepare_GEE_feature_dict(years[:-1])

            if mode:
                con, seq = config.get_various_dataset_features_serializing_or_parsing('Serializing', mode)
                self.context_features_serializing = con
                self.sequence_features_serializing = seq
            else:
                # Set context and sequence serializing feature dicts
                self.context_features_serializing = config.CONTEXT_FEATURES_SERIALIZING
                self.sequence_features_serializing = config.SEQUENCE_FEATURES_SERIALIZING

        if context_features_serializing:
            self.context_features_serializing = context_features_serializing

        if sequence_features_serializing:
            self.sequence_features_serializing = sequence_features_serializing

    # Convert values into features according to given type
    @classmethod
    def _create_feature(self, value, feature_type):
        """Create a feature according to the given feature type.

        Args:
            value (misc): Value of the feature.
            feature_type (str): Type of the feature. 

        Returns:
            type_list: List according to given feature type.
        """
        # TF record helper functions
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        def _float_feature(value):
            """Returns a float_list from a float / double."""
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value)) 

        if type(value) != list:
            value = [value]
        if feature_type == 'bytes':
            return _bytes_feature(value)
        if feature_type == 'float':
            return _float_feature(value)
        if feature_type == 'int64':
            return _int64_feature(value)   

    @classmethod
    def _tf_to_list(self, tensor):
        """Transform tensor to list.
        """
        return list(tensor.numpy().flatten())


    def _context_features(self, context_feature_values):
        """Create context features.

        Args:
            context_feature_values (dict): Holds all context features.

        Returns:
            tf.train.Features: Context features in tf features type.
        """
        # Create a feature for each entry in the dict, according to the type dict.
        context_dict = {key:TFRecordHelper._create_feature(value, self.context_features_serializing[key])
                        for key,value in context_feature_values.items()}
        return tf.train.Features(feature = context_dict)

    def _sequence_feature_lists(self, sequence_feature_values):
        """Create sequence feature lists.

        Args:
            sequence_feature_values (dict): Holds all sequence features.

        Returns:
            tf.train.FeatureLists: Sequence features in tf.train.FeatureLists type.
        """
        sequence_feature_list = {}
        # Loop over all sequence features.
        for feature, values in sequence_feature_values.items():
            features = []
            # Append each sequence element of the feature.
            for value in values:
                features.append(TFRecordHelper._create_feature(value, self.sequence_features_serializing[feature]))    
            sequence_feature_list[feature] = tf.train.FeatureList(feature=features)
        # Create feature lists
        return tf.train.FeatureLists(feature_list=sequence_feature_list) 
        
    def _sequence_example(self, context_feature_values, sequence_feature_values):
        """Creates a sequence example from given context and sequence features.

        Args:
            context_feature_values (dict): Holds all context features.
            sequence_feature_values (dict): Holds all sequence features.

        Returns:
            tf.train.SequenceExample: A tf.train.SequenceExample object, that holds context and sequence features.
        """
        # Generate context features
        context_features = self._context_features(context_feature_values)

        # Create sequence feature lists
        sequence_feature_lists = self._sequence_feature_lists(sequence_feature_values)
    
        # Create sequence example
        sequence_example = tf.train.SequenceExample(
                            context=context_features,
                            feature_lists=sequence_feature_lists)
        
        return sequence_example

    def serialize_example(self, context_feature_values):
        """Create a serialized example.

        Args:
            context_feature_values (dict): Holds all context features.

        Returns:
            tf.train.Example: A serialized tf.train.Example object.
        """
        return self._example(context_feature_values).SerializeToString()

    def _example(self, context_feature_values):
        """Creates a sequence example from given context and sequence features.

        Args:
            context_feature_values (dict): Holds all context features.

        Returns:
            tf.train.Example: A tf.train.Example object, that holds features.
        """
        # Generate context features
        context_features = self._context_features(context_feature_values)

        # Create example
        example = tf.train.Example(features=context_features,)
        
        return example

    def serialize_sequence_example(self, context_feature_values, sequence_feature_values):
        """Create a serialized sequence example.

        Args:
            context_feature_values (dict): Holds all context features.
            sequence_feature_values (dict): Holds all sequence features.

        Returns:
            tf.train.SequenceExample: A serialized tf.train.SequenceExample object.
        """
        return self._sequence_example(context_feature_values,sequence_feature_values).SerializeToString()

    # write sample
    def write_serialized_samples(self, serialized_samples, file_path):
        """Write all serialized examples in serialized samples to drive.

        Args:
            serialized_samples (list): Contains the samples, that should be serialized.
            file_path (str): File path, of the output file.
        """
        tf_record_options = tf.io.TFRecordOptions(compression_type='GZIP')
        with tf.io.TFRecordWriter(file_path, options=tf_record_options) as writer:
            for serialized_example in serialized_samples:
                writer.write(serialized_example)

    # Extract features from serialized_example 
    def parse_sequence_example(self, serialized_example):
        """Parse a serialized sequence sample.

        Args:
            serialized_example (tf.train.SequenceExample): Serialized sequence example.

        Returns:
            (dict,dict): Tuple of two dicts, with context and sequence features.
        """
        return tf.io.parse_single_sequence_example(serialized=serialized_example,
                                                context_features=config.CONTEXT_FEATURES_PARSING,
                                                    sequence_features=config.SEQUENCE_FEATURES_PARSING)
    
    # Parsing function to parse the concatedated dataset coming from GEE
    def parse_concatenated_dataset(self, fileNames, feature_dict = None):

        if not feature_dict:
            feature_dict = self._initial_parsing_feature_dict

        # Read `TFRecordDatasets` 
        dataset = tf.data.TFRecordDataset(fileNames, compression_type='GZIP')
            
        # Make a parsing function
        def _parse(example_proto):
            parsed_features = tf.io.parse_single_example(example_proto, feature_dict)    
            return parsed_features
        
        # Map the function over the dataset
        dataset = dataset.map(_parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Prefetch data
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

class ProcessSampledTilesHelper:
    """ Class, that process raw sampled tiles from GEE, to a format that is feedable to tensorflow.
    """
    def __init__(self,
                input_dir, 
                shapefiles_path, 
                years, 
                output_dir, 
                baseline, 
                preprocess_part_data, 
                preprocess_only, 
                preprocess_output_dir, 
                processors,
                mode = None,
                concat_normal_baseline = False,
                baseline_file_dir = None):
        """Initialize the class and set attributes.

        Args:
            input_dir (str): Input directory of the files, that are about to be processed.
            shapefiles_path (str): Path where the different dataset shapefiles can be found.
                                   Placeholder for dataset type included.
            years (range/list): Years that are contained in the input dataset.
            output_dir (str): Output directory of the processed files.
            baseline (bool): Bool, if baseline samples are processed. 
            preprocess_part_data (bool): Bool, if preprocessing is needed.
            preprocess_only (bool): Bool, if only preprocessing.
            preprocess_output_dir (str): Output dir for preprocessed data.
            processors (int): Specifies the amount of processors that should be used.
            mode (str): Determines different sampling modes that are available for parsing the main data.
            concat_normal_baseline (bool): Should a dataset created that contains baseline features as context features.
            baseline_file_dir(str): File dir of baseline features.
        """
        # File directory of the files that are going to be processed
        self.file_dir = input_dir

        # Base path for the shapefiles
        self.shapefiles_path = shapefiles_path

        # Years
        self.years = years
            
        # Mode
        self.mode = mode

        # Concat normal baseline
        self.concat_normal_baseline = concat_normal_baseline

        # File directory for basline features, if concatenating them with normal is needed
        self.baseline_file_dir = baseline_file_dir

        # Usuable processors
        self.processors = processors

        # Baseline processing
        self.baseline = baseline

        # Preprocessing of data
        self.preprocess_part_data = preprocess_part_data

        # Preprocessing of data
        self.preprocess_only = preprocess_only

        # Preprocessing of data
        if preprocess_output_dir:
            self.preprocess_output_dir = preprocess_output_dir
        else:
            self.preprocess_output_dir = f'{self.file_dir}/processed'

        # Create path for mode
        if self.mode:
            output_dir = f'{output_dir}/{mode}'
        else:
            output_dir = f'{output_dir}/standard'

        # Check if outputdir exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Target path, where the processed dataset is saved
        self.target_path = f'{output_dir}' + '/{dataset_type}/{dir_i}/'

        # Load the datasets, that contain if datasets are train, test or val
        self.gdf_datasets = {dataset_type:gpd.read_file(self.shapefiles_path%(dataset_type))
                            for dataset_type in config.DATASET_TYPES}

        # Get a dictionary that maps dataset type and belonging points
        self.dataset_points = {dataset_type:gdf.FID.to_list()
                               for dataset_type, gdf in self.gdf_datasets.items()}
        
        self.coords = {row['FID']:
                       [coord[0] for coord in row['geometry'].centroid.coords.xy]
                       for dataset_type, gdf in self.gdf_datasets.items()
                       for idx, row in gdf.iterrows()}
        
        if baseline:
            # Save up to batch_size_tf_record samples in one tf record
            self.shard_size_tf_record = config.SHARD_SIZE_TF_RECORD_BASELINE
            # Set preprocesssing tf record helper if preprocessing enabled
            if self.preprocess_part_data:
                # Init context features
                context_features_serializing = config.prepare_baseline_GEE_feature_dict(self.years, mode = 'Serializing')
                # Init record helper
                self.preprocess_tfrecord_helper = TFRecordHelper(self.years, context_features_serializing = context_features_serializing)
                # Part feature dicts
                self.preprocess_feature_dicts = config.prepare_baseline_GEE_feature_dict(self.years, features_per_part=3)
        else:
            # Save up to batch_size_tf_record samples in one tf record
            self.shard_size_tf_record = config.SHARD_SIZE_TF_RECORD

        # Set maximum allowed number of file in dir
        self.max_files_in_dir = config.MAX_FILES_IN_DIR

        # Initialize serialized examples
        self.serialized_examples = {dataset_type:[] for dataset_type in self.dataset_points}

        # Initialize running batch and dir number
        self.batch_number = {dataset_type:0 for dataset_type in self.dataset_points}
        self.dir_number = {dataset_type:0 for dataset_type in self.dataset_points}

        # Initialize tf record input parsing helper
        self.tfrecord_helper = TFRecordHelper(years, baseline, mode=self.mode)

        # Specifiy logger namer
        if self.preprocess_part_data and self.preprocess_only:
            self.logger_name = f'preprocess_samples_baseline_{baseline}'
        elif self.preprocess_part_data:
            self.logger_name = f'pre_and_process_samples_baseline_{baseline}'
        else:
            self.logger_name = f'process_samples_baseline_{baseline}'

        # Init logger
        logging_config.create_logger(self.logger_name, config.LOGGING_PATH, True)
    
    # Get dataset type
    def _get_dataset_type(self, point_id):
        """Get type of the dataset.
        """
        for key, value in self.dataset_points.items():
            if point_id in value:
                return key

    # Save batch to drive     
    def get_file_name_for_data_chunk(self, dataset_type, data_chunk):
        """Get file name for the next processed data chunk.

        Args:
            dataset_type (str): Dataset type to which shard belongs.
        """
        if len(data_chunk):
            # Get current batch number
            batch_i = self.batch_number[dataset_type] 

            # Increase batch number
            self.batch_number[dataset_type] +=1

            # Increase dir_number if max_dir_size reached
            if not ((batch_i+1) % self.max_files_in_dir):
                self.dir_number[dataset_type]  +=1

            # Get current dir_number
            dir_i = self.dir_number[dataset_type] 

            # Get file dir
            file_dir = self.target_path.format(dataset_type = dataset_type,
                                               dir_i = dir_i
                                            )
            # Create file dir if not existing
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)

            # file_name
            file_name = f'{dataset_type}_{batch_i}.tfrecord.gz'

            if self.baseline:
                file_name = f'baseline_{file_name}'

            # File path
            file_path = f'{file_dir}/{file_name}'

            return file_path

    def process_samples(self):
        """ Process all samples coming from GEE and output in a tf.SequenceExample format.
        """

        # Get all the tf record files
        tf_record_files = glob(f'{self.file_dir}/*/*.tfrecord.gz')

        # Get the total points that have to be processed
        total_points = np.sum([len(gdf) for gdf in self.gdf_datasets.values()])

        # Initialize data chunks
        data_chunk = {dataset_type:[] for dataset_type in self.dataset_points}

        # Mode to concat normal and baseline features
        if self.concat_normal_baseline:
            self.tf_record_helper_baseline = TFRecordHelper(self.years, True)

            # Find out the location of every point id is stored
            point_info = {}
            csvs = glob(f'{self.baseline_file_dir}/*/*.csv')
            for csv in csvs:
                tf_record_name = csv.replace('_point_ids.csv', '.tfrecord.gz')
                point_ids = pd.read_csv(csv).point_id.to_list()
                point_info = {**point_info,**{point_id:tf_record_name for point_id in point_ids}}
            self.point_info = point_info
        # init mulitproc pool
        if self.processors:
            pool = pathos.multiprocessing.Pool(self.processors)
        else:
            pool = pathos.multiprocessing.Pool()

        results = []
        chunk_num = 0

        # Loop over every data point in the dataset
        for i, data in tqdm(enumerate(self.tfrecord_helper.parse_concatenated_dataset(tf_record_files)), total = total_points):
                    
            # Get point_id
            point_id = data['point_id'].numpy()[0].decode("utf-8")
            # Get dataset type
            dataset_type = self._get_dataset_type(point_id)
            # Skip points that are not in the accompanying shapefiles
            if dataset_type:
                # Process chunks of data after presorting
                # Append data to data chunk
                data_chunk[dataset_type].append(data)

                # Check if size for data chunk is reached, then save chards
                if len(data_chunk[dataset_type]) >= self.shard_size_tf_record:
                        # Get filename for data chunk
                        file_path = self.get_file_name_for_data_chunk(dataset_type, data_chunk[dataset_type])
                        # Get point ids 
                        point_ids =  {data['point_id'].numpy()[0].decode("utf-8") for data in data_chunk[dataset_type]}
                        # Save point ids to track in which chunk which point id was found
                        pd.DataFrame(set(point_ids), columns=['point_id']).to_csv(f'{file_path}_point_ids.csv')
                        # Process data chunk
                        # self.process_data_chunk(data_chunk[dataset_type], file_path)
                        pool.apply_async(self.process_data_chunk,args=(data_chunk[dataset_type],file_path), callback=results.append)
                        # Reset data chunk
                        data_chunk[dataset_type] = []
                        chunk_num+=1
            else:
                # Too much logs for baseline
                if not self.baseline:
                    logging_config.logger.info(f'{data["point_id"].numpy()[0].decode("utf-8")} tile skipped because it is not in stratified sample.')

            # Log how many tiles are processed
            if i%5000 == 0:
                logging_config.logger.info(f'{i} tiles are processed or in processing')

        # After loop is done save last chunks
        logging_config.logger.info(f'Looping over all data is finished, waiting for data creation to be finished')
        for dataset_type, data in data_chunk.items():
            # Get filename for data chunk
            file_path = self.get_file_name_for_data_chunk(dataset_type, data)
            # Get point ids 
            point_ids =  {data['point_id'].numpy()[0].decode("utf-8") for data in data_chunk[dataset_type]}
            # Save point ids to track in which chunk which point id was found
            pd.DataFrame(set(point_ids), columns=['point_id']).to_csv(f'{file_path}_point_ids.csv')
            # Process data chunk
            # self.process_data_chunk(data, file_path)
            pool.apply_async(self.process_data_chunk,args=(data_chunk[dataset_type],file_path), callback=results.append)

            # Reset data chunk
            data_chunk[dataset_type] = []
            chunk_num+=1

        progress_bar = tqdm(total=chunk_num, position=0)

        def _update_progess(num_results):
            # Update progressbar
            logging_config.logger.info(f'Processing {num_results} out {chunk_num} are finished!')
            progress_bar.n = num_results
            progress_bar.last_print_n = num_results
            progress_bar.refresh()

        previous_finished_results = 0
        while len(results) < chunk_num:
            if previous_finished_results != len(results):
                # update progress
                _update_progess(len(results))
                previous_finished_results = len(results)
            time.sleep(5)

        # Wait for all processes to be closed
        pool.close()
        pool.join()
        # update progress
        _update_progess(len(results))
        # log progress
        logging_config.logger.info(f'Process samples finished')

    def process_data_chunk(self, data_chunk, file_path):
        if '64' in self.mode:
            target_size = (64, 64)
        else:
            target_size = (config.KERNEL_SIZE, config.KERNEL_SIZE)
        logging_config.create_logger(self.logger_name, config.LOGGING_PATH, True)
        try:
            if data_chunk:
                start_time = time.time()
                logging_config.logger.info(f'Processing chunk {file_path} started')
                # Check if file is already created
                if os.path.exists(file_path):
                    pass
                else:
                    if self.concat_normal_baseline:
                        logging_config.logger.info(f'Get baseline features for {file_path}')
                        # Store baseline features
                        baseline_features = {}
                        # Get point ids in data chunk
                        point_ids =  [data['point_id'].numpy()[0].decode("utf-8") for data in data_chunk]
                        # Retriveve correct tf record files
                        tf_record_files_baseline= list({self.point_info[point_id] for point_id in point_ids})
                        for data in self.tf_record_helper_baseline.parse_concatenated_dataset(tf_record_files_baseline):
                            # Get point id
                            point_id = data['point_id'].numpy()[0].decode("utf-8")
                            # Only process if in point_ids list
                            if point_id in point_ids:
                                point_id, baseline_features[point_id] = self.process_baseline_sample(data, target_size)

                    serialized_examples = []
                    for data in data_chunk:
                        # Merge normal and baseline features
                        if self.concat_normal_baseline:
                            # Get point id
                            point_id = data['point_id'].numpy()[0].decode("utf-8")
                            serialized_examples += self.process_and_serialize_normal_concat_sample(data, baseline_features[point_id])
                        # Normal processing
                        elif not self.baseline:  
                            serialized_examples += self.process_and_serialize_normal_sample(data)
                        # Baseline
                        else:
                            serialized_examples += self.process_and_serialize_baseline_sample(data, target_size)

                    # Create tf record file with data chunk
                    self.tfrecord_helper.write_serialized_samples(serialized_examples, file_path)
                elapsed_time =  time.time()-start_time

                logging_config.logger.info(f'Processing chunk finished, elapsed: {elapsed_time}, amount: {len(data_chunk)}, per tile: {elapsed_time/len(data_chunk)}')
        except:
            logging_config.create_logger('ExceptionLog', config.LOGGING_PATH, True)
            logging_config.logger.exception('Exception occurred: ')
            raise  # Re-raise the exception so that the process exits 

    def process_and_serialize_normal_sample(self, data):

        # Process normal sample
        example = self.process_normal_sample(data)
        # Serialize example
        serialized_example = self.tfrecord_helper.serialize_sequence_example(example['context_features'],
                                                                             example['sequential_features'])
        return [serialized_example]
    
    def process_normal_sample(self, data):
        # Init sample
        example = {}
        
        # Get point_id
        point_id = data['point_id'].numpy()[0].decode("utf-8")
        
        # Target size
        if '64' in self.mode:
            target_size = (64,64)
        elif '128' in self.mode:
            target_size = (128,128)

        # Downscale to 64px or 128px
        if self.mode == '64px' or self.mode == '128px' or self.mode == '64px_non_cumu' or self.mode == '64px_comb' or self.mode == '128px_comb' or self.mode == '64px_normal_concat_baseline':

            def _resize_and_prepare(feature):
                if feature.shape == (config.KERNEL_SIZE,config.KERNEL_SIZE):
                    return list(skimage.transform.resize(feature.numpy(), target_size).flatten())
                else:
                    return TFRecordHelper._tf_to_list(feature)

            # Parse context features
            example['context_features'] = {context_feature: _resize_and_prepare(data[context_feature])
                                            for context_feature in config.CONTEXT_FEATURES} #TODO: check which features are in there

            # Pointid
            example['context_features']['point_id'] = TFRecordHelper._tf_to_list(data['point_id'])
            # Get coords
            coords = self.coords[point_id]
            example['context_features']['coords'] = coords

            # Parse forest label
            binary_forest_loss_label = data['lossyear'] == int(str(self.years[-1])[-2:])
            example['context_features'][config.LABEL] = _resize_and_prepare(binary_forest_loss_label)

            # Parse sequential features
            example['sequential_features'] = {}
            # Init different bands
            for band in config.BANDS:
                example['sequential_features'][band] = []
            
            example['sequential_features']['forest_loss'] = []
            example['sequential_features']['year'] = []

            if self.mode == '64px_comb' or self.mode == '64px_normal_concat_baseline' or self.mode == '128px_comb':
                example['sequential_features']['forest_loss_cumu'] = []

            # Parse forest loss and satellite bands of the different years as sequential features
            for year in self.years[:-1]:
                if self.mode == '64px_comb' or self.mode == '64px_normal_concat_baseline' or self.mode == '128px_comb':
                    binary_forest_loss_mask = data['lossyear']== int(str(year)[-2:])

                    binary_forest_loss_mask_cumu = tf.math.logical_and(data['lossyear']> 0,
                                                                       data['lossyear']<= int(str(year)[-2:]))

                    example['sequential_features']['forest_loss'].append(_resize_and_prepare(binary_forest_loss_mask))
                    example['sequential_features']['forest_loss_cumu'].append(_resize_and_prepare(binary_forest_loss_mask_cumu))

                else:
                    if self.mode == '64px_non_cumu':
                        binary_forest_loss_mask = data['lossyear']== int(str(year)[-2:])
                    else:
                        binary_forest_loss_mask = tf.math.logical_and(data['lossyear']> 0,
                                                                    data['lossyear']<= int(str(year)[-2:]))
            
                    example['sequential_features']['forest_loss'].append(_resize_and_prepare(binary_forest_loss_mask))
                
                example['sequential_features']['year'].append(year)
                
                # Loop over different bands
                for band in config.BANDS:
                    example['sequential_features'][band].append(_resize_and_prepare(data[f'{band}_{year}']))
        elif self.mode=='256px_loss_only' or self.mode=='256px_cum_loss_only':
            example['context_features'] = {}
            # Parse context features
            # Pointid
            example['context_features']['point_id'] = TFRecordHelper._tf_to_list(data['point_id'])
            # Get coords
            coords = self.coords[point_id]
            example['context_features']['coords'] = coords

            # Parse forest label
            # Store complete sample!
            #binary_forest_loss_label = tf.cast(data['lossyear'] == int(str(self.years[-1])[-2:]), dtype=tf.float32)
            forest_loss_label = TFRecordHelper._tf_to_list(data['lossyear'])
            example['context_features'][config.LABEL] = forest_loss_label
            # Parse sequence features
            
            # Parse sequential features
            example['sequential_features'] = {}
            example['sequential_features']['forest_loss'] = []
            # Parse forest loss and satellite bands of the different years as sequential features
            for year in self.years[:-1]:
                if self.mode=='256px_cum_loss_only':
                    # Cumulated, every forest loss occured before this date
                    binary_forest_loss_mask = tf.cast(
                                                      tf.math.logical_and(data['lossyear'] >0,
                                                                          data['lossyear']<= int(str(year)[-2:]))
                                                      ,dtype=tf.float32)
                else:
                    # Only forest loss per year
                    binary_forest_loss_mask = tf.cast(data['lossyear'] == int(str(year)[-2:]), dtype=tf.float32)
                example['sequential_features']['forest_loss'].append(TFRecordHelper._tf_to_list(binary_forest_loss_mask))
        
        elif self.mode == '256px_seg_only' or self.mode == '256px_cum_seg_only':
            example['context_features'] = {}
            # Parse context features
            # Pointid
            example['context_features']['point_id'] = TFRecordHelper._tf_to_list(data['point_id'])
            # Get coords
            coords = self.coords[point_id]
            example['context_features']['coords'] = coords

            # Parse forest label
            # Store complete sample!
            #binary_forest_loss_label = tf.cast(data['lossyear'] == int(str(self.years[-1])[-2:]), dtype=tf.float32)
            forest_loss_label = TFRecordHelper._tf_to_list(data['lossyear'])
            example['context_features'][config.LABEL] = forest_loss_label
            # Parse sequence features
            
            # Parse sequential features
            example['sequential_features'] = {}
            example['sequential_features']['forest_loss'] = []

            # Parse forest loss and satellite bands of the different years as sequential features
            for year in range(2001,2019):
                if self.mode == '256px_cum_seg_only':
                    # Cumulated, every forest loss occured before this date
                    binary_forest_loss_mask = tf.cast(
                                                        tf.math.logical_and(data['lossyear'] >0,
                                                                            data['lossyear']<= int(str(year)[-2:]))
                                                        ,dtype=tf.float32)
                else:
                    # Only forest loss per year
                    binary_forest_loss_mask = tf.cast(data['lossyear'] == int(str(year)[-2:]), dtype=tf.float32)
                example['sequential_features']['forest_loss'].append(TFRecordHelper._tf_to_list(binary_forest_loss_mask))

        else:        
            # Parse context features
            example['context_features'] = {context_feature: TFRecordHelper._tf_to_list(data[context_feature])
                                            for context_feature in config.CONTEXT_FEATURES} #TODO: check which features are in there

            # Get coords
            coords = self.coords[point_id]
            example['context_features']['coords'] = coords

            # Parse forest label
            binary_forest_loss_label = tf.cast(data['lossyear'] == int(str(self.years[-1])[-2:]), dtype=tf.float32)
            example['context_features'][config.LABEL] = TFRecordHelper._tf_to_list(binary_forest_loss_label)

            # Parse sequential features
            example['sequential_features'] = {}
            # Init different bands
            for band in config.BANDS:
                example['sequential_features'][band] = []
            
            example['sequential_features']['forest_loss'] = []
            example['sequential_features']['year'] = []

            # Parse forest loss and satellite bands of the different years as sequential features
            for year in self.years[:-1]:
                binary_forest_loss_mask = tf.cast(data['lossyear'] == int(str(year)[-2:]), dtype=tf.float32)
                example['sequential_features']['forest_loss'].append(TFRecordHelper._tf_to_list(binary_forest_loss_mask))
                example['sequential_features']['year'].append(year)
                
                # Loop over different bands
                for band in config.BANDS:
                    example['sequential_features'][band].append(TFRecordHelper._tf_to_list(data[f'{band}_{year}']))
            
        return example

    def process_and_serialize_baseline_sample(self, data, target_size = (config.KERNEL_SIZE, config.KERNEL_SIZE)):

        # Get context features
        point_id, context_features = self.process_baseline_sample(data, target_size)
        # Serialize samples
        # Init list of serialized examples
        serialized_examples = []

        def _serialize_baseline_sample(i):
            # Init sample
            example = {}     
            # Init context features
            example['context_features'] = {}

            # Init other features 
            for context_feature, context_feature_value in context_features.items():
                if len(context_feature_value) == 1:
                    example['context_features'][context_feature] = context_feature_value[0]
                else:
                    example['context_features'][context_feature] = context_feature_value[i] # TODO: Check if working

            # Point id
            example['context_features']['point_id'] = point_id.encode("utf-8")
            # Init coords
            coords = self.coords[point_id]
            example['context_features']['coords'] = coords

            # init part point id
            example['context_features']['px_id'] = i

            # Return serialized example
            return self.tfrecord_helper.serialize_example(example['context_features'])

        # Get logger
        logging_config.create_logger(self.logger_name, config.LOGGING_PATH, True)

        # Log
        logging_config.logger.info(f'start processing {point_id}')
        start_time = time.time()
        serialized_examples = [_serialize_baseline_sample(i) for i in range(np.prod(target_size))]
        # Log elapsed time
        logging_config.logger.info(f'{start_time-time.time()} elapsed')

        return serialized_examples

    def process_baseline_sample(self, data, target_size = (config.KERNEL_SIZE, config.KERNEL_SIZE)):

        def _resize_and_prepare(feature):
            if target_size == config.KERNEL_SIZE:
                return TFRecordHelper._tf_to_list(feature)
                
            if feature.shape == (config.KERNEL_SIZE,config.KERNEL_SIZE):
                return list(skimage.transform.resize(feature.numpy(), target_size).flatten())
            else:
                return TFRecordHelper._tf_to_list(feature)

        # Get point_id
        point_id = data['point_id'].numpy()[0].decode("utf-8")

        # Init context features dict
        context_features = {}

        # Get forest loss label
        binary_forest_loss_mask = data[config.FOREST_BANDS[0]] == int(str(self.years[-1])[-2:])

        context_features[config.FOREST_LOSS_LABEL] = _resize_and_prepare(binary_forest_loss_mask)

        # Get treecover
        context_features[config.FOREST_BANDS[1]] = _resize_and_prepare(data[config.FOREST_BANDS[1]])

        # Get pop density
        pop_years = config._get_closest_years(self.years[-1], config.POPULATION_DENSITY_YEARS)

        # Get second last year
        year = self.years[-2]
        pop_density_band = f'{config.POP_IMAGE_BAND}_%s'
        # Pop density calc per years   
        if year in pop_years:
            context_features[pop_density_band%(year)] =  _resize_and_prepare(data[pop_density_band%(year)])
        else:
            first_year, second_year = pop_years

            first_year_value = data[pop_density_band%(first_year)]
            second_year_value = data[pop_density_band%(second_year)]
        
            # Get linearly adjusted value of this year
            value = (year-first_year)*((second_year_value - first_year_value)/(second_year - first_year)) + first_year_value 

            # Set value
            context_features[config.POP_IMAGE_BAND] = _resize_and_prepare(value)

            del value

        # Distance to city centre
        dist_city_centre_band = f'{config.DIST_TO_CITY_CENTRE_BAND}_%s'

        if year in pop_years:
            context_features[config.DIST_TO_CITY_CENTRE_BAND] =  _resize_and_prepare(data[dist_city_centre_band%(year)])
        else:
            # Pick closest earlier year
            first_year, _ = pop_years
            context_features[config.DIST_TO_CITY_CENTRE_BAND] =  _resize_and_prepare(data[dist_city_centre_band%(first_year)])
                                                                            
        # Distance to forest loss.select('cumulative_cost')
        band_name = config.DIST_TO_FOREST_BAND
        context_features[band_name] = _resize_and_prepare(data[band_name])

        # Distance to roads loss.select('cumulative_cost')
        band_name = config.DIST_TO_ROADS_BAND
        context_features[band_name] = _resize_and_prepare(data[band_name])

        # Forest loss in 256*256 patch - for all years
        # Get forest loss for all years
        forest_loss_all_years_greater = data[config.FOREST_BANDS[0]] > 0
        forest_loss_all_years_lower =  data[config.FOREST_BANDS[0]] < int(str(self.years[-1])[-2:])
        forest_loss_all_years = tf.cast(
                                        tf.math.logical_and(
                                            forest_loss_all_years_greater,
                                            forest_loss_all_years_lower)
                                        ,dtype=tf.float32)
        proportion = tf.reduce_sum(forest_loss_all_years)/(config.KERNEL_SIZE*config.KERNEL_SIZE)
        context_features[config.FOREST_LOSS_PROPORTION_ALL] = [proportion.numpy()]

        # Forest loss in 256*256 patch - for the last years

        greater_than = data[config.FOREST_BANDS[0]] >= int(str(self.years[0])[-2:])
        less_than_lbl_year = data[config.FOREST_BANDS[0]] < int(str(self.years[-1])[-2:])

        forest_loss_last_years = tf.cast(tf.math.logical_and(greater_than, less_than_lbl_year),dtype=tf.float32)
        proportion = tf.reduce_sum(forest_loss_last_years)/(config.KERNEL_SIZE*config.KERNEL_SIZE)
        context_features[config.FOREST_LOSS_PROPORTION_LAST_YEARS] = [proportion.numpy()]

        # Forest loss in direct vicinity
        k = tf.constant([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=tf.float32, name='k')

        kernel = tf.reshape(k, [3, 3, 1, 1], name='kernel')
        image  = tf.reshape(forest_loss_last_years, [1, config.KERNEL_SIZE, config.KERNEL_SIZE, 1], name='image')
        context_features[config.FOREST_LOSS_PROPORTION_DIRECT] = _resize_and_prepare(tf.squeeze(tf.nn.conv2d(image, kernel, [1, 1, 1, 1], "SAME"))/9)

        # Forest loss occured before
        context_features[config.FOREST_LOSS_ON_THIS_PX_BEFORE] =  _resize_and_prepare(forest_loss_all_years)

        # Elevation
        band_name = config.ELEV_BAND
        context_features[band_name] = _resize_and_prepare(data[band_name])

        # Slope
        band_name = config.SLOPE_BAND
        context_features[band_name] = _resize_and_prepare(data[band_name])
        
        return point_id, context_features

    def process_and_serialize_normal_concat_sample(self, data, data_baseline):
        # Logging
        logging_config.create_logger(self.logger_name, config.LOGGING_PATH, True)
        # Process normal sample
        example = self.process_normal_sample(data)
        
        
        # Add more context features
        for feature_name, feature_value in data_baseline.items():
            if feature_name not in example['context_features'] and feature_name in self.tfrecord_helper.context_features_serializing:
                example['context_features'][feature_name] = feature_value
        
        # logging_config.logger.info(f'Normal elevation shape: {len(example["context_features"][config.ELEV_BAND])}, baseline elevation shape: {len(data_baseline[config.ELEV_BAND])} ')

        # Serialize example
        serialized_example = self.tfrecord_helper.serialize_sequence_example(example['context_features'],
                                                                             example['sequential_features'])

        return [serialized_example]

    def preprocess_folder_batch(self, folder_batch):
        logging_config.create_logger(self.logger_name, config.LOGGING_PATH, True)
        try:
            # Get logger
            # Loop over every folder and so over every conacatenated samples
            for folder in folder_batch:
                # Get last folder name
                last_folder = (os.path.normpath(folder)).split(os.sep)[-1]

                # Check if path already exists
                file_dir = f'{self.preprocess_output_dir}/{last_folder}/'
                file_path = f'{file_dir}/{last_folder}%s'

                # Get tf record file
                tf_record_files = glob(f'{folder}/*.tfrecord.gz')

                # Check if already processed
                if os.path.exists(file_path%('.tfrecord.gz')):
                    logging_config.logger.info(f'Folder {folder} skipped, already processed file found.')
                # Check if there are tf record files
                elif tf_record_files:
                    # Process folder
                    self.process_one_folder(folder, tf_record_files, file_dir, file_path)
                else:
                    logging_config.logger.info(f'Folder {folder} skipped, because no tfrecord files were found.')
        except:
            logging_config.create_logger('ExceptionLog', config.LOGGING_PATH, True)
            logging_config.logger.exception('Exception occurred: ')
            raise  # Re-raise the exception so that the process exits 
        return len(folder_batch)

    def preprocess_part_datasets(self):
        # Get folders
        folders = glob(f'{self.file_dir}/*/')
        progress_bar = tqdm(total=len(folders), position=0)

        # Create multiprocessing pool
        if self.processors:
            pool = pathos.multiprocessing.Pool(self.processors)
        else:
            pool = pathos.multiprocessing.Pool()

        batches = 32
        num_batches = int(math.ceil(len(folders)/batches))
        folder_batches = np.array_split(folders, num_batches)

        results = []
        for folder_batch in folder_batches:
            pool.apply_async(self.preprocess_folder_batch, args=(folder_batch,), callback=results.append)
            # results.append(self.preprocess_folder_batch(folder_batch))

        previous_finished_results = 0
        while np.sum(np.array(results)) < len(folders):
            if previous_finished_results != np.sum(np.array(results)):
                total_finished = np.sum(np.array(results))
                # Update progressbar
                logging_config.logger.info(f'Processing {total_finished} out {len(folders)} are finished!')
                progress_bar.n = total_finished
                progress_bar.last_print_n = total_finished
                progress_bar.refresh()
                # Save current result length
                previous_finished_results = total_finished
                time.sleep(10)
            else:
                time.sleep(5)
        else:
            # Update progressbar
            pool.close()
            pool.join()
            time.sleep(1)
            # Wait for multiprocessing to be finished completely
            total_finished = np.sum(np.array(results))
            logging_config.logger.info(f'Processing {total_finished} out {len(folders)} are finished!')
            logging_config.logger.info(f'Folder processing finished!')
            # Update progressbar
            progress_bar.n = total_finished
            progress_bar.last_print_n = total_finished
            progress_bar.refresh()

        # Set file dir to processed data
        self.file_dir = self.preprocess_output_dir

    def process_one_folder(self, folder, tf_record_files, file_dir, file_path):
        # Init logger
        logging_config.create_logger(self.logger_name, config.LOGGING_PATH, True)
        # Log start time
        start_time = time.time()
        data_points = {} 
        # Create a list of point ids, that take track which point ids are stored within the file
        point_ids = []
        # Loop over every part file and set together the combined file
        for tf_record_file in tf_record_files:
            part = int(re.findall(r'part_(\d+)', tf_record_file)[0])
            feature_dict = self.preprocess_feature_dicts[part-1]
            
            # loop over data
            for data in self.preprocess_tfrecord_helper.parse_concatenated_dataset(tf_record_file, feature_dict):
                point_id = data['point_id'].numpy()[0].decode("utf-8")
                # Append point_id
                point_ids.append(point_id)
                if point_id not in data_points:
                    data_points[point_id] = {}
                    data_points[point_id]['point_id'] = point_id.encode("utf-8")
                # Loop over all features that are found in this part
                for feature_key in feature_dict:
                    if feature_key != 'point_id':                
                        data_points[point_id][feature_key] = TFRecordHelper._tf_to_list(data[feature_key])       
        
        # Get complete list of all distinct features
        self.features = {feature:feature_type
                            for feature_dict in self.preprocess_feature_dicts
                            for feature, feature_type in feature_dict.items()}

        # Serialize samples
        serialized_examples = []
        missing_features_all = set()
        #logging_config.logger.info(f'Thread serializing started')
        for point_id, data_point in data_points.items():
            missing_features, serialized_example = self.serialize_data_point(point_id, data_point)
            serialized_examples.append(serialized_example)
            missing_features_all.union(set(missing_features))

        if missing_features_all:
            # Log error
            logging_config.logger.info(f'Features {missing_features_all} are missing for {file_path%("")}')

        # Create file dir
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        # Write information about contained point ids
        pd.DataFrame(set(point_ids), columns=['point_id']).to_csv(file_path%('_point_ids.csv'))

        # Write serialized examples
        self.preprocess_tfrecord_helper.write_serialized_samples(serialized_examples, file_path%('.tfrecord.gz'))
        # Log processing time
        logging_config.logger.info(f'Finished folder {folder}, {time.time()-start_time} elapsed')

        del serialized_examples
        del data_points
        # Clear session
        tf.keras.backend.clear_session()
        gc.collect()

    def serialize_data_point(self, point_id, data_point):
        missing_features = []
        # Make sure, that every feature is set, even if it was not sampled via GEE
        # Check for missing data and fill missing data with -99 as value
        for feature, feature_value in self.features.items():
            if feature not in data_point:
                shape = feature_value.shape
                data_point[feature] = [-99]*np.prod(np.array(shape))
                logging_config.logger.info(f'Missing values for feature {feature} for point {point_id}')
                missing_features.append(feature)
        
        # Logging
        # Serialize example
        return missing_features, self.preprocess_tfrecord_helper.serialize_example(data_point)

    def start_processing(self):
        # Preprocess dataset
        if self.preprocess_part_data:
            self.preprocess_part_datasets()

        # Process samples
        if not self.preprocess_only:
            self.process_samples()

def main(input_dir,
         shapefiles_path,
         year_range,
         output_dir,
         baseline = False,
         preprocess_part_data = False,
         preprocess_only = False,
         preprocess_output_dir = None,
         processors = None,
         mode = None,
         concat_normal_baseline = False,
         baseline_file_dir = None):   
    # Init process helper
    process_helper = ProcessSampledTilesHelper(input_dir,
                                               shapefiles_path,
                                               year_range,
                                               output_dir,
                                               baseline,
                                               preprocess_part_data,
                                               preprocess_only,
                                               preprocess_output_dir,
                                               processors,
                                               mode,
                                               concat_normal_baseline,
                                               baseline_file_dir)
    
    process_helper.start_processing()

if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser(description='Process sampled patches.')

    # Input directory
    parser.add_argument('--input_dir', default=None)

    # Input directory
    parser.add_argument('--baseline_file_dir', default=None)

    # Shapefiles path
    parser.add_argument('--shapefiles_path', default=None)

    # Sample years
    parser.add_argument('--year_range',  nargs="+", default = [2013, 2020], type=int)

    # Output directoryinput_base_name
    parser.add_argument('--output_dir', default=None)

    # Output directoryinput_base_name
    parser.add_argument('--preprocess_output_dir', default=None)

    # Process mode
    parser.add_argument('--baseline', action="store_true", default=False)

    # Process mode
    parser.add_argument('--concat_normal_baseline', action="store_true", default=False)

    # Preprocessing needed
    parser.add_argument('--preprocess_part_data', action="store_true", default=False)

    # Preprocessing needed
    parser.add_argument('--preprocess_only', action="store_true", default=False)

    # Processors to be used
    parser.add_argument('--processors', default=None, type=int)

    # Processors
    parser.add_argument('--mode', default=None, type=str)

    # Evaluate argparse object
    args = parser.parse_args()

    # Call main 
    main(args.input_dir,
         args.shapefiles_path,
         range(*args.year_range),
         args.output_dir,
         args.baseline,
         args.preprocess_part_data,
         args.preprocess_only,
         args.preprocess_output_dir,
         args.processors,
         args.mode,
         args.concat_normal_baseline,
         args.baseline_file_dir)