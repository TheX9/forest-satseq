import config
from glob import glob

import argparse
import os
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

from tqdm import tqdm
import tensorflow as tf

class LoadDatasetHelper:
    def __init__(self,
                 file_path,
                 baseline=False,
                 check_dataset=False,
                 normalization=False,
                 train_batch_size=config.BATCH_SIZE,
                 val_test_batch_size=config.BATCH_SIZE,
                 mode = None
                 ):
        self.file_path = file_path
        self.baseline = baseline
        self.check_dataset = check_dataset
        self.train_batch_size = train_batch_size
        self.val_test_batch_size = val_test_batch_size
        self.normalization = False
        self.mode = mode
        self.context_features = None
        self.sequence_features = None
        self.sample_weights = 400
        self.complete_label = False
        self.years = 6
        self.kernel_size = config.KERNEL_SIZE

        # Get context and sequence parsing features
        if baseline:
            self.context_features = config.get_features_baseline_serializing_or_parsing(mode = 'Parsing')
        else:
            if mode:
                con, seq = config.get_various_dataset_features_serializing_or_parsing('Parsing', mode)
                self.context_features = con
                self.sequence_features = seq
            else:
                self.context_features = config.CONTEXT_FEATURES_PARSING
                self.sequence_features = config.SEQUENCE_FEATURES_PARSING

    def get_parse_function(self, baseline = False, check_dataset = False):
        # If baseline dataset
        if baseline:
            # If check dataset
            if check_dataset:
                def _parse(example_proto):
                    return tf.io.parse_example(example_proto, self.context_features)
        else:
            # If check dataset
            if check_dataset:
                def _parse(example_proto):
                    return tf.io.parse_sequence_example(example_proto,
                                                        self.context_features,
                                                        self.sequence_features)
            else:
                def _parse(example_proto):
                    return tf.io.parse_sequence_example(example_proto,
                                                        self.context_features,
                                                        self.sequence_features)
        
        return _parse

    def prepare_baseline_dataset(self, file_path, dataset_type ='train', filter_lbl = True):
        context_features = config.get_features_baseline_serializing_or_parsing(mode = 'Parsing')
        def _parse(example_proto):
            parsed_features = tf.io.parse_example(example_proto, context_features)
            label = parsed_features[config.BASELINE_LABEL]
            
            del parsed_features[config.BASELINE_LABEL]
            # Create two tensors for coords
            #features['coord_lat'], features['coord_lon'] = tf.split(features['coords'], 2, axis=1)
            coords = parsed_features['coords']
            del parsed_features['coords']

            point_id = parsed_features['point_id']
            px_id = parsed_features['px_id']

            # Clean unwanted features
            del parsed_features['point_id']
            del parsed_features['px_id']

            # Unscaled
            features = {}
            features['forest_loss_direct_direct'] = parsed_features['forest_loss_direct_direct']
            features['forest_loss_on_this_px_before'] = parsed_features['forest_loss_on_this_px_before']
            if 'add_feat' in self.mode:
                features['forest_loss_proportion_all'] = parsed_features['forest_loss_proportion_all']
                features['forest_loss_proportion_last_years'] = parsed_features['forest_loss_proportion_last_years']
            del parsed_features['forest_loss_direct_direct']
            del parsed_features['forest_loss_on_this_px_before']
            del parsed_features['forest_loss_proportion_all']
            del parsed_features['forest_loss_proportion_last_years']

            # normalization
            features = {**features, **{(feature):
                        (value/config.BASELINE_NORMALIZATION[feature][1]
                        if config.BASELINE_NORMALIZATION[feature][0] =='/'
                        #y = (x - min) / (max - min)
                        else (value - config.BASELINE_NORMALIZATION[feature][1])/(config.BASELINE_NORMALIZATION[feature][2]-config.BASELINE_NORMALIZATION[feature][1]))
                        for feature, value in parsed_features.items()}
                        }

            # features = {(feature):(value/config.BASELINE_NORMALIZATION[feature][1])
            #             for feature, value in features.items()
            #             if config.BASELINE_NORMALIZATION[feature][0] =='/'}

            # concat features
            features = tf.concat(list(features.values()), axis=1)

            return {'in_bas':features, 'point_id':point_id, 'px_id': px_id, 'coords':coords},{'out':label}

        # Find tf records files
        tf_record_files = glob(f'{file_path}/*/*.tfrecord.gz')
        tf_record_files = [transform_path(file) for file in tf_record_files]

        # Load dataset files
        dataset = tf.data.Dataset.from_tensor_slices(tf_record_files)

        if dataset_type == 'train':
            # Random shuffle ordering of files
            dataset = dataset.shuffle(config.SHUFFLE_SIZE_FILES)

        # Create TFRecord dataset using interleaving (num_parallel_reads)
        dataset = tf.data.TFRecordDataset(dataset,
                                        compression_type='GZIP',
                                        num_parallel_reads=tf.data.experimental.AUTOTUNE
                                        )
        if dataset_type == 'train':
            dataset = dataset.shuffle(config.SHUFFLE_SIZE)           


        print(dataset_type)
        if dataset_type in ['test', 'val']:
            print('val_batch')
            # Batch the data
            dataset = dataset.batch(self.val_test_batch_size, drop_remainder = False) # ,  drop_remainder = True)
        else:
            dataset = dataset.batch(self.train_batch_size, drop_remainder = False) # ,  drop_remainder = True)
        
        # Parse the data from tf_record format
        dataset = dataset.map(_parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)    

        # Prefetch data
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


    def prepare_dataset(self, file_path, _parse, dataset_type ='train'):
        # Determine batch size
        if dataset_type in ['test', 'val']:
            print('test', self.val_test_batch_size)
            batch_size = self.val_test_batch_size
        else:
            print('train', self.train_batch_size)
            batch_size = self.train_batch_size

        def _parse(example_proto):
            parsed_features = tf.io.parse_sequence_example(example_proto,
                                                self.context_features,
                                                self.sequence_features)

            label = tf.reshape(parsed_features[0][config.LABEL], [batch_size, self.kernel_size,self.kernel_size,1])
            point_id =  parsed_features[0]['point_id']
            coords = parsed_features[0]['coords']
            
            if self.mode == '64px_normal_concat_baseline':
                # Parse all features
                skip_features = ['forest_loss_cumu', 'forest_loss', 'year']
                # Normalization
                sequence_features = tf.concat([tf.reshape(
                                                (feature-config.NORMALIZATION[key][1])/
                                                (config.NORMALIZATION[key][2]-config.NORMALIZATION[key][1])
                                                ,
                                                [batch_size,self.years,self.kernel_size,self.kernel_size,1])
                                        for key, feature in parsed_features[1].items() if key not in skip_features], axis=-1)
                
                
                context_names = [config.POP_IMAGE_BAND, config.DIST_TO_CITY_CENTRE_BAND, config.DIST_TO_FOREST_BAND, config.DIST_TO_ROADS_BAND, config.FOREST_BANDS[1], config.ELEV_BAND, config.SLOPE_BAND]
                last_binary_forest_mask = tf.reshape(parsed_features[1]['forest_loss_cumu'][:,-1,:,:],[batch_size,self.kernel_size,self.kernel_size,1])
                    
                context_features = tf.concat([tf.reshape(
                                                (feature-config.NORMALIZATION[key][1])/
                                                (config.NORMALIZATION[key][2]-config.NORMALIZATION[key][1])
                                                ,
                                                [batch_size,self.kernel_size,self.kernel_size,1])
                                        for key, feature in parsed_features[0].items() if key in context_names]
                                        + [last_binary_forest_mask], axis=-1)

                return {'in_seq':sequence_features, 'in_context':context_features} ,{'out':label}            
            elif self.mode == '256px_loss_only':
                # For now only test loss_year
                features = tf.reshape(parsed_features[1]['forest_loss'],[batch_size,self.years,self.kernel_size,self.kernel_size,1])
                
                if self.sample_weights:
                    sample_weights = parsed_features[0][config.LABEL]

                    sample_weights = tf.cast(
                                            tf.reduce_sum(
                                                        tf.reshape(parsed_features[0][config.LABEL],
                                                                    [batch_size,self.kernel_size,self.kernel_size]),
                                                        axis=(1,2)
                                                        ) 
                                            >= self.sample_weights,
                                            dtype=tf.float32)

                    sample_weights += tf.ones_like(sample_weights)
                    sample_weights = tf.reshape(sample_weights, [batch_size, 1])
                    
            elif self.mode == '256px_cum_loss_only':
                # For now only test loss_year
                features = tf.reshape(parsed_features[1]['forest_loss'],[batch_size,self.years,self.kernel_size,self.kernel_size,1])
                
            elif self.mode == '64px_comb_img_only': 
                skip_features = ['forest_loss_cumu', 'forest_loss', 'year']
                if self.normalization:
                    features = tf.concat([tf.reshape(
                                                    (feature-config.NORMALIZATION[key][1])/
                                                    (config.NORMALIZATION[key][2]-config.NORMALIZATION[key][1])
                                                    ,
                                                    [batch_size,self.years,self.kernel_size,self.kernel_size,1])
                                          for key, feature in parsed_features[1].items() if key not in skip_features], axis=-1)
                else:
                    features = tf.concat([tf.reshape(feature,[batch_size,self.years,self.kernel_size,self.kernel_size,1])
                                          for key, feature in parsed_features[1].items() if key not in skip_features], axis=-1)
            
            elif self.mode == '64px_comb_img_only_context':  
                # Parse all features
                skip_features = ['forest_loss_cumu', 'forest_loss', 'year']
                # normalization
                if self.normalization:
                    sequence_features = tf.concat([tf.reshape(
                                                    (feature-config.NORMALIZATION[key][1])/
                                                    (config.NORMALIZATION[key][2]-config.NORMALIZATION[key][1])
                                                    ,
                                                    [batch_size,self.years,self.kernel_size,self.kernel_size,1])
                                          for key, feature in parsed_features[1].items() if key not in skip_features], axis=-1)
                    context_names = ['elevation', 'treecover2000']

                    last_binary_forest_mask = tf.reshape(parsed_features[1]['forest_loss_cumu'][:,-1,:,:],[batch_size,self.kernel_size,self.kernel_size,1])
                     
                    context_features = tf.concat([tf.reshape(
                                                    (feature-config.NORMALIZATION[key][1])/
                                                    (config.NORMALIZATION[key][2]-config.NORMALIZATION[key][1])
                                                    ,
                                                    [batch_size,self.kernel_size,self.kernel_size,1])
                                          for key, feature in parsed_features[0].items() if key in context_names]
                                          + [last_binary_forest_mask], axis=-1)
                        

                else:
                    sequence_features = tf.concat([tf.reshape(feature,[batch_size,self.years,self.kernel_size,self.kernel_size,1])
                                          for key, feature in parsed_features[1].items() if key != 'year'], axis=-1)

                return {'in_seq':sequence_features, 'in_context':context_features} ,{'out':label}
            else:  
                # Parse all features
                #skip_features = ['forest_loss_cumu', 'forest_loss', 'year']
                skip_features = ['year']
                # normalization
                if self.normalization:
                    features = tf.concat([tf.reshape(
                                                    (feature-config.NORMALIZATION[key][1])/
                                                    (config.NORMALIZATION[key][2]-config.NORMALIZATION[key][1])
                                                    ,
                                                    [batch_size,self.years,self.kernel_size,self.kernel_size,1])
                                          for key, feature in parsed_features[1].items() if key not in skip_features], axis=-1)
                else:
                    features = tf.concat([tf.reshape(feature,[batch_size,self.years,self.kernel_size,self.kernel_size,1])
                                          for key, feature in parsed_features[1].items() if key not in skip_features], axis=-1)

            if len(self.sample_weights):
                reduced_values = tf.reduce_sum(label, axis=[1,2,3]).numpy()
                sample_weights = tf.Variable([list(self.sample_weights <= value).index(False)/len(self.sample_weights)
                                              for value in reduced_values], shape=(32,))
                return features, label, sample_weights
            else:
                return {'in_seq':features, 'point_id':point_id, 'coords':coords},{'out':label}

        # Find tf records files
        tf_record_files = glob(f'{file_path}/*/*.tfrecord.gz')
        tf_record_files = [transform_path(file) for file in tf_record_files]

        # Load dataset files
        dataset = tf.data.Dataset.from_tensor_slices(tf_record_files)

        if dataset_type == 'train':
            # Random shuffle ordering of files
            dataset = dataset.shuffle(config.SHUFFLE_SIZE_FILES)

        # Create TFRecord dataset using interleaving (num_parallel_reads)
        dataset = tf.data.TFRecordDataset(dataset,
                                        compression_type='GZIP',
                                        num_parallel_reads=tf.data.experimental.AUTOTUNE
                                        )
        if dataset_type == 'train':
            dataset = dataset.shuffle(config.SHUFFLE_SIZE)   
        
        # Batch size
        dataset = dataset.batch(batch_size, drop_remainder = True)
        
        # Parse the data from tf_record format
        dataset = dataset.map(_parse)

        # Prefetch data
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def load_dataset(self, file_path, baseline, dataset_type):
        if not baseline:
            dataset = self.prepare_dataset(f'{file_path}/{dataset_type}', dataset_type)
        else:
            dataset = self.prepare_baseline_dataset(f'{file_path}/{dataset_type}', dataset_type, False)
        return dataset

    def get_raw_dataset(self, file_path, baseline, dataset_type):
        # Find tf records files
        tf_record_files = glob(f'{file_path}/{dataset_type}/*/*.tfrecord.gz')

        tf_record_files = [transform_path(file) for file in tf_record_files]

        # Load dataset files
        dataset = tf.data.Dataset.from_tensor_slices(tf_record_files)

        # Create TFRecord dataset using interleaving (num_parallel_reads)
        dataset = tf.data.TFRecordDataset(dataset,
                                        compression_type='GZIP',
                                        num_parallel_reads=tf.data.experimental.AUTOTUNE
                                        )
        _parse = self.get_parse_function(baseline = baseline, check_dataset=True)

        if dataset_type in ['test', 'val']:
            # Batch the data
            dataset = dataset.batch(self.val_test_batch_size, drop_remainder = False) # ,  drop_remainder = True)
        else:
            dataset = dataset.batch(self.train_batch_size, drop_remainder = False) # ,  drop_remainder = True) 

        # Parse the data from tf_record format
        dataset = dataset.map(_parse)
        return dataset

    def get_datasets(self):
        datasets = {}
        for dataset_type in config.DATASET_TYPES:
            if self.check_dataset:
                datasets[dataset_type] = self.get_raw_dataset(self.file_path, self.baseline, dataset_type)
            else:
                datasets[dataset_type] = self.load_dataset(self.file_path, self.baseline, dataset_type) 
        return datasets

def main(file_path, baseline = False, check_dataset = False, batch_size = config.BATCH_SIZE):
    # Init helper
    load_dataset_helper = LoadDatasetHelper(file_path, baseline, check_dataset, batch_size)

    # Get datasets
    datasets = load_dataset_helper.get_datasets()

    for dataset_type, dataset in datasets.items():
        point_ids = []
        point_ids_concat = []
        for i, data in tqdm(enumerate(dataset)):
            point_id = data['point_id'].numpy()[0].decode("utf-8")
            point_ids.append(point_id)
            if 'pix_id' in data:
                pix_id = data['pix_id'].numpy()[0].decode("utf-8")
                point_ids_concat.append(f'{point_id}_{pix_id}')

        if args.baseline:
            print(f'{dataset_type}: {i+1} samples found, {len(set(point_ids))} with unique point id')
        else:
            print(f'{dataset_type}: {i+1} samples found, {len(set(point_ids))} unique point ids, {len(set(point_ids_concat))} unique pixels')


if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser(description='Process sampled patches.')

    # Input directory
    parser.add_argument('--file_path', default=None)

    # Process mode
    parser.add_argument('--baseline', action="store_true", default=False)

    # option to check dataset
    parser.add_argument('--check_dataset', action="store_true", default=True)

    # Evaluate argparse object
    args = parser.parse_args()

    #main(args.file_path, args.baseline, args.check_dataset)


    load_dataset_helper = LoadDatasetHelper(file_path='data/datasets/2013_2019/256px_seg',
                                        baseline=False,
                                        check_dataset = False,
                                        train_batch_size = 32,
                                        val_test_batch_size=32,
                                        mode = '256px_cum_loss_only'
                                        )
    load_dataset_helper.sample_weights = []
    datasets = load_dataset_helper.get_datasets()