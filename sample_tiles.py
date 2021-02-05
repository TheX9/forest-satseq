import config
import process_statistic_sample

import ee
ee.Initialize()
from tqdm import tqdm

import math
import numpy as np
import datetime
import os 
import time
import geopandas as gpd
import itertools
import logging_config
import argparse
import threading

if os.name == 'nt':
    import win32api
    import pywintypes
    
    def path_exists(path):
        dir_name = os.path.dirname(path)
        base_name = os.path.basename(path)
        try:
            dir_name = win32api.GetShortPathName(dir_name)
        except pywintypes.error:
            dir_name = dir_name
        path = f'{dir_name}/{base_name}'
        return os.path.exists(path)
else:
    def path_exists(path):
        return os.path.exists(path)

## Google Drive File Handling

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

class GoogleDriveAPIHelper:
    """This class is supposed to help handling the GoogleDrive API connections.
       It is assumed that during runtime the relevant folder structure of Google Drive does not change due synchronous nature of the script.
       Thus queries only have to be done once and are saved to increase performance.
    """
    # Static variables
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.
    drive = GoogleDrive(gauth)
    def __init__(self):
        # Variable that keeps track of previous queries to avoid doing the same query over and over again
        self.queries = {}
        # Files that are already created for that run
        self.complete_file_list = None

    def query_api_get_list(self, query):
        if query in self.queries:
            res = self.queries[query]
        else:
            res = GoogleDriveAPIHelper.drive.ListFile({"q": query}).GetList()
            self.queries[query] = res # save query for next use
        return res
    
    def add_new_folder_to_query(self, query, up_folder): 
        if query in self.queries:
            self.queries[query].append(up_folder)

    def get_sub_dirs(self, folder_path, root_folder_id = config.DRIVE_DATA_FILE_PATH_ID):
        folder_path = os.path.normpath(folder_path)
        check_folders = folder_path.split(os.sep)
        folder_found = True
        for check_folder in check_folders:
            if folder_found:
                folder_found = False
                query = f"'{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                drive_folders = self.query_api_get_list(query) # TODO: Check if working
            
                for drive_folder in drive_folders:
                    if drive_folder['title'] == check_folder:
                        root_folder_id = drive_folder['id']
                        folder_found = True

        if folder_found:
            query = f"'{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            drive_folders = self.query_api_get_list(query) # TODO: Check if working
            return drive_folders
        else:
            return []


    def create_sub_dirs_if_not_existing(self, folder_path, root_folder_id):
        folder_path = os.path.normpath(folder_path)
        check_folders = folder_path.split(os.sep)

        folder_found = True
        for check_folder in check_folders:
            if folder_found:
                folder_found = False
                query = f"'{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                drive_folders = self.query_api_get_list(query) # TODO: Check if working
                for drive_folder in drive_folders:
                    if drive_folder['title'] == check_folder:
                        root_folder_id = drive_folder['id']
                        folder_found = True

            if not folder_found:
                # Delete query where folder was not found
                up_folder = GoogleDriveAPIHelper.drive.CreateFile({'title' : check_folder, 'mimeType' : 'application/vnd.google-apps.folder', 'parents':[{'id':root_folder_id}]})
                up_folder.Upload()
                # Add folder to query
                self.add_new_folder_to_query(query, up_folder)
                root_folder_id = up_folder['id']
                
    def check_if_file_exists(self, file_path, root_folder_id):
        file_path = os.path.normpath(file_path)
        folder_path, file_name = os.path.split(file_path)
        check_folders = folder_path.split(os.sep)
        
        check_folders[:-1]

        folder_found = True
        for check_folder in check_folders:
            if folder_found:
                folder_found = False
                drive_folders = GoogleDriveAPIHelper.drive.ListFile({"q": f"'{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
                for drive_folder in drive_folders:
                    if drive_folder['title'] == check_folder:
                        root_folder_id = drive_folder['id']
                        folder_found = True
            else:
                return False

        drive_files = GoogleDriveAPIHelper.drive.ListFile({"q": f"'{root_folder_id}' in parents and trashed=false"}).GetList()
        for file in drive_files:
            if file['title'] == file_name:
                return True
        return False
    
    def check_if_task_is_already_done(self, task_folder, task_output_file_name):
        if self.complete_file_list == None:
            self.complete_file_list = []
            # Init complete file_task_list
            sub_folders = self.get_sub_dirs(task_folder)
            if sub_folders:
                sub_folders_ids = [sub_folder['id'] for sub_folder in sub_folders]
                split_amounts = math.ceil(len(sub_folders_ids)/100)
                sub_folders_ids_splits = np.array_split(np.array(sub_folders_ids), split_amounts)

                for sub_folders_ids in sub_folders_ids_splits:
                    folder_filter =f"('{sub_folders_ids[0]}' in parents"
                    for sub_folder in sub_folders_ids[1:]:
                        folder_filter = f"{folder_filter} or '{sub_folder}' in parents"
                    query = {'q': f"{folder_filter}) and trashed=false"}
                    file_list = GoogleDriveAPIHelper.drive.ListFile(query).GetList()
                    file_list = [found_file['title'] for found_file in file_list]

                    self.complete_file_list += file_list
        
        if task_output_file_name in self.complete_file_list:
            return True
        else:
            return False

def _create_point_collection(gdf):
    return ee.FeatureCollection(list(map(lambda data: ee.Feature(ee.Geometry.Point(data[1].geometry.centroid.coords[:][0], proj=f'EPSG:{gdf.crs.to_epsg()}')).set('point_id',data[1].FID), gdf.iterrows())))

class TaskHandler:
    def __init__(self):
        self.tasks_active = {}
        self.tasks_finished = []
        self.mult_task_allowance = config.MULT_TASK_ALLOWANCE
        self.task_count = 0
        self.avg_task_time = datetime.timedelta(0)
        self.google_drive_helper = GoogleDriveAPIHelper()
        self.last_task_started = None

        self.tasks_starting_threads = {}

    def wait_for_thread_to_be_finished(self):
        logging_config.logger.info(f'Wait for starting threads to be finished')
        while True:
            if len(self.tasks_starting_threads) < 5:
                break
            time.sleep(1)         
        logging_config.logger.info(f'Waiting for thread finished')

    def start_new_task(self, task, task_desc, folder, sub_dir, task_output_file_name, task_dict):
        
        finished_tasks = False
        last_task_times = None
        avg_task_time = None
        
        if self.google_drive_helper.check_if_task_is_already_done(folder, task_output_file_name):
            logging_config.logger.info(f'skip because task {task_desc} already exists.')
            return True, None, None
        else:
            logging_config.logger.info(f'Create task folder, if not already created {task_desc}')
            # Create task folder, if not already created
            self.google_drive_helper.create_sub_dirs_if_not_existing(f'{folder}/{sub_dir}', config.DRIVE_DATA_FILE_PATH_ID)
            logging_config.logger.info(f'Task folder created {task_desc}')

            # Keep maximum threads running at the same time
            self.wait_for_thread_to_be_finished()

            if len(self.tasks_active) >= self.mult_task_allowance:
                logging_config.logger.info(f'Wait until tasks are finished')
                # Wait until the first task in the list is finished
                finished_tasks, last_task_times, avg_task_time  = self.wait_for_next_task_to_be_finished()
                logging_config.logger.info(f'Waiting finished')

            # Do not wait if there is no active task
            if not len(self.tasks_active):
                sleep_time = 0
            if self.last_task_started:
                # Wait 0.5 seconds below the average task time, but at least 1 seconds, max 10 seconds
                next_task_start_time = self.last_task_started + (self.avg_task_time - datetime.timedelta(seconds = 0.5))
                sleep_time = max((next_task_start_time - datetime.datetime.now()).total_seconds(), 5)
                logging_config.logger.info(f'wait for {sleep_time}')
            else:
                sleep_time = 10
                time.sleep(sleep_time)

            # Wait before starting new task
            time.sleep(sleep_time) # TODO: the problem lays here to many started tasks/requests - increase time between task starts or reduce allowed parallel tasks

            self.start_task_thread_wrapper(task, task_desc, task_dict)

            return finished_tasks, last_task_times, avg_task_time 
    
    def start_task_thread_wrapper(self, task, task_desc, task_dict):
         # Start new task
        logging_config.logger.info(f'start task thread {task_desc}')
        # Create thread
        thr = threading.Thread(target=self.start_task, args=(task, task_desc, task_dict), kwargs={})
        # Start thread
        thr.start() 

        self.tasks_starting_threads[task_desc] = thr
        logging_config.logger.info(f'task_thread started {task_desc}')

    def start_task(self, task, task_desc, task_dict):
        # Start task
        task.start()
        # Log that task is created
        logging_config.logger.info(f'task started  {task_desc}')
        # Set last task started property
        self.last_task_started = datetime.datetime.now()
        task_dict['start_time'] = datetime.datetime.now() 
        # Append to active tasks
        self.tasks_active[task_desc] = task_dict
        # Remove task starting thread
        del self.tasks_starting_threads[task_desc]

    def active(self):
        if len(self.tasks_active):
            return True
        else:
            return False

    def create_task(self, sample, task_desc, folder_path, file_name_prefix, selectors):

        # Start task to download sampled image
        task = ee.batch.Export.table.toDrive(
            collection = sample,
            description = task_desc,
            folder = folder_path,
            fileNamePrefix = file_name_prefix,
            fileFormat = 'TFRecord',
            selectors = selectors
        )
        # Task output file name
        task_output_file_name = f'{file_name_prefix}.tfrecord.gz'
        
        return task, task_output_file_name

    # LEGACY code
    # def start_and_wait_for_task(self, task, task_desc, task_output_file_name):
        
    #     t1 = datetime.datetime.now()
    #     if path_exists(task_output_file_name):
    #         #print(f'skip because task {task_desc} already exists')
    #         logging_config.logger.info(f'skip because task {task_desc} already exists.')
    #     else:
    #         task.start()
            
    #         # Wait until task is finished
    #         while task.active():
    #             time.sleep(5)
    #         else:
    #             status = task.status()
    #             if status['state'] != 'COMPLETED':
    #                 #print(status)
    #                 logging_config.logger.error(f'{task_desc} not finished')
    #                 logging_config.logger.error(f'{status}')
    #                 #raise(RuntimeError(f'Task {task_desc} not finished.'))
    #             else:
    #                 logging_config.logger.info(f'{task_desc} finished.')

    #     task_time = (datetime.datetime.now() - t1)

    #     return task_time

    def wait_for_next_task_to_be_finished(self):
        #TODO: diff time calc
        # Wait until the first task in the list is finished
        current_finished_tasks = len(self.tasks_finished)
        task_times = []
        while True and self.tasks_active.values():
            for task_value in list(self.tasks_active.values())[:min(self.mult_task_allowance, 20)]: # 20 max TODO: check
                task = task_value['task']
                task_desc = task_value['task_desc']
                task_start_time = task_value['start_time']

                if not(task.active()):
                    status = task.status()
                    if status['state'] == 'FAILED' and task_value['task_desc'] in self.tasks_active:
                        #print(status)
                        logging_config.logger.error(f'{task_desc} not finished')
                        logging_config.logger.error(f'{status}')
                        # Delete task from task list
                        del self.tasks_active[task_value['task_desc']]
                        #raise(RuntimeError(f'Task {task_desc} not finished.'))
                        # Retry task with halved batch if 'Image.reduceRegions' error occured
                        if status['error_message'] == 'Image.reduceRegions: Computed value is too large.' or status['error_message']=='User memory limit exceeded.':
                            # Split batch in sub batches
                            batches = np.array_split(task_value['batch'], 2)
                            for i, batch in enumerate(batches):
                                batch_points = _create_point_collection(batch)
                                arrays = task_value['arrays']
                                # Sample image from batch points
                                sample = arrays.sampleRegions(collection = batch_points, 
                                                            scale = config.SCALE,
                                                            geometries = True,
                                                            )

                                file_name_prefix =  f'{task_value["file_name_prefix"]}_split_{i}'
                                task_desc = f'retry_{task_value["task_desc"]}:_split_{i}'

                                # Create task dict
                                task_dict = task_value.copy()

                                # Modify task dict
                                task_dict['file_name_prefix'] = file_name_prefix 
                                task_dict['task_desc'] = task_desc
                                task_dict['batch'] = batch

                                # Create task
                                task, task_output_file_name = self.create_task(sample, task_desc, task_value['short_file_path'], file_name_prefix, task_value['selectors'])
                                task_dict['task'] = task
                                # Start task with task handler
                                finished_tasks, task_time, avg_task_time = self.start_new_task(task, task_desc, task_value['folder_prefix'], task_value['short_file_path'], task_output_file_name, task_dict)
                    else:
                        if status['state'] != 'COMPLETED':
                            logging_config.logger.error(f'{task_desc} not finished')
                            logging_config.logger.error(f'{status}')
                        else:
                            logging_config.logger.info(f'{task_desc} finished.')
                    
                        # Add task to finished tasks
                        self.tasks_finished.append(task)

                        # Calc task_time
                        last_task_start_time = list(self.tasks_active.values())[-1]['start_time']
                        #task_time = (datetime.datetime.now() - first_task_start_time)
                        task_time = (datetime.datetime.now() - last_task_start_time)
                        task_times.append(task_time)
                        if task_value['task_desc'] in self.tasks_active:
                            # Remove task from task_list
                            del self.tasks_active[task_value['task_desc']]
                        
                        # Calculate average task time and set progress
                        self.avg_task_time = (self.avg_task_time * self.task_count + task_time)/(self.task_count+1)
                        self.task_count += 1

            newly_finished_tasks = len(self.tasks_finished) - current_finished_tasks
            if newly_finished_tasks:
                return newly_finished_tasks, np.array(task_times).mean(), self.avg_task_time

            time.sleep(5)

        return 0, datetime.timedelta(0), self.avg_task_time

        #while first_task.active():
        #    time.sleep(5)
        #else:
        #    status = first_task.status()
        #    if status['state'] != 'COMPLETED':
        #        #print(status)
        #        logging_config.logger.error(f'{first_task_desc} not finished')
        #        logging_config.logger.error(f'{status}')
        #        #raise(RuntimeError(f'Task {first_task_desc} not finished.'))
        #    else: 
        #        logging_config.logger.info(f'{first_task_desc} finished.')
        
# LEGACY Code
# def create_batch_folders(batch_count, folder, sub_folder_prefix):
#     progress_bar_folder_batches = tqdm(range(batch_count),
#                                        total = batch_count, 
#                                        desc = 'Create folder',
#                                        leave = False)

#     # Google drive helper
#     google_drive_helper = GoogleDriveAPIHelper()

#     # Get already created folders
#     existing_sub_dirs = google_drive_helper.get_sub_dirs(folder)
#     existing_sub_dirs = [sub_dir['title'] for sub_dir in existing_sub_dirs]

#     for batch_i in progress_bar_folder_batches:
#         # Check if path exists - and create for all 
#         sub_dir = f'{sub_folder_prefix}_{batch_i}'
#         file_path = f'{folder}\{sub_dir}'
#         if sub_dir not in existing_sub_dirs:
#             pass
#             #google_drive_helper.create_sub_dirs_if_not_existing(file_path, config.DRIVE_DATA_FILE_PATH_ID)
#     progress_bar_folder_batches.close()
#     return folder


class SamplingHelper:
    def __init__(self,
                patch_amount = 1000,
                sample_regions = False,
                sample_concatenated_years = True,
                seed = 0, statistic_sample = False, baseline = False,
                gdf_points = gpd.GeoDataFrame(),
                years = None,
                time_slices = None,
                ecoregion = False):
        
        # Initialize class variables
        self.patch_amount = patch_amount
        self.sample_regions = sample_regions
        self.sample_concatenated_years = sample_concatenated_years
        self.seed = seed
        self.statistic_sample = statistic_sample
        self.baseline = baseline
        self.gdf_points = gdf_points 
        self.years = years
        self.time_slices = time_slices
        self.amazon_ecoregion = ecoregion

        if self.amazon_ecoregion:
            self.sample_shapefile = config.AMAZON_ECOREGION
        else:
            self.sample_shapefile = config.DEFOREST_HOTSPOTS

        # Init task handler
        self.task_handler = TaskHandler()

        # Generate random points if no points are given
        if self.gdf_points.empty and self.sample_regions:    
            # Generate random points
            file_path = self.generate_random_points(self.patch_amount, self.seed) 

            # Read random points
            gdf_all = gpd.read_file(file_path)
            
            # Create FID column
            gdf_all = gdf_all.reset_index().rename(columns={'index':'FID'})

            # Incorporate seed in FID column
            gdf_all['FID'] = gdf_all['FID'].astype(str) + f'_{seed}'

            # File_dir, file name
            file_path = os.path.normpath(file_path)
            file_dir, file_name = os.path.split(file_path)
            file_name = f'{file_name.replace(".shp","")}_stats_{statistic_sample}.shp'
            
            # Get new file path
            self.rand_buffer_file_path = f'{file_dir}/{file_name}'

            # Save corresponding shapefile
            gdf_all.to_file(self.rand_buffer_file_path)

            self.gdf_points = gdf_all
            self.patch_amount = len(self.gdf_points)
        elif not self.gdf_points.empty:
            self.patch_amount = len(self.gdf_points)
    
    def _create_point_collection(self, gdf):
        return ee.FeatureCollection(list(map(lambda data: ee.Feature(ee.Geometry.Point(data[1].geometry.centroid.coords[:][0], proj=f'EPSG:{gdf.crs.to_epsg()}')).set('point_id',data[1].FID), gdf.iterrows())))

    def generate_baseline_features_neighborhood_array(self):
        # Define cost image, every pixel gets a cost of 1 attached
        cost = ee.Image(1).toByte()

        def _gen_feature_dict(image, band_name, complex_flag):
            feature = {}
            # Append image to feature stack
            feature['feature_image'] = image
            # Add selector
            feature['selector'] = band_name
            # Complex flag
            feature['complex_flag'] = complex_flag
            return feature

        # List with all features
        feature_list = []

        pop_years = config._get_closest_years(self.years[-1], config.POPULATION_DENSITY_YEARS)
        first_pop_year = pop_years[0]
        for year in pop_years:
            # Population density
            band_name = f'{config.POP_IMAGE_BAND}_{year}'
            pop_density_image = (config.POP_IMAGE
                                .filterDate(f'{year}-01-01', f'{year}-12-31')
                                .first()
                                .select(config.POP_IMAGE_BAND)
                                .rename(band_name))

            # Add to feature list
            feature_list.append(_gen_feature_dict(pop_density_image, band_name, False))

            # Only compute distance for year before prediction year
            if year == first_pop_year: 
                # Mask city centres (more than 100 inhabitants per square kilometer)
                city_centre_image = pop_density_image.gt(100).selfMask()

                # Band name
                band_name = f'{config.DIST_TO_CITY_CENTRE_BAND}_{year}'

                # Compute the cumulative cost to the next source, in this case city centre
                cost_to_city_centre = (cost.cumulativeCost(source=city_centre_image,
                                                           maxDistance=500 * 1000 # 100 kilometers
                                                           )
                                           .select('cumulative_cost')
                                           .rename(band_name))

                # Add to feature list
                feature_list.append(_gen_feature_dict(cost_to_city_centre, band_name, True))
 
        forest_loss_image = config.FOREST_LOSS_IMAGE

        # Add selector for forest loss and treecover
        for band in config.FOREST_BANDS:
            # Add to feature list
            feature_list.append(_gen_feature_dict(forest_loss_image, band, False))

        ## Distance to forest loss in the period of the first year in the range up to the second last year
        first_forest_loss_year = int(str(self.years[0])[2:])
        label_forest_loss_year = int(str(self.years[-1])[2:])
        
        forest_loss_source_image = (config.FOREST_LOSS_IMAGE_MASKED
                                    .select(config.FOREST_IMAGE_BAND)
                                    # Greater or equal than the first year
                                    .gte(first_forest_loss_year) 
                                    # Less than the last year
                                    .And(config.FOREST_LOSS_IMAGE_MASKED.select(config.FOREST_IMAGE_BAND)
                                    .lt(label_forest_loss_year)) 
                                    )

        forest_loss_source_image = forest_loss_source_image.selfMask()

        # Compute the cumulative cost to the next source, in this case lost forest
        band_name = config.DIST_TO_FOREST_BAND
        cost_to_forest_loss= (cost.cumulativeCost(source=forest_loss_source_image,
                                                    maxDistance=25 * 1000 # 100 kilometers
                                                    )
                                    .select('cumulative_cost')
                                    .rename(band_name))

        # Add to feature list
        feature_list.append(_gen_feature_dict(cost_to_forest_loss, band_name, True))

        ## Distance to roads        
        roads_sources = ee.Image().toByte().paint(config.ROADS_AMAZON, 1)
        # Mask the sources image with itself.
        roads_sources = roads_sources.updateMask(roads_sources)
        # Define band name
        band_name = config.DIST_TO_ROADS_BAND
        # Compute the cumulative cost to traverse
        dist_roads = (cost.cumulativeCost(source=roads_sources,
                                          maxDistance=250 * 1000 # 50 kilometers
                                          )
                           .select('cumulative_cost')
                           .rename(band_name))

        # Add to feature list
        feature_list.append(_gen_feature_dict(dist_roads, band_name, True))

        # Add elevation image to feature stack
        elevation_image = config.DIGITAL_ELEVATION_IMAGE
        # Add to feature list
        feature_list.append(_gen_feature_dict(config.DIGITAL_ELEVATION_IMAGE, config.ELEV_BAND, False))
        
        # Add slope image to feature stack
        slope_image = ee.Terrain.slope(elevation_image)
        # Add to feature list
        feature_list.append(_gen_feature_dict(slope_image, config.SLOPE_BAND, False))

        arrays_list = []
        selectors_list = []

        # Create several feature stacks for sampling
        features_per_stack = 3
        feature_stack_images = []
        selectors = []
        ii = 0
        # Get non-complex features
        for i, feature in enumerate(feature_list):
            if not feature['complex_flag']:
                # Add feature stack image
                feature_stack_images.append(feature['feature_image'])    
                # Add selector
                selectors.append(feature['selector'])
                ii+=1
                # Create stack and get next stack
                if (ii)%features_per_stack == 0 or (i+1) == len(feature_list):
                    # Create neighborhood array
                    arrays = self.create_neighborhood_array_from_feature_stack(feature_stack_images)
                    arrays_list.append(arrays)
                    # Append selectors
                    selectors_list.append(selectors + ['point_id']) # always select point id for identification
                    # Reset variables
                    feature_stack_images = []
                    arrays = []
                    selectors = []
                    ii = 0
            # Complex features are added directly
            else:
                # Create neighborhood array
                arrays = self.create_neighborhood_array_from_feature_stack(feature['feature_image'])
                arrays_list.append(arrays)
                # Append selectors
                selectors_list.append([feature['selector'], 'point_id']) # always select point id for identification
                # Reset
                arrays = []

        return arrays_list, selectors_list

    def create_neighborhood_array_from_feature_stack(self, feature_stack_images):
        # Create feature stack of images
        featureStack = ee.Image.cat([feature_stack_images]).float()
            
        # Create kernel     
        lst = ee.List.repeat(1, config.KERNEL_SIZE)
        lsts = ee.List.repeat(lst, config.KERNEL_SIZE)
        kernel = ee.Kernel.fixed(config.KERNEL_SIZE, config.KERNEL_SIZE, lsts)

        # Create neighborhood array
        arrays = featureStack.neighborhoodToArray(kernel)

        return arrays

    def generate_bands_for_statistics_neighborhood_array(self):

        # Prepare forest loss image
        forest_loss_image = config.FOREST_LOSS_IMAGE
        
        # Configure selectors
        selectors = ['lossyear', 'point_id']

        mask = None
        # Create mask for forest loss image, last year can be ignored, since it is the label year
        for year in self.years[:-1]:
            if year in range(2013,2020):
                image = config.get_landsat8_image(year)
            elif year in range(2001, 2013):
                image = config.get_landsat5_image(year)
            else:
                continue

            # Create mask
            if not(mask):
                mask = image.select(0).mask()
            else:
                mask = mask.multiply(image.select(0).mask())

        # Update forest_loss_image mask
        forest_loss_image = forest_loss_image.updateMask(mask)

        # Create feature stack of images
        featureStack = ee.Image.cat([forest_loss_image]).float()
            
        # Create kernel     
        lst = ee.List.repeat(1, config.KERNEL_SIZE)
        lsts = ee.List.repeat(lst, config.KERNEL_SIZE)
        kernel = ee.Kernel.fixed(config.KERNEL_SIZE, config.KERNEL_SIZE, lsts)

        # Create neighborhood array
        arrays = featureStack.neighborhoodToArray(kernel)

        return [arrays], [selectors]

    def generate_concat_bands_neighborhood_array(self):
            # Prepare time invariant images
            forest_loss_image = config.FOREST_LOSS_IMAGE
            digital_elevation_image = config.DIGITAL_ELEVATION_IMAGE
            
            # Loop over all time slices - time invariant bands and all sample years
            feature_stack_images = []
            selectors = config.PROPERTIES
            for time_slice in self.time_slices:
                if time_slice == config.TIME_INVARIANT:
                    # Append feature images for time invariant features
                    feature_stack_images.append(forest_loss_image.select(config.FOREST_BANDS))
                    feature_stack_images.append(digital_elevation_image.select(config.ELEV_BAND))

                    selectors = selectors + config.FOREST_BANDS + [config.ELEV_BAND]
                elif time_slice in range(2013,2020) and time_slice != self.years[-1]:
                    # Prepare image to sample from
                    image = config.get_landsat8_image(time_slice)
                    
                    new_bands = [f'{band}_{time_slice}' for band in config.BANDS]
                    feature_stack_images.append(image.select(config.BANDS).rename(new_bands))
                    selectors = selectors + new_bands
                
                elif time_slice in range(2000,2013) and time_slice != self.years[-1]:
                    # Prepare image to sample from
                    image = config.get_landsat5_image(time_slice)
                    
                    new_bands = [f'{band}_{time_slice}' for band in config.BANDS]
                    feature_stack_images.append(image.select(config.BANDS).rename(new_bands))
                    selectors = selectors + new_bands
                
            # Create feature stack of images
            featureStack = ee.Image.cat([
            feature_stack_images
            ]).float()
                
            # Create kernel     
            lst = ee.List.repeat(1, config.KERNEL_SIZE)
            lsts = ee.List.repeat(lst, config.KERNEL_SIZE)
            kernel = ee.Kernel.fixed(config.KERNEL_SIZE, config.KERNEL_SIZE, lsts)

            # Create neighborhood array
            arrays = featureStack.neighborhoodToArray(kernel)

            return [arrays], [selectors]

    def generate_yearly_bands_neighborhood_array(self, time_slice):
        if time_slice == config.TIME_INVARIANT:
                # Create feature stack of images for time invariant features
                featureStack = ee.Image.cat([
                config.FOREST_LOSS_IMAGE.select(config.FOREST_BANDS),
                config.DIGITAL_ELEVATION_IMAGE.select(config.ELEV_BAND)
                ]).float()
                selectors = config.FOREST_BANDS + [config.ELEV_BAND] + config.PROPERTIES
        elif time_slice in range(2013,2020):
            # Prepare image to sample from
            image = config.get_landsat8_image(time_slice)
            # Create feature stack of images
            featureStack = ee.Image.cat([
            image.select(config.BANDS)
            ]).float()
            selectors = config.BANDS + config.PROPERTIES    
        elif time_slice in range(2001,2013):
            # Prepare image to sample from - landsat 5 has data available for 2000 to 2013
            image = config.get_landsat5_image(time_slice)
            # Create feature stack of images
            featureStack = ee.Image.cat([
            image.select(config.BANDS)
            ]).float()
            selectors = config.BANDS + config.PROPERTIES
    
        # Create kernel     
        lst = ee.List.repeat(1, config.KERNEL_SIZE)
        lsts = ee.List.repeat(lst, config.KERNEL_SIZE)
        kernel = ee.Kernel.fixed(config.KERNEL_SIZE, config.KERNEL_SIZE, lsts)

        # Create neighborhood array
        arrays = featureStack.neighborhoodToArray(kernel)

        return [arrays], [selectors]


    def generate_random_points(self, n_rand_points, seed):

        # Intialize google earth engine
        import ee
        ee.Initialize()

        # File name
        sampling_area = self.sample_shapefile.get("system:id").getInfo().split('/')[-1]
        file_name = f'rand_buffer_gee_{sampling_area}_{n_rand_points}_seed_{seed}'

        # File path of random buffer shape file - TODO: Replace with relative name
        short_file_path = file_name
        rand_buffer_file_path = f'{config.SHAPEFILES_PATH}/{file_name}/'

        # Create folder if it does not exist yet
        if not(path_exists(rand_buffer_file_path)):
            os.makedirs(rand_buffer_file_path)

        rand_buffer_complete_file_path = f'{rand_buffer_file_path}/{file_name}.shp'

        # Check if buffer file was already generated
        if not(path_exists(rand_buffer_complete_file_path)):

            # Sample random points from google earth engine
            random_points = ee.FeatureCollection.randomPoints(self.sample_shapefile, n_rand_points, seed)

            # Random buffer
            random_buffer = random_points.map(lambda feature: feature.buffer(config.PATCH_SIZE_METERS/2).bounds())

            # Export random points to drive
            task = ee.batch.Export.table.toDrive(collection=random_buffer,
                                                description=f'Random hotspot buffer-{n_rand_points}_random_points_seed_{seed}',
                                                folder=short_file_path,
                                                fileNamePrefix=file_name,
                                                fileFormat='SHP')
            task.start()

            # Only continue, if status is finished
            while task.status()['state'] in ['READY', 'RUNNING']:
                print(task.status())
                time.sleep(10)
            else:
                print(task.status())
                if task.status()['state'] != 'COMPLETED':
                    raise(RuntimeError('Google Earth Engine task was not finished!'))
        

            # wait for the file to be generated
            time_out = 600
            t1 = datetime.datetime.now()
            while not(path_exists(rand_buffer_file_path)):
                time.sleep(5)
                elapsed_time = datetime.datetime.now() - t1
                if elapsed_time.seconds > time_out:
                    raise(TimeoutError('Shapefile was not synchronized yet. Time out error.'))

        return rand_buffer_complete_file_path

    def fetch_patches(self):
        files_per_patch = 1
        # If statistics sample is set, overwrite globally defined variables
        if self.statistic_sample:
            print('NOTE: increase features per patch limit to increase performance')
            folder_prefix = '_statistics_sample'
            features_per_patch = config.KERNEL_SIZE*config.KERNEL_SIZE*2.5 # 2.5 seems to work TODO: Check features per patch calc
        elif self.sample_concatenated_years:
            folder_prefix = '_concat_years'
            features_per_patch = (config.KERNEL_SIZE*config.KERNEL_SIZE*len(config.BANDS)*(len(self.years)-1)
                                  +len([config.FOREST_BANDS + [config.ELEV_BAND]])*config.KERNEL_SIZE*config.KERNEL_SIZE)/2
        else: 
            folder_prefix = '_yearly'
            files_per_patch = len(self.time_slices)
            features_per_patch = config.KERNEL_SIZE*config.KERNEL_SIZE*len(config.BANDS)+len(config.PROPERTIES)
        
        if self.amazon_ecoregion:
            folder_prefix = f'{folder_prefix}_amazon_ecoregion'

        # baseline features
        if self.baseline:
            folder_prefix = f'{folder_prefix}_baseline'

            # Calculate features per patch
            features_per_patch = (2*config.KERNEL_SIZE*config.KERNEL_SIZE  # Population density
                                  + config.KERNEL_SIZE*config.KERNEL_SIZE # distance to city centre
                                  + config.KERNEL_SIZE*config.KERNEL_SIZE # distance to road
                                  + 2*config.KERNEL_SIZE*config.KERNEL_SIZE # forest loss picture (lossyear and treecover)
                                  + config.KERNEL_SIZE*config.KERNEL_SIZE  # distance to forest
                                  + config.KERNEL_SIZE*config.KERNEL_SIZE  # elevation
                                  + config.KERNEL_SIZE*config.KERNEL_SIZE  # slope
                                  )   
            features_per_patch = (config.KERNEL_SIZE*config.KERNEL_SIZE)*6
        
        # Generate task split, that is needed to not reach the maximum feature export, supported by google earth engine
        points_per_task = math.floor(config.GEE_FEATURE_LIMIT/features_per_patch)

        # Calculate the amount of tasks needed
        task_amount = math.ceil(self.patch_amount/points_per_task)
        
        # If points are given
        if not(self.gdf_points.empty):
            total_patch_amount = files_per_patch*len(self.gdf_points)
            batches = np.array_split(self.gdf_points, task_amount)
            folder_prefix = f'satseq_sampleRegions{folder_prefix}'
        else:
            total_patch_amount = files_per_patch*self.patch_amount 
            batches = math.floor(self.patch_amount/points_per_task)*[points_per_task]
            if self.patch_amount%points_per_task:
                batches.append(self.patch_amount%points_per_task)
            folder_prefix = f'satseq_sample{folder_prefix}'

        # Create file paths
        sub_folder_prefix = f'{folder_prefix}_{self.seed}_{self.patch_amount}_{self.years[0]}_{self.years[-1]}'
        folder_prefix = f'{folder_prefix}/{self.years[0]}_{self.years[-1]}_patches_{self.patch_amount}/'
            
        # Init logger
        logging_config.create_logger(sub_folder_prefix, config.LOGGING_PATH, True)

        # Init statistics
        i = 0
        avg_task_time = datetime.timedelta(0)
        statistics_bar = tqdm(total = 0, bar_format='{desc}', position=0)
        task_bar = tqdm(total = 0, desc = 'total tasks', position=1)
        progress_bar_overall = tqdm(total = total_patch_amount, desc = 'total sampled patches', position=2)

        # Loop over all batches
        progress_bar_batches = tqdm(enumerate(batches),
                                    total = len(batches), 
                                    desc = f'{folder_prefix} batches',
                                    leave = False)

        for batch_i,batch in progress_bar_batches:
            
            # Set file path
            short_file_path = f'{sub_folder_prefix}_{batch_i}'   

            if self.sample_concatenated_years or self.statistic_sample or self.baseline:
                
                # If statistics neighborhood should be generated
                if self.statistic_sample:
                    arrays_list, selectors_list = self.generate_bands_for_statistics_neighborhood_array()
                elif self.baseline:
                    # Create baseline neighborhood
                    arrays_list, selectors_list = self.generate_baseline_features_neighborhood_array()
                else:
                    # Create neighborhood array with concatenated bands
                    arrays_list, selectors_list = self.generate_concat_bands_neighborhood_array()
                

                # Some tasks are to big to be completed in one run
                # Thats why arrays and selectors will be povided in list format to do it partwise

                task_parts = len(arrays_list)

                # Update progress bars with new totals
                if task_parts > 1:
                    progress_bar_overall.total = total_patch_amount*task_parts
                    progress_bar_overall.refresh()

                    task_bar.total = len(batches)*task_parts
                    task_bar.refresh()

                for i in range(task_parts):
                    arrays = arrays_list[i]
                    selectors = selectors_list[i]

                    if not(self.gdf_points.empty):
                        # Prepare batch points
                        batch_points = self._create_point_collection(batch)

                        # Get batch size
                        batch_size = len(batch)

                        # Sample image from batch points
                        sample = arrays.sampleRegions(
                            collection = batch_points, 
                            scale = config.SCALE,
                            geometries = True,
                            )
                        if task_parts > 1:
                            task_desc = f'seed_{self.seed}_patch_amount_{self.patch_amount}_batch_{batch_i+1}-{len(batches)}_part{i+1}_{task_parts}'
                            file_name_prefix = f'seed_{self.seed}_patch_amount_{self.patch_amount}_batch_{batch_i}_part_{i+1}'
                        else:
                            task_desc = f'seed_{self.seed}_patch_amount_{self.patch_amount}_batch_{batch_i+1}-{len(batches)}'
                            file_name_prefix = f'seed_{self.seed}_patch_amount_{self.patch_amount}_batch_{batch_i}'
                    else:
                        # Define batch size
                        batch_size = batch

                        # Sample image
                        sample = arrays.sample(
                                    region = self.sample_shapefile,
                                    scale = config.SCALE,
                                    numPixels = int(batch), 
                                    seed = self.seed + batch_i,
                                    geometries = True
                                    )

                        task_desc = f'seed_{self.seed + batch_i}_numPixels_{int(batch)}_batch_{batch_i+1}-{len(batches)}_part{i+1}_{task_parts}'
                        file_name_prefix = f'seed_{self.seed + batch_i}_numPixels_{int(batch)}_batch_{batch_i}_part_{i+1}'

                    # Create task dict, that holds all information about tasks
                    task_dict = {}

                    task_dict['task_desc'] = task_desc
                    task_dict['arrays'] = arrays
                    task_dict['selectors'] = selectors
                    task_dict['batch'] = batch
                    task_dict['folder_prefix'] = folder_prefix
                    task_dict['short_file_path'] = short_file_path
                    task_dict['file_name_prefix'] = file_name_prefix

                    # Create task
                    task, task_output_file_name = self.task_handler.create_task(sample, task_desc, short_file_path, file_name_prefix, selectors)
                    task_dict['task'] = task
                    # Start task with task handler
                    finished_tasks, task_time, avg_task_time = self.task_handler.start_new_task(task, task_desc, folder_prefix, short_file_path, task_output_file_name, task_dict)

                    # Update progessbars if tasks were finished
                    if finished_tasks:
                        task_bar.update(finished_tasks)
                        progress_bar_overall.update(batch_size*finished_tasks)
                        if task_time:
                            statistics_bar.set_description_str(f'Task Time - Avg.: {avg_task_time} - Last avg: {task_time}')              
                                    
        # Wait for all tasks to be finished
        while self.task_handler.active():
            finished_tasks, last_avg_task_time, avg_task_time = self.task_handler.wait_for_next_task_to_be_finished()
            avg_task_time = (avg_task_time * i + last_avg_task_time)/(i+1)
            statistics_bar.set_description_str(f'Task Time - Avg.: {avg_task_time} - Last Avg: {last_avg_task_time}')
            # TODO: wrong for last batch!
            progress_bar_overall.update(len(batch)*finished_tasks)
            
        # If statistics sample further processing should already take place, to filter sampled points
        if self.statistic_sample:
            process_statistic_sample.main(f'./{config.RELATIVE_DATA_FILE_PATH}/{folder_prefix}', self.years, self.rand_buffer_file_path)

def main(patch_amount = 1000,
                sample_regions = False,
                sample_concatenated_years = True,
                seed = 0, statistic_sample = False, baseline = False,
                gdf_points = gpd.GeoDataFrame(),
                years = None,
                time_slices = None,
                ecoregion = False):
    # Initalize sampling helper
    sampling_helper = SamplingHelper(patch_amount,
                                     sample_regions,
                                     sample_concatenated_years,
                                     seed,
                                     statistic_sample,
                                     baseline,
                                     gdf_points,
                                     years,
                                     time_slices,
                                     ecoregion)
    
    # Start export process
    sampling_helper.fetch_patches()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sample patches.')

    # seed
    parser.add_argument('--seed', type=int, dest='seed', default = 0)

    # Sample concatenated years
    parser.add_argument('--sample_concatenated', action="store_true", default=False)

    # Sample years
    parser.add_argument('--year_range',  nargs="+", default = [2013, 2020], type=int)
    
    # sample regions
    parser.add_argument('--sample_regions', action="store_true", default=False)

    # sample regions
    parser.add_argument('--sample_stats', action="store_true", default=False)

    # sample regions
    parser.add_argument('--sample_baseline', action="store_true", default=False)

    # patch amount
    parser.add_argument('--patch_amount', type=int, default=100)

    # shapefile
    parser.add_argument('--shapefile', default=None)

    # Sample amazon ecoregion
    parser.add_argument('--amazon_ecoregion', action="store_true", default=False)

    args = parser.parse_args()

    if args.shapefile:
        gdf_points = gpd.read_file(args.shapefile)
    else:
        gdf_points = gpd.GeoDataFrame()
    
    # Set years, that should be sampled
    time_slices = [config.TIME_INVARIANT] + list(range(*args.year_range))

    # Sample
    main(patch_amount = args.patch_amount,
         sample_regions = args.sample_regions,
         seed = args.seed,
         statistic_sample = args.sample_stats,
         baseline = args.sample_baseline,
         gdf_points = gdf_points,
         years = range(*args.year_range),
         time_slices = time_slices,
         ecoregion = args.amazon_ecoregion)

         # TODO: try except block to avoid interruptions, maybe raise exception,
         # when a task took more than 60 seconds to finish 