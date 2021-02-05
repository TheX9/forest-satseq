import config
import logging_config

import argparse
import time
import threading
import os
from tqdm import tqdm

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
    def __init__(self, name = None, threads = 4):
        # Variable that keeps track of previous queries to avoid doing the same query over and over again
        self.queries = {}
        # Files that are already created for that run
        self.complete_file_list = None
        self.active_threads = {}
        self.allowed_threads = threads
        # Create logger
        logging_config.create_logger(f'download_dataset_{name}', config.LOGGING_PATH, True)

    def query_api_get_list(self, query):
        if query in self.queries:
            res = self.queries[query]
        else:
            res = GoogleDriveAPIHelper.drive.ListFile({"q": query}).GetList()
            self.queries[query] = res # save query for next use
        return res

    def get_sub_dirs(self, folder_id):
        query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        drive_folders = self.query_api_get_list(query)
        return drive_folders
       
    def download_sub_folder(self, folder_id, output_dir):
        sub_dirs = self.get_sub_dirs(folder_id)
        total = len(sub_dirs)
        self.progress_bar = tqdm(total=total)
        for sub_dir in sub_dirs:
            while len(self.active_threads) > self.allowed_threads: # max allowed threads
                time.sleep(2)

            # Create thread
            thr = threading.Thread(target=self.process_sub_dir, args=(sub_dir, output_dir))
            # Start thread
            thr.start() 

            self.active_threads[sub_dir['id']] = thr


    def process_sub_dir(self, sub_dir, output_dir):
        sub_dir_id = sub_dir['id']
        sub_dir_title = sub_dir['title']
        # Only get files
        query = f"'{sub_dir_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
        sub_dir_files = self.query_api_get_list(query)

        if not sub_dir_files:
            logging_config.logger.info(f'No files found for {sub_dir_title}.')
        # Loop over files
        for sub_dir_file in sub_dir_files:
            sub_dir_file_title = sub_dir_file['title']
            file_dir = f'{output_dir}/{sub_dir_title}'
            logging_config.logger.info(f'Process file {sub_dir_file_title} finished.')
            # Check if path exists
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)

            file_path = f'{file_dir}/{sub_dir_file_title}'
            # Check if file already exists
            if not os.path.exists(file_path):
                drive_file = GoogleDriveAPIHelper.drive.CreateFile({'id': sub_dir_file['id']})
                drive_file.GetContentFile(file_path)

        # Del task
        del self.active_threads[sub_dir_id]
        # Update progress bar
        self.progress_bar.update(1)
        logging_config.logger.info(f'{sub_dir_title} finished.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download dataset.')

     # name
    parser.add_argument('--name', type=str, default=None)

    # folder id
    parser.add_argument('--folder_id', type=str)

    # output dir
    parser.add_argument('--output_dir', type=str)

    # allowed threads
    parser.add_argument('--threads', type=int, default = 4)

    # parse arguments
    args = parser.parse_args()

    # initialize google drive helper
    google_drive_helper = GoogleDriveAPIHelper(args.name, args.threads)

    # Initialize GoogleDriveFile instance with file id.
    google_drive_helper.download_sub_folder(args.folder_id, args.output_dir)