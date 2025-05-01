import os
import urllib.request as request
from src.datascience import logger
import zipfile
from src.datascience.entity.config_entity import (DataIngestionconfig)
import pandas as pd 

class DataIngestion:
    def __init__(self, config:DataIngestionconfig) :
        self.config = config 
    '''
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists")
    '''
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    def convert_to_csv(self):
        unzip_path = self.config.unzip_dir  # This is a directory
        for file in os.listdir(unzip_path):
            if file.endswith(".xlsx"):
                xlsx_file_path = os.path.join(unzip_path, file)
                csv_path = os.path.join(unzip_path, "data.csv")

                df = pd.read_excel(xlsx_file_path)
                df.to_csv(csv_path, index=False)

                logger.info(f"Converted '{xlsx_file_path}' to '{csv_path}'")
                break
            else:
                logger.info("No .xlsx file found; no conversion needed.")
 