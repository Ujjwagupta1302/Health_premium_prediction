import os
import urllib.request as request
from src.datascience import logger
import zipfile
from src.datascience.entity.config_entity import (DataIngestionconfig, DataValidationConfig)
import pandas as pd


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation of Columns status: {validation_status}\n")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation of Columns status: {validation_status}\n")

            return validation_status
        
        except Exception as e:
            raise e



    def validate_column_dtypes(self) -> bool:
        try:
            data = pd.read_csv(self.config.unzip_data_dir)
            validation_status = True

            for col, expected_dtype in self.config.all_schema.items():
                if col in data.columns:
                    if str(data[col].dtype) != expected_dtype:
                        validation_status = False
                        logger.warning(f"Column {col} has incorrect dtype: found {data[col].dtype}, expected {expected_dtype}")

            with open(self.config.STATUS_FILE, 'a') as f:
                f.write(f"Column Dtype Validation Status: {validation_status}\n")

            logger.info("Dtype validation complete.")
            return validation_status

        except Exception as e:
            raise e

    def check_null_values(self) -> bool:
        try:
            data = pd.read_csv(self.config.unzip_data_dir)
            null_summary = data.isnull().sum().to_dict()
            has_nulls = any(val > 0 for val in null_summary.values())

            with open(self.config.STATUS_FILE, 'a') as f:
                f.write(f"Null Value Check: {'Fail' if has_nulls else 'Pass'}\n")
                f.write(f"Nulls Summary: {null_summary}\n")

            logger.info("Null value check complete.")
            return True

        except Exception as e:
            raise e

    def check_duplicates(self) -> bool:
        try:
            data = pd.read_csv(self.config.unzip_data_dir)
            num_duplicates = data.duplicated().sum()
            has_duplicates = num_duplicates > 0

            with open(self.config.STATUS_FILE, 'a') as f:
                f.write(f"Duplicate Check: {'Fail' if has_duplicates else 'Pass'} ({num_duplicates} duplicates)\n")

            logger.info("Duplicate check complete.")
            return not has_duplicates

        except Exception as e:
            raise e
