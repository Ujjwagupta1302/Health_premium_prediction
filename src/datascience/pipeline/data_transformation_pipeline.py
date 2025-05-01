from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.data_transformation import DataTransformation
from src.datascience import logger
from pathlib import Path 


STAGE_NAME = "Data Transformation Stage" 

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass 

    def initiate_data_transformation(self) :
        try:
            with open(Path("artifacts/data_validation/status.txt"),'r') as f:
                first_two_lines = [next(f).strip() for _ in range(2)]
                status1 = first_two_lines[0].split()[-1].lower()
                status2 = first_two_lines[1].split()[-1].lower()

                if status1=="true" and status2=="true":
                    config = ConfigurationManager()
                    data_transformation_config = config.get_data_transformation_config()
                    data_transformation = DataTransformation(config=data_transformation_config)
                    data_transformation.transforming_data()
                else :
                    raise Exception("Your data scheme is not valid") 
        except Exception as e:
            print(e) 
        

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.initiate_data_transformation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    