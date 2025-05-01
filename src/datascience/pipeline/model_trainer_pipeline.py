from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.model_trainer import ModelTrainer
from src.datascience import logger


STAGE_NAME = "Model training Stage" 

class ModelTrainingPipeline:
    def __init__(self):
        pass 

    def initiate_model_training(self) :
        model_names = ["ElasticNet", "RandomForestRegressor"]
        accuracy = float('-inf')
        for model_name in model_names:
            try:
                config = ConfigurationManager()
                model_trainer_config = config.get_model_trainer_config(model_name)
                model_trainer = ModelTrainer(config=model_trainer_config)
                accuracy = model_trainer.train(model_name,accuracy)
            except Exception as e:
                raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.initiate_model_training()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e