from imageai.Prediction.Custom import ModelTraining
import os

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory(os.path.join(os.getcwd() , "cars"))
model_trainer.trainModel(
    num_objects=1,
    num_experiments=100,
    enhance_data=True,
    batch_size=32,
    show_network_summary=True
)