from dataset import Dataset
from settings import Settings

settings = Settings()
dataset = Dataset(settings)

dataset.create_training_data(30)
print('------------------- TRAINING DATA CREATED ------------------')