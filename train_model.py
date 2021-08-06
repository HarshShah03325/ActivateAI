from settings import Settings
from dataset import Dataset
from model import model, train_model


settings = Settings()

dataset = Dataset(settings)

dataset.load_dataset()

model = model(settings)

train_model(model,dataset)
print("Model trained!")


