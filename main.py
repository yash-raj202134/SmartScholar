from utils.logging import logger
from GetData.download import DataIngestion
import pandas as pd # type: ignore
from src.preprocessing import preprocess
from src.vectorization import vectorization
from src.model_trainer import train_model,plot_loss


# data = DataIngestion()
# file_path = data.download_datasets()
# data.extract_datasets(file_path)

arxiv_data = pd.read_csv("dataset/arxiv_data_210930-054931.csv")

train, validation, test, vocab_size, lookup = preprocess(arxiv_data = arxiv_data)

train,validation,test = vectorization(train,test,validation,vocabulary_size=vocab_size)

history = train_model(train_dataset=train,validation_dataset=validation,lookup=lookup)

plot_loss(history=history,item="loss",save_path="models")
plot_loss(history=history,item="binary_accuracy",save_path="models")

