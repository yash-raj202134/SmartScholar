from utils.logging import logger
from GetData.download import DataIngestion
import pandas as pd
from src.preprocessing import preprocess


# data = DataIngestion()
# file_path = data.download_datasets()
# data.extract_datasets(file_path)

arxiv_data = pd.read_csv("dataset/arxiv_data_210930-054931.csv")

train, validation, test = preprocess(arxiv_data = arxiv_data)
