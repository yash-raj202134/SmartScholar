from utils.logging import logger

from GetData.download import DataIngestion


data = DataIngestion()
file_path = data.download_datasets()
data.extract_datasets(file_path)