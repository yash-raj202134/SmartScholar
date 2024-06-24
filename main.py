from utils.logging import logger

from GetData.download import DataIngestion
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

from sklearn.model_selection import train_test_split

from ast import literal_eval
# is used for safely evaluating strings containing Python literals or container displays
# (e.g., lists, dictionaries) to their corresponding Python objects.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



data = DataIngestion()
file_path = data.download_datasets()
data.extract_datasets(file_path)

arxiv_data = pd.read_csv("dataset/arxiv_data_210930-054931.csv")

