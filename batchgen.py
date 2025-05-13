import csv
import re
import random
import numpy as np

from IPython import embed

# Load Dataset from a single file with labels
def get_dataset_singlefile(filename, limit=None, randomize=True):
    import pandas as pd

    df = pd.read_csv(filename)

    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("El CSV debe tener columnas llamadas 'text' y 'label'")

    df = df[["label", "text"]]

    if randomize:
        df = df.sample(frac=1).reset_index(drop=True)

    if limit:
        df = df.head(limit)

    data_x = [clean_str(text) for text in df["text"]]
    data_y = [[0, 1] if label == 1 else [1, 0] for label in df["label"]]

    return data_x, np.array(data_y)

# Clean Dataset
def clean_str(string):
    string = re.sub(r":\)","emojihappy1",string)
    string = re.sub(r":P","emojihappy2",string)
    string = re.sub(r":p","emojihappy3",string)
    string = re.sub(r":>","emojihappy4",string)
    string = re.sub(r":3","emojihappy5",string)
    string = re.sub(r":D","emojihappy6",string)
    string = re.sub(r" XD ","emojihappy7",string)
    string = re.sub(r" <3 ","emojihappy8",string)

    string = re.sub(r":\(","emojisad9",string)
    string = re.sub(r":<","emojisad10",string)
    string = re.sub(r":<","emojisad11",string)
    string = re.sub(r">:\(","emojisad12",string)

    string = re.sub(r"(@)\w+","mentiontoken",string)
    string = re.sub(r"http(s)*:(\S)*","linktoken",string)
    string = re.sub(r"\\x(\S)*","",string)

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()

# Generate random batches
def gen_batch(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = [data[i] for i in shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
