#! /usr/bin/env python
import sys
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.keras.layers import TextVectorization
from IPython import embed
import pandas as pd
from datasets import load_dataset

# SELECT WHICH MODEL YOU WISH TO RUN:
from cnn_lstm import CNN_LSTM   # OPTION 0
from lstm_cnn import LSTM_CNN   # OPTION 1
from cnn import CNN             # OPTION 2
from lstm import LSTM           # OPTION 3
MODEL_TO_RUN = 0

import batchgen

# Parameters
# ==================================================
dev_size = .10
embedding_dim = 32
max_seq_length = 70
filter_sizes = [3, 4, 5]
num_filters = 32
dropout_prob = 0.5
l2_reg_lambda = 0.0
batch_size = 128
num_epochs = 10
evaluate_every = 100
checkpoint_every = 100000
num_checkpoints = 1
allow_soft_placement = True
log_device_placement = False

# Data Preparation
# ==================================================
# Cargar el dataset
dataset = load_dataset("glue", "sst2", split="train")

# Convertir a DataFrame con solo las columnas necesarias
df = dataset.to_pandas()
df = df[["sentence", "label"]]
df.columns = ["text", "label"]  

# Guardar como CSV
df.to_csv("sst2.csv", index=False)
# Load data
print("Loading data...")
x_text, y = batchgen.get_dataset_singlefile("sst2.csv", limit= 200000)


# Build vocabulary with random embeddings (no GloVe)
max_document_length = max([len(x.split(" ")) for x in x_text])
vectorizer = TextVectorization(
    max_tokens=10000,
    output_sequence_length=max_document_length
)
vectorizer.adapt(x_text)
x = vectorizer(np.array(x_text)).numpy()
vocab = vectorizer.get_vocabulary()
vocab_size = len(vocab)

# Random shuffle
np.random.seed(42)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Train/dev split
dev_sample_index = -1 * int(dev_size * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(vocab_size))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training
# ==================================================
with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
    if MODEL_TO_RUN == 0:
        model = CNN_LSTM(x_train.shape[1], y_train.shape[1], vocab_size, embedding_dim, filter_sizes, num_filters, l2_reg_lambda)
    elif MODEL_TO_RUN == 1:
        model = LSTM_CNN(x_train.shape[1], y_train.shape[1], vocab_size, embedding_dim, filter_sizes, num_filters, l2_reg_lambda)
    elif MODEL_TO_RUN == 2:
        model = CNN(x_train.shape[1], y_train.shape[1], vocab_size, embedding_dim, filter_sizes, num_filters, l2_reg_lambda)
    elif MODEL_TO_RUN == 3:
        model = LSTM(x_train.shape[1], y_train.shape[1], vocab_size, embedding_dim)
    else:
        print("PLEASE CHOOSE A VALID MODEL!\n0 = CNN_LSTM\n1 = LSTM_CNN\n2 = CNN\n3 = LSTM\n")
        exit()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint_dir = os.path.join("checkpoints", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "model.weights.h5"),
        save_weights_only=True,
        save_best_only=True
    )

    history = model.fit(
    x_train,
    y_train,
    validation_data=(x_dev, y_dev),
    batch_size=batch_size,
    epochs=num_epochs,
    callbacks=[tensorboard_callback, checkpoint_callback]
)


    print("\nEvaluation on Dev Set:")
    model.evaluate(x_dev, y_dev)


# Guardar el modelo completo (arquitectura + pesos + optimizador)
model_save_path = os.path.join("saved_models", "cnn_lstm_model.keras")
os.makedirs("saved_models", exist_ok=True)
model.save(model_save_path)  # Guarda todo en un solo archivo

# Guardar el vectorizador (importante para preprocesar nuevas frases)
import pickle
vectorizer_save_path = os.path.join("saved_models", "vectorizer.pkl")
with open(vectorizer_save_path, 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"Modelo guardado en {model_save_path}")
print(f"Vectorizador guardado en {vectorizer_save_path}")