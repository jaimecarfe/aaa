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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc

# SELECT WHICH MODEL YOU WISH TO RUN:
from cnn_lstm import CNN_LSTM   # OPTION 0
MODEL_TO_RUN = 0

import batchgen

# Parameters
# ==================================================
dev_size = .10
test_size = .10
embedding_dim = 300  # Aumentado para GloVe
max_seq_length = 70
filter_sizes = [3, 4, 5]
num_filters = 64  # Aumentado
dropout_prob = 0.6  # Aumentado
l2_reg_lambda = 0.01  # Añadido
batch_size = 64
num_epochs = 10  # Aumentado
patience = 5  # Para EarlyStopping

# Data Preparation
# ==================================================
# Cargar y dividir datos
dataset = load_dataset("glue", "sst2", split="train")
df = dataset.to_pandas()[["sentence", "label"]]
df.columns = ["text", "label"]

# Dividir en train/dev/test
train_dev, test = np.split(df.sample(frac=1), [int((1 - test_size) * len(df))])
train, dev = np.split(train_dev, [int((1 - dev_size) * len(train_dev))])

# Guardar splits
for split, data in [("train", train), ("dev", dev), ("test", test)]:
    data.to_csv(f"sst2_{split}.csv", index=False)

# Cargar datos preprocesados
x_train, y_train = batchgen.get_dataset_singlefile("sst2_train.csv")
x_dev, y_dev = batchgen.get_dataset_singlefile("sst2_dev.csv")
x_test, y_test = batchgen.get_dataset_singlefile("sst2_test.csv")

# Vectorización
vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=max_seq_length)
vectorizer.adapt(x_train)
x_train = vectorizer(np.array(x_train)).numpy()
x_dev = vectorizer(np.array(x_dev)).numpy()
x_test = vectorizer(np.array(x_test)).numpy()
vocab = vectorizer.get_vocabulary()
vocab_size = len(vocab)

# Cargar embeddings GloVe (ejemplo)
def load_glove_embeddings(vocab):
    embeddings_index = {}
    with open("glove.6B.300d.txt", encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for i, word in enumerate(vocab):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

embedding_matrix = load_glove_embeddings(vocab) if os.path.exists("glove.6B.300d.txt") else None

# Training
# ==================================================
with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
    model = CNN_LSTM(
        max_seq_length,
        y_train.shape[1],
        vocab_size,
        embedding_dim,
        filter_sizes,
        num_filters,
        l2_reg_lambda,
        num_hidden=128  # Aumentado
    )
    
    if embedding_matrix is not None:
        model.embedding.set_weights([embedding_matrix])
        model.embedding.trainable = False  # O True para fine-tuning

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    checkpoint_dir = os.path.join("checkpoints", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "model.weights.h5"),
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss'
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    # Entrenamiento
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_dev, y_dev),
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=[tensorboard_callback, checkpoint_callback, early_stop],
        class_weight={0: 1.2, 1: 0.8} if np.mean(y_train[:, 1]) < 0.5 else None
    )

    # Evaluación
    print("\nEvaluación en Test Set:")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Visualizaciones
    plt.figure(figsize=(18, 6))
    
    # Gráfica de Exactitud
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.axhline(y=test_acc, color='g', linestyle='--', label='Test')
    plt.title('Exactitud por Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Exactitud')
    plt.legend()

    # Gráfica de Pérdida
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.axhline(y=test_loss, color='r', linestyle='--', label='Test')
    plt.title('Pérdida por Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    # Curva ROC
    plt.subplot(1, 3, 3)
    fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()

    # Matriz de Confusión Normalizada
    cm = confusion_matrix(y_true, y_pred_classes, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["neg", "pos"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión Normalizada")
    plt.savefig("confusion_matrix_normalized.png")
    plt.show()

    # Reporte de Clasificación
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred_classes))

    # Guardar Modelo
    model_save_path = os.path.join("saved_models", "cnn_lstm_model.keras")
    os.makedirs("saved_models", exist_ok=True)
    model.save(model_save_path)

    # Guardar Vectorizador
    import pickle
    with open(os.path.join("saved_models", "vectorizer.pkl"), 'wb') as f:
        pickle.dump(vectorizer, f)