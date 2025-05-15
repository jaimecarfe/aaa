#! /usr/bin/env python
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from IPython import embed
import pandas as pd
from datasets import load_dataset

# SELECT WHICH MODEL YOU WISH TO RUN:
from cnn_lstm import CNN_LSTM   # OPTION 0
MODEL_TO_RUN = 0

import batchgen

# Parameters
# ==================================================
dev_size = .15  # Aumentado para tener más datos de validación
embedding_dim = 24  # Reducido
max_seq_length = 70
filter_sizes = [3, 4, 5]
num_filters = 24  # Reducido
dropout_prob = 0.65  # Aumentado
l2_reg_lambda = 0.002  # Aumentado
batch_size = 128
num_epochs = 25  # Aumentado por las dudas
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
x_text, y = batchgen.get_dataset_singlefile("sst2.csv", limit=250000)  # Aumentado a 250k

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
        model = CNN_LSTM(
            x_train.shape[1], 
            y_train.shape[1], 
            vocab_size, 
            embedding_dim, 
            filter_sizes, 
            num_filters, 
            l2_reg_lambda,
            num_hidden=64  # Reducido
        )
    elif MODEL_TO_RUN == 1:
        model = LSTM_CNN(x_train.shape[1], y_train.shape[1], vocab_size, embedding_dim, filter_sizes, num_filters, l2_reg_lambda)
    elif MODEL_TO_RUN == 2:
        model = CNN(x_train.shape[1], y_train.shape[1], vocab_size, embedding_dim, filter_sizes, num_filters, l2_reg_lambda)
    elif MODEL_TO_RUN == 3:
        model = LSTM(x_train.shape[1], y_train.shape[1], vocab_size, embedding_dim)
    else:
        print("PLEASE CHOOSE A VALID MODEL!\n0 = CNN_LSTM\n1 = LSTM_CNN\n2 = CNN\n3 = LSTM\n")
        exit()

    # Optimizer con learning rate más bajo y clipnorm
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0005,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    # Callbacks
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1,
        embeddings_freq=1,
        update_freq='epoch'
    )
    
    checkpoint_dir = os.path.join("checkpoints", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "model.weights.h5"),
        save_weights_only=True,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=4,  # Aumentado a 4
        restore_best_weights=True,
        verbose=1,
        mode='max'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )

    print("Training model...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_dev, y_dev),
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=[
            tensorboard_callback,
            checkpoint_callback,
            early_stopping,
            reduce_lr
        ],
        verbose=1
    )

    # Gráficas mejoradas
    plt.figure(figsize=(14, 6))
    
    # Gráfica de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    plt.title("Training vs Validation Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Gráfica de precisión
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
    plt.title("Training vs Validation Accuracy", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    plt.savefig("training_metrics_improved.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("\nEvaluation on Dev Set:")
    loss, accuracy = model.evaluate(x_dev, y_dev)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Obtener predicciones del modelo
    y_pred_probs = model.predict(x_dev)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_dev, axis=1)

    # Calcular F1-Score
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"\nF1-Score (weighted): {f1:.4f}")

    # Reporte de clasificación completo
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Matriz de confusión mejorada
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca(), colorbar=False)
    plt.title("Confusion Matrix (Improved Model)", fontsize=14)
    plt.grid(False)
    plt.savefig("confusion_matrix_improved.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Guardar el modelo completo
    model_save_path = os.path.join("saved_models", "improved_cnn_lstm_model.keras")
    os.makedirs("saved_models", exist_ok=True)
    model.save(model_save_path)

    # Guardar el vectorizador
    import pickle
    vectorizer_save_path = os.path.join("saved_models", "improved_vectorizer.pkl")
    with open(vectorizer_save_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"\nModel saved to {model_save_path}")
    print(f"Vectorizer saved to {vectorizer_save_path}")