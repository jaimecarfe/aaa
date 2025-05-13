import tensorflow as tf
from tensorflow.keras import layers
from keras.saving import register_keras_serializable

@register_keras_serializable()
class CNN_LSTM(tf.keras.Model):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 filter_sizes, num_filters, l2_reg_lambda=0.0, num_hidden=100):
        super(CNN_LSTM, self).__init__()

        # 1. Embedding layer
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_size,

            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg_lambda)
        )

        # 2. Convolution + MaxPooling layers (for each filter size)
        self.conv_blocks = []
        for filter_size in filter_sizes:
            conv = layers.Conv1D(
                filters=num_filters,
                kernel_size=filter_size,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg_lambda)
            )
            pool = layers.GlobalMaxPooling1D()
            self.conv_blocks.append(tf.keras.Sequential([conv, pool]))

        # 3. Dropout
        self.dropout = layers.Dropout(rate=0.5)

        # 4. LSTM
        self.lstm = layers.LSTM(num_hidden)

        # 5. Output layer
        self.output_layer = layers.Dense(
            units=num_classes,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg_lambda)
        )

    def call(self, inputs, training=False):
        x = self.embedding(inputs)

        # Apply each convolution block
        conv_outputs = [conv_block(x) for conv_block in self.conv_blocks]

        # Concatenate conv features
        x = tf.concat(conv_outputs, axis=-1)

        # Dropout
        if training:
            x = self.dropout(x, training=training)

        # Expand dims for LSTM input (optional, if needed)
        x = tf.expand_dims(x, axis=1)

        # LSTM
        x = self.lstm(x)

        # Output layer
        return self.output_layer(x)
