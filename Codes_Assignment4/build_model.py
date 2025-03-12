from tensorflow.keras.optimizers import Adagrad, Adadelta, RMSprop, SGD
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, LeakyReLU
import tensorflow as tf

def build_model(input_shape=(128, 128, 3), num_classes=21):
    model = tf.keras.Sequential([
        Conv2D(32, (1,1), padding="same", input_shape=input_shape),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(64, (3,3), padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        MaxPooling2D(pool_size=(2,2)),

        Conv2D(128, (3,3), padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        MaxPooling2D(pool_size=(2,2)),

        GlobalAveragePooling2D(),
        Dense(128),
        LeakyReLU(alpha=0.01),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])

    # Use Adagrad optimizer
    optimizer = Adagrad(learning_rate=0.1)  # Default learning rate is 0.001, adjust as needed

    # To use the other optimizers comment the above line and uncomment the optimez below that we want to use
  
    # Use Adadelta optimizer
    # optimizer = Adadelta(learning_rate=0.01)

    # Use RMSprop optimizer
    # optimizer = RMSprop(learning_rate=0.01)
  
    # Use SGD optimizer
    # optimizer = SGD(learning_rate=0.01, momentum=0.9)
  
    # Use default Adam optimizer
    # optimizer = Adam()
  
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

# Build model
model = build_model()
model.summary()
