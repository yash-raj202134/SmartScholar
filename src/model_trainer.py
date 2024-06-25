# model training
import os
from tensorflow.keras import layers # type: ignore
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt # type: ignore

from tensorflow.keras.callbacks import EarlyStopping # type: ignore


# creating shallow_mlp_model  (MLP)
def create_mlp_mode(lookup):

    # Creating shallow_mlp_model (MLP) with dropout layers
    model1 = keras.Sequential([
        # First hidden layer: 512 neurons, ReLU activation function, with dropout.
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),  # Adding dropout for regularization.

        # Second hidden layer: 256 neurons, ReLU activation function, with dropout.
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),  # Adding dropout for regularization.

        # Output layer: The number of neurons equals the vocabulary size (output vocabulary of the StringLookup layer), with a sigmoid activation function.
        layers.Dense(lookup.vocabulary_size(), activation='sigmoid')
    ])
    # Compile the model
    model1.compile(loss="binary_crossentropy", optimizer='adam', metrics=['binary_accuracy'])

    return model1


def train_model(train_dataset,validation_dataset,lookup):
    
    # Creating shallow_mlp_model (MLP) with dropout layers
    model1 = create_mlp_mode(lookup)


    # Add early stopping
    # Number of epochs with no improvement after which training will be stopped.
    # Restore weights from the epoch with the best value of the monitored quantity.
    early_stopping = EarlyStopping(patience=5,restore_best_weights=True)

    # Train the model
    # Add early stopping callback.verbose=1
    history = model1.fit(train_dataset,validation_data=validation_dataset,epochs=20,callbacks=[early_stopping])

    return history

def plot_loss(history,item,save_path=None):
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title(f"Train and Validation {item} Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    
    # Save the plot
    plt.savefig(os.path.join(save_path, f"{item}_plot.png"))
    plt.close()

