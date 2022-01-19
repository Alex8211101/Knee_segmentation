import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img

# PARAMS
deepness = 1
epochs = 250
img_size = (256, 256)
num_classes = 6
batch_size = 8
howManyInSeries = 100
howManySeries = 10
zestaw = "z1"


def get_model(img_size, num_classes, deep):
    inputs = keras.Input(shape=img_size + (1,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    #x = layers.GaussianNoise(0.1)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        for d in range(deep):
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        for d in range(deep):
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(
        num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


class KneeSequence(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size, dtype="uint8")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size,
                           color_mode="grayscale")
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size,
                           color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            #y[j] -= 1
        return x, y


dirname = os.path.dirname(__file__)
input_dir = dirname+"\\png2\\"
target_dir = dirname+"\\prepared2\\"  # we are using data with augmentation


input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

metrics = []
losses = []
val_metrics = []
val_losses = []

for fold_no in range(7, howManySeries):
    modelName = f"{zestaw}_{deepness}_"+str(fold_no)+"_"

    # Create test series
    test_input_img_paths = input_img_paths[howManyInSeries *
                                           fold_no:howManyInSeries*fold_no+howManyInSeries-1]
    test_target_img_paths = target_img_paths[howManyInSeries *
                                             fold_no:howManyInSeries*fold_no+howManyInSeries-1]

    # Create validation series
    if fold_no < 9:
        val_input_img_paths = input_img_paths[howManyInSeries*(
            fold_no+1):howManyInSeries*(fold_no+1)+howManyInSeries-1]
        val_target_img_paths = target_img_paths[howManyInSeries*(
            fold_no+1):howManyInSeries*(fold_no+1)+howManyInSeries-1]
    else:
        val_input_img_paths = input_img_paths[0:howManyInSeries-1]
        val_target_img_paths = target_img_paths[0:howManyInSeries-1]

    # Create train series
    train_input_img_paths = [path for path in input_img_paths if (
        path not in test_input_img_paths) and (path not in val_input_img_paths)]
    train_target_img_paths = [path for path in target_img_paths if (
        path not in test_target_img_paths) and (path not in val_target_img_paths)]

    train_gen = KneeSequence(batch_size, img_size,
                             train_input_img_paths, train_target_img_paths)
    val_gen = KneeSequence(batch_size, img_size,
                           val_input_img_paths, val_target_img_paths)
    test_gen = KneeSequence(batch_size, img_size,
                            test_input_img_paths, test_target_img_paths)

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()
    physical_devices = tf.config.list_physical_devices('GPU')
    for p in physical_devices:
        tf.config.experimental.set_memory_growth(p, True)

    model = get_model(img_size, num_classes, deep=deepness)
    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=[
                  'accuracy', 'sparse_categorical_accuracy'])

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=os.path.join(
            f"./models/Z1/{deepness}/"+modelName+"{epoch:03d}.hdf5"), save_weights_only=True, save_best_only=False, monitor='sparse_categorical_accuracy', verbose=0)
    ]
    print("------------------------------------------------------")
    print('Training for fold '+str(fold_no))
    history = model.fit(train_gen, epochs=epochs,
                        callbacks=callbacks, verbose=1, validation_data=val_gen)

    loss = history.history["loss"]
    metric = history.history["sparse_categorical_accuracy"]
    val_loss = history.history["val_loss"]
    val_metric = history.history["val_sparse_categorical_accuracy"]
    losses.append(loss)
    metrics.append(metric)
    val_losses.append(val_loss)
    val_metrics.append(val_metric)

losses = np.array(losses).mean(axis=0)
metrics = np.array(metrics).mean(axis=1)
val_losses = np.array(val_losses).mean(axis=0)
val_metrics = np.array(val_metrics).mean(axis=1)
plt.plot(metric)
plt.plot(val_metric)
plt.title('Model accuracy during training')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()
plt.plot(losses)
plt.plot(val_losses)
plt.title('Model loss during training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()


# clear folders
saved_models = sorted(
    [
        os.path.join(f"./models/Z1/{deepness}/", fname)
        for fname in os.listdir(f"./models/Z1/{deepness}/")
        if fname.endswith(".hdf5")
    ]
)
for m in saved_models:
    names = m.split('_')
    temp = names[3].split('.')
    e = temp[0]
    if e not in ["150", "200", "250"]:
        os.remove(m)
