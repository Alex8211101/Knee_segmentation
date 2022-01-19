import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array, load_img

keras.backend.clear_session()
physical_devices = tf.config.list_physical_devices('GPU')
for p in physical_devices:
    tf.config.experimental.set_memory_growth(p, True)


def get_model(img_size, num_classes, deep):
    inputs = keras.Input(shape=img_size + (1,))

    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for filters in [64, 128, 256]:
        for d in range(deep):
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    for filters in [256, 128, 64, 32]:
        for d in range(deep):
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])
        previous_block_activation = x

    outputs = layers.Conv2D(
        num_classes, 3, activation="softmax", padding="same")(x)

    model = keras.Model(inputs, outputs)
    return model


def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    if np.sum(union) == 0:
        iou_score = np.sum(intersection) / (np.sum(union)+1)
    else:
        iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def load_models_paths(zestaw, deepness):
    saved_models = sorted(
        [
            os.path.join(f".\\models\\{zestaw}\\{deepness}", fname)
            for fname in os.listdir(f".\\models\\{zestaw}\\{deepness}")
            if fname.endswith(".hdf5")
        ]
    )
    return saved_models


path = os.path.dirname(os.path.abspath(__file__))
input_dir = path+"/png/"
true_dir = path+"/contours/"
przygotowane_dir = path+"/prepared/"
zestaw = "Z1"
deepness = 1
img_size = (256, 256)
num_classes = 6
model = get_model(img_size, num_classes, deepness)
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 'sparse_categorical_accuracy'])
paths = load_models_paths(zestaw, deepness)


sum_iou = [0, 0, 0]
sum_iou_pisz = [0, 0, 0]
sum_iou_udo = [0, 0, 0]
sum_iou_rzep = [0, 0, 0]
sum_iou_wkp = [0, 0, 0]
sum_iou_wkt = [0, 0, 0]
count_pisz = [0, 0, 0]
count_udo = [0, 0, 0]
count_rzep = [0, 0, 0]
count_wkp = [0, 0, 0]
count_wkt = [0, 0, 0]


for p in paths:
    model.load_weights(p)
    numer = p.split("_")[2]
    temp_sum_iou = 0
    for i in range(int(numer)*20, (int(numer)+1)*20):
        i += 1
        img = load_img(input_dir+f"{i}.png",
                       target_size=img_size, color_mode="grayscale")
        img_array = img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        val = model.predict(img_batch)

        output_dir = f"{path}/models/{zestaw}/{deepness}/{numer}/"
        threshold = 220
        mask1 = val[0, :, :, 0]*255
        mask2 = val[0, :, :, 1]*255
        mask2[mask2 < threshold] = 0
        mask2[mask2 >= threshold] = 50
        mask3 = val[0, :, :, 2]*255
        mask3[mask3 < threshold] = 0
        mask3[mask3 >= threshold] = 101
        mask4 = val[0, :, :, 3]*255
        mask4[mask4 < threshold] = 0
        mask4[mask4 >= threshold] = 152
        mask5 = val[0, :, :, 4]*255
        mask5[mask5 < threshold] = 0
        mask5[mask5 >= threshold] = 203
        mask6 = val[0, :, :, 5]*255
        mask6[mask6 < threshold] = 0
        mask6[mask6 >= threshold] = 255
        mask = mask2+mask3+mask4+mask5+mask6
        Image.fromarray(np.uint8(mask)).save(output_dir+f"{i}.png")

        true_img = load_img(
            true_dir+f"{i}.png", target_size=img_size, color_mode="grayscale")
        predicted_img = load_img(
            output_dir+f"{i}.png", target_size=img_size, color_mode="grayscale")
        temp_sum_iou += calculate_iou(true_img, predicted_img)
        mask_true = img_to_array(load_img(
            przygotowane_dir+f"{i}.png", target_size=img_size, color_mode="grayscale"))[:, :, 0]

        ############################################################################################
        mask_piszczel = np.empty_like(mask_true, dtype=np.int)
        mask_piszczel[:, :] = 0
        mask_piszczel[mask_true == 1] = 50
        piszczel_iou = calculate_iou(mask_piszczel, mask2)
        ############################################################################################
        mask_udo = np.empty_like(mask_true, dtype=np.int)
        mask_udo[:, :] = 0
        mask_udo[mask_true == 2] = 101
        udo_iou = calculate_iou(mask_udo, mask3)

        ############################################################################################
        mask_rzep = np.empty_like(mask_true, dtype=np.int)
        mask_rzep[:, :] = 0
        mask_rzep[mask_true == 3] = 152
        rzep_iou = calculate_iou(mask_rzep, mask4)

        ############################################################################################
        mask_wkp = np.empty_like(mask_true, dtype=np.int)
        mask_wkp[:, :] = 0
        mask_wkp[mask_true == 4] = 203
        wkp_iou = calculate_iou(mask_wkp, mask5)

        ############################################################################################
        mask_wkt = np.empty_like(mask_true, dtype=np.int)
        mask_wkt[:, :] = 0
        mask_wkt[mask_true == 5] = 5
        mask4[mask4 == 255] = 5
        wkt_iou = calculate_iou(mask_wkt, mask6)

        epoka = p.split("_")[3].split('.')[0]
        idx = None
        if epoka == "150":
            idx = 0
        if epoka == "200":
            idx = 1
        if epoka == "250":
            idx = 2

        sum_iou_pisz[idx] += piszczel_iou
        if piszczel_iou != 0:
            count_pisz[idx] += 1
        sum_iou_udo[idx] += udo_iou
        if udo_iou != 0:
            count_udo[idx] += 1
        sum_iou_rzep[idx] += rzep_iou
        if rzep_iou != 0:
            count_rzep[idx] += 1
        sum_iou_wkp[idx] += wkp_iou
        if wkp_iou != 0:
            count_wkp[idx] += 1
        sum_iou_wkt[idx] += wkt_iou
        if wkt_iou != 0:
            count_wkt[idx] += 1

    mean_iou = (temp_sum_iou/20)
    epoka = p.split("_")[3].split('.')[0]
    if epoka == "150":
        sum_iou[0] += mean_iou
    if epoka == "200":
        sum_iou[1] += mean_iou
    if epoka == "250":
        sum_iou[2] += mean_iou
    #print(f"Mean IoU for {numer} set is: "+format(mean_iou, ".2f")+"%")


mean_iou = np.true_divide(sum_iou, 10)
for i in range(3):
    print("=============================================")
    print(f"EPOCH {150+i*50}:")
    print("K. piszczelowa acc: " +
          format(sum_iou_pisz[i]/count_pisz[i]*100, ".2f")+"%")
    print("K. udowa acc: " +
          format(sum_iou_udo[i]/count_udo[i]*100, ".2f")+"%")
    print("Rzepka acc: " +
          format(sum_iou_rzep[i]/count_rzep[i]*100, ".2f")+"%")
    print("WKP acc: " + format(sum_iou_wkp[i]/count_wkp[i]*100, ".2f")+"%")
    print("WKT acc: " + format(sum_iou_wkt[i]/count_wkt[i]*100, ".2f")+"%")
    print(f"Mean IoU: "+format(mean_iou[i]*100, ".2f")+"%")
