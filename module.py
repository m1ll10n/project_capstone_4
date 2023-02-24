import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2, os, glob

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from keras.layers import Layer, RandomFlip, RandomRotation, RandomZoom, Input, Concatenate, Conv2DTranspose
from keras.utils import array_to_img
from tensorflow_examples.models.pix2pix import pix2pix
from keras.applications import MobileNetV2
from keras.models import Model
from keras.callbacks import Callback
from keras.losses import SparseCategoricalCrossentropy
from keras.utils import plot_model

class Augment(Layer):
    def __init__(self,seed=42):
        super().__init__()
        self.augment_inputs = RandomRotation(0.2, seed=seed)
        self.augment_labels = RandomRotation(0.2, seed=seed)

    def call(self,inputs,labels, seed=42):
        inputs = self.augment_inputs(inputs)
        inputs = RandomFlip('horizontal', seed=seed)(inputs)
        inputs = RandomZoom(0.2, seed=seed)(inputs)

        labels = self.augment_labels(labels)
        labels = RandomFlip('horizontal', seed=seed)(labels)
        labels = RandomZoom(0.2, seed=seed)(labels)

        return inputs,labels

class TerminateOnBaseline(Callback):
    def __init__(self, monitor='accuracy', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True 

def data_load(TRAIN_PATH):
    train_ids = next(os.walk(TRAIN_PATH))[1]

    images = []
    masks = []

    for id in tqdm(train_ids):
        img = cv2.imread(os.path.join(TRAIN_PATH, id, 'images', f'{id}.png'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        images.append(img)

        final_mask = 0
        for mask_file in glob.glob(os.path.join(TRAIN_PATH, id, 'masks', '*.png')):
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (128, 128))
            final_mask = np.maximum(final_mask, mask)
        
        masks.append(final_mask)
    
    images_np = np.array(images)
    masks_np = np.array(masks)

    return images_np, masks_np

def data_inspect(images_np, masks_np):
    plt.figure(figsize=(10,10))
    for i in range(1,4):
        plt.subplot(1,3,i)
        plt.imshow(images_np[i])
        plt.axis('off')
        
    plt.show()

    plt.figure(figsize=(10,10))
    for i in range(1,4):
        plt.subplot(1,3,i)
        plt.imshow(masks_np[i], cmap="gray")
        plt.axis('off')
        
    plt.show()

def data_split(images_np, masks_np):
    masks_np_exp = np.expand_dims(masks_np, axis=-1)
    converted_masks = np.round(masks_np_exp/255).astype(np.int64)
    converted_images = images_np / 255.0

    X_train, X_test, y_train, y_test = train_test_split(converted_images, converted_masks, test_size=0.2, random_state=42)

    X_train_tensor = tf.data.Dataset.from_tensor_slices(X_train)
    X_test_tensor = tf.data.Dataset.from_tensor_slices(X_test)
    y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
    y_test_tensor = tf.data.Dataset.from_tensor_slices(y_test)

    train_dataset = tf.data.Dataset.zip((X_train_tensor, y_train_tensor))
    test_dataset = tf.data.Dataset.zip((X_test_tensor, y_test_tensor))

    return train_dataset, test_dataset

def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image','True Mask','Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()

def unet_model(input_shape, output_channels:int, MODEL_PNG_PATH):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False)

    layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project',
    ]

    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    down_stack = Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False
    up_stack = [
        pix2pix.upsample(512, 3),
        pix2pix.upsample(256, 3),
        pix2pix.upsample(128, 3),
        pix2pix.upsample(64, 3),
    ]
    
    inputs = Input(shape=input_shape)

    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = Concatenate()
        x = concat([x, skip])
    # x = Dropout(0.3)(x)

    last = Conv2DTranspose(filters=output_channels, kernel_size=3, strides=2, padding='same')
    outputs = last(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    loss = SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    plot_model(model, to_file=MODEL_PNG_PATH)

    return model

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

def show_predictions(model, dataset,num=1):
    for image,mask in dataset.take(num):
        pred_mask = model.predict(image)
        display([image[0],mask[0],create_mask(pred_mask)])

def test_load(TEST_PATH):
    test_images = []
    test_masks = []

    test_image_dir = os.path.join(TEST_PATH,'inputs')
    for image_file in os.listdir(test_image_dir):
        img = cv2.imread(os.path.join(test_image_dir,image_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(128,128))
        test_images.append(img)
        
    test_mask_dir = os.path.join(TEST_PATH,'masks')
    for mask_file in os.listdir(test_mask_dir):
        mask = cv2.imread(os.path.join(test_mask_dir,mask_file),cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask,(128,128))
        test_masks.append(mask)

    test_images = np.array(test_images)
    test_masks = np.array(test_masks)

    return test_images, test_masks

def test_preparation(test_images, test_masks, BATCH_SIZE):
    masks_np_exp = np.expand_dims(test_masks, axis=-1)
    converted_masks = np.round(masks_np_exp/255).astype(np.int64) # To convert to class labels [0, 1]
    converted_images = test_images / 255.0 # Normalize image values

    X_tensor = tf.data.Dataset.from_tensor_slices(converted_images)
    y_tensor = tf.data.Dataset.from_tensor_slices(converted_masks)

    dataset = tf.data.Dataset.zip((X_tensor, y_tensor))
    dataset_batches = dataset.batch(BATCH_SIZE)

    return dataset_batches