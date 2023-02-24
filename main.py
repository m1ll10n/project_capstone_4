# %% Import
import tensorflow as tf
import numpy as np
import os, datetime
from module import Augment, TerminateOnBaseline, data_load, data_inspect, data_split, display, unet_model, show_predictions


from keras.callbacks import TensorBoard

TRAIN_PATH = os.path.join(os.getcwd(), 'dataset', 'data-science-bowl-2018', 'stage1_train')

MODEL_PATH = os.path.join(os.getcwd(), 'saved_models', 'model.h5')
MODEL_PNG_PATH = os.path.join(os.getcwd(), 'saved_models', 'model.png')
MODEL_FOLDER_PATH = os.path.join(os.getcwd(), 'saved_models')
if not os.path.exists(MODEL_FOLDER_PATH):
    os.makedirs(MODEL_FOLDER_PATH)
# %% Data Loading
images_np, masks_np = data_load(TRAIN_PATH)

# %% Exploratory Data Analysis
data_inspect(images_np, masks_np)

# %% Data Preprocessing
train_dataset, test_dataset = data_split(images_np, masks_np)

# %%
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
TRAIN_SIZE = len(train_dataset)
STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE

train_batches = (train_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=AUTOTUNE)
)

test_batches = test_dataset.batch(BATCH_SIZE)

# %%
for images, masks in train_batches.take(2):
    sample_image,sample_mask = images[0],masks[0]
    display([sample_image,sample_mask])

# %% Model Development
input_shape = np.shape(images_np)[1:]
OUTPUT_CLASSES = 2

model = unet_model(input_shape=input_shape, output_channels=OUTPUT_CLASSES)

# %%
EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(test_dataset)//BATCH_SIZE//VAL_SUBSPLITS

log_dir = os.path.join(os.getcwd(), 'tensorboard_logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = TensorBoard(log_dir=log_dir)

history = model.fit(train_batches,validation_data=test_batches,epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,validation_steps=VALIDATION_STEPS,callbacks=[tb_callback, TerminateOnBaseline(monitor='accuracy', baseline=0.98)])
# %%
show_predictions(model, test_batches)

# %%
model.save(MODEL_PATH)
# %%
