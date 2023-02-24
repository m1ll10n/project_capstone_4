# %% Import
import os
from module import test_load, data_inspect, test_preparation, show_predictions
from keras.models import load_model

TEST_PATH = os.path.join(os.getcwd(), 'dataset', 'data-science-bowl-2018-2', 'test')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_models', 'model.h5')

BATCH_SIZE = 16
# %% Data Loading
test_images, test_masks = test_load(TEST_PATH)

# %% Data Inspection
data_inspect(test_images, test_masks)

# %% Prepare test data for prediction
dataset = test_preparation(test_images, test_masks, BATCH_SIZE)

# %% Make predictions
loaded_model = load_model(MODEL_PATH)
show_predictions(loaded_model, dataset, 3)

# %%
