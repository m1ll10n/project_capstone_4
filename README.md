![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
# Automated Nuclei Detection with Semantic Segmentation

This repository contains an AI model for semantic segmentation that can automatically detect nuclei in images. The model was developed for the 2018 Data Science Bowl on Kaggle, participants were to create algorithms to automate the identification of nuclei in images of human cells to advance medical discoveries.

Identifying cell nuclei is a crucial first step in many research studies as it enables researchers to analyze the DNA contained within the nucleus, which holds the genetic information that determines the function of each cell. By identifying the nuclei of cells, researchers can examine how cells respond to different treatments and gain insights into the underlying biological processes at play. This is particularly relevant to the study of a wide range of diseases, including cancer, heart disease, and rare disorders, where the identification of nuclei is essential to expedite research and speed up the development of cures.


## Applications
Below are the steps taken on solving the task.

### Data Loading
Images are converted to RGB and masks are converted to Grayscale which are then loaded as numpy array.
```python
images_np, masks_np = data_load(TRAIN_PATH)
```

### Exploratory Data Analysis
![Inspect_Image_Mask](https://user-images.githubusercontent.com/49486823/226815847-07ee7275-0663-4104-91d7-fe0a6da63c13.jpg)

The top row shows the images of cell nuclei while the bottom row shows the corresponding masks to the images.

### Data Preprocessing
These are the steps for data preprocessing:

- Images are scaled by dividing the pixel valuers by 255.0.
- Masks are converted by rounding the pixels value to 0 and 1 value to be turned into target variable.
- Augment function were applied to the train dataset before splitting into training batches (i.e. random rotation, random flip, and random zoom of images).

### 4. Model Development
This is the model architecture which are based of U-Net architecture. Few notable settings not included in the screenshot:

- The downstack uses MobileNetV2 as the base model for transfer learning.
- The upstack uses pix2pix upsample.
- The last layer is Conv2DTranspose for features extraction.
- Test data is 20% of the whole dataset and the rest is allocated to training data.
- Metrics is accuracy.
- Optimizer is Adam optimization.
- Loss function is Sparse Categorical Crossentropy function.
- Model is trained for 20 epochs.
- No early stopping implemented.

Model Architecture:

![model](https://user-images.githubusercontent.com/49486823/226815890-432c951c-750c-49bb-8000-f29f29a18908.png)

Model Parameters (with more than 4.6 million trainable parameters)

![Model_Total-Params](https://user-images.githubusercontent.com/49486823/226815922-4653ff11-f8ed-466e-8485-47696f9924ea.jpg)

## Results
This section shows all the performance of the model and the reports.
### Training Logs
The model shows training accuracy is higher than validation which indicates a good training without overfitting nor underfitting due to high accuracy achieved.

Training Accuracy:

![Training-Accuracy(TensorBoard)](https://user-images.githubusercontent.com/49486823/226815948-539ab315-4406-4af4-abd8-9da8d0318fe3.jpg)

Training Loss:

![Training-Loss(TensorBoard)](https://user-images.githubusercontent.com/49486823/226815972-8e40719c-cd8b-47a1-b0f6-dcbb68ce99c9.jpg)

### Model Performance
The model performance can be seen on the demo to better showcase its capabilities.

Below are the three sample images from an unseen dataset that the model has made segmentations on.

- Prediction #1

![Deploy_Pred](https://user-images.githubusercontent.com/49486823/226816009-91e0039c-f47f-46c0-b0e6-3d20e68d6376.png)

- Prediction #2

![Deploy_Pred(1)](https://user-images.githubusercontent.com/49486823/226816029-76b352ee-90d4-47a8-be27-90da85623c52.png)

- Prediction #3

![Deploy_Pred(2)](https://user-images.githubusercontent.com/49486823/226816048-95aa1c6e-25b2-4d75-baa7-b3c9ace4d390.png)

### Discussion
The goal of this assignment is to prevent the model from overfitting as earlier training experiments indicates the model to perform badly on the testing dataset. To avoid the model from overfitting, techniques such as data augmentation and reducing batch size can be used to increase the diversity of the data and prevent the model from memorizing rather than making inference of the training dataset. Increasing the training dataset to 80% of the whole dataset helped reducing the model from overfitting by providing more data for the model to learn from.
## Credits
This dataset is from the 2018 Data Science Bowl at Kaggle which can be obtained from https://www.kaggle.com/competitions/data-science-bowl-2018/overview.
