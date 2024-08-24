# Global Wheat Detection Competition
https://www.kaggle.com/competitions/global-wheat-detection

# Dataset and problem Overview

This competition is to discover the paths to identify wheat heads in a given image which is an object detection problem. 

The given dataset consists of 

- Train images (3422 images),
- Test images (10 images)

And notably the labels are:

- Train.csv (147793 rows and 5 columns)

Which infers that each image has multiple bounding boxes which detect the presence of wheat heads. And we are destined to find the same in the given test images. 

# Open and Load Images or dataset

Based on the observation, the csvs contain the labels, coordinates of bounding boxes, the photo ids. 
- Train test split is performed in the csvs and then their respective values can be mapped with the id in the csv file.
- Here I have used the open cv library to read the images and then albumentations library to transform the images.

**DatasetClass**
- Firstly the id and label are tapped from the csv file, then the id is sent along with the file format and is concatenated as string along with the data directory followed by file format like jpeg, png etc. 
- Then the cv2 command 'imread' is used to open the images and convert them to arrays. 

**Preprocess**
- Then the format is changed from bgr to rgb which is its default return type. Also in some grayscale images are used. 
- Then the respective labels are extracted and are attached before return.
- Then post opening the image it is sent to the transforms method and then brought back in. 
- The return value of the method contains image tensors and which can be later unpacked if needed.

**DataLoader**
- The processed data from the above step are passed through data loader to split into batches and shuffle the dataset to add some more randomness.
- Post the data loader the output was in the form of list of tensors of all batches input data, tensors of all labels.
Here the parameters used are batch size 1 for most models. Often batch sizes are kept low to reduce computational efficiency and speed.


# Experiments List:

**With Only prerequisite augmentations**

| EXP | MODEL | AUGMENTATIONS | SIZE | OTHER MODEL PARAMETERS | TEST mAP | PRIVATE | PUBLIC |
| -------- | ------- | ------- | ---- | ------- | ------- | ------- | ------- |
| 1 | Faster RCNN | ToTensor | 1024 | AdamW, 20 epochs | 0.6258 | 0.5492 | 0.6108 |
| 2 | Faster RCNN | ToTensor | 1024 | SGD, 10 epochs | 0.6164 | 0.5489 | 0.6107 |
| 3 | Faster RCNN | Sharpen, ToTensor | 1024 | SGD, 10 epochs | 0.6108 | 0.5456 | 0.6018 |
| 4 | Faster RCNN | ToTensor | 1024 | AdamW, 5 epochs | 0.6034 | 0.5267 | 0.6334 |
| 5 | Faster RCNN | ToTensor + lessloss | 1024 | SGD, 8 epochs | 0.5549 | 0.4954 | 0.6145 |

**With augmentations**

| EXP | MODEL | AUGMENTATIONS | SIZE | OTHER MODEL PARAMETERS | TEST mAP | PRIVATE | PUBLIC |
| -------- | ------- | ------- | ---- | ------- | ------- | ------- | ------- |
| 6 | Faster RCNN | Gray, ToTensor | 1024 | SGD, 10 epochs | 0.5884 | 0.5283 | 0.6157 |
| 7 | Faster RCNN | Brightness Contrast, ToTensor | 1024 | AdamW, 10 epochs | 0.5995 | 0.5267 | 0.6334 |
| 8 | Faster RCNN | Hue Saturation, ToTensor | 512 | SGD, 10 epochs | 0.5835 | 0.5236 | 0.6025 |
| 9 | Faster RCNN | Gray, ToTensor | 512 | AdamW, 10 epochs | 0.5661 | 0.5179 | 0.6305 |

**TTA, Pseudolabelling**

Pseudolabelling is the process of adding a proportion of test or prediction data with high confidence with the train data. And retrain the model to boost the performance. 

TTA(Test Time Augmentations) is used to incorporate the augmentations to test data while prediction one by one and the predicted values are averaged for final predictions.

| EXP | MODEL | AUGMENTATIONS | SIZE | OTHER MODEL PARAMETERS | TEST mAP | PRIVATE | PUBLIC |
| -------- | ------- | ------- | ---- | ------- | ------- | ------- | ------- |
| BEST | Faster RCNN | ToTensor, TTA | 1024 | AdamW, 10 epochs | 0.6405 | 0.5817 | 0.654 |
| 10 | Faster RCNN | ToTensor, Pseudolabelling (high mAP)| 1024 | AdamW, 10 epochs | 0.6949 | 0.4991 | 0.5756 |
| 11 | Faster RCNN | ToTensor, Pseudolabelling (less loss)| 1024 | AdamW, 8 epochs | 0.6949 | 0.4943 | 0.5533 |

Even tried with efficient det which was next preferred to fast rcnn but it didnot work out well with the scores.

# Conclusion:

Upon choosing the best mAP and submission of models the best variant turned out to be the one mentioned below:

    MODEL : Faster RCNN
    Learning rate: 0.0001
    Optimizer: AdamW
    Local mAP : 0.6627
    Public Score : 0.654
    Private Score: 0.5817

# Future Scope:

- Trials upon pseudolabelling multiple times(recurrent).
- Add on some more data
- Trials with some more augmentations can hopefully improve the mdoel.
- Have to try with some more models.
