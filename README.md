# Facial Emotion Recognition (FER) using Convolution Neural Networks
![Python](https://img.shields.io/badge/python-v3.11.4+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)

![ezgif com-video-to-gif](https://github.com/ACM40960/project-22200226/assets/114998243/c9af3c5c-41ad-4d99-acd9-4758b4a4972a)

## Introduction
In this project, we build a Convolutional Neural Network (CNN) model with the ability to categorize seven different emotional states in human faces: Anger, Disgust, Fear, Happy, Sad, Surprise, and Neutral. Furthermore, we put this model into practical use by integrating it with a webcam for real-time application.

The benefits of this project reach different areas. For example, in healthcare, it can make it easier for people with Alexithymia to understand how others feel and to express their own emotions. It also has uses in public safety, education, jobs, and more.

## Getting Started - Instructions
Get ready for fun! Follow instructions for real-time facial emotion recognition through your webcam, all done using Python in Jupyter Notebook (Anaconda). Let's go! 🚀😃

Follow the [main file](fer_main.ipynb) for neural network training. Files Structure:
- [fer_main.ipynb](fer_main.ipynb) - Main project file about the dataset and to train the CNN
- [fer_webcam.ipynb](fer_webcam.ipynb) - Uses the pre-trained model to predict emotions via webcam
- [haarcascade_frontalface_default](haarcascade_frontalface_default.xml) - Face detection algorithm
- [model.json](model.json) - Neural network architecture
- [model.h5](model.h5) - Trained model weights
- [Literature Review](literature_review.pdf) - Synopsis of Facial Emotion Recognition project
- CNN Visualization folder - Source code for generating visual representation of CNN architecture (created using LaTeX format)
- [gitattributes](gitattributes) - Source code to upload large file to Github (to be ignored)

## Prerequisites
Install these prerequisites before proceeding:
```
pip3 install numpy
pip3 install pandas
pip3 install seaborn
pip3 install keras
pip3 install matplotlib
pip3 install plotly
pip3 install scikit-learn
pip3 install tensorflow
pip3 install opencv-python
```


### Method 1 : Using the built model 

No need to train from scratch! Just use fer_webcam.ipynb with pre-trained model.json and model.h5 to predict facial emotions real time on your webcam. Customize it for your needs! 🤩

### Method 2 : Start from scratch
Let's get started with building the model from scratch! Follow these steps:

1. Clone the repository using the command:  
```
https://github.com/ACM40960/project-22200226.git
```

2. Download and extract the FER2013 dataset from the Kaggle link: 
```
https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge
```

This dataset contains 35,887 facial image extracts where emotion labels are linked to the corresponding pixel values in each image. It covers 7 distinct emotions/classes: Angry (0), Disgust (1), Fear (2), Happy (3), Sad (4), Surprise (5), and Neutral (6). The dataset is split into sections for training, validation, and testing purposes. All images are in grayscale and have dimensions of 48 x 48 pixels.

3. Download Haar Cascade classifier from this repository:
```
https://github.com/opencv/opencv/tree/master/data/haarcascades
```

4. Run fer_main.ipynb and modify to your needs!


## Our Analysis & Findings
The classes in the dataset show imbalance,  where 'Happy' is predominant and 'Disgust' is minority. 
<img width="734" alt="Class Distribution" src="https://github.com/ACM40960/project-22200226/assets/114998243/d14b09d5-e6cc-4934-a508-219b02799d34">


### CNN Build Model and Model Summary
> :rocket: **Alert!** Buckle up, because the training process for our model takes around *14.1 hours*! :hourglass_flowing_sand:
* Utilizes 6 convolutional layers.
* Images resized to 48 x 48 before entering the first convolutional layer.
* Data augmentation techniques (such as rotations, shifts, and flips) are employed to enhance the model's ability to generalize
* Feature maps from convolutional layers go through Exponential Linear Unit (ELU) activation function.
* Includes batch normalization, dropout layers, early stopping, and ReduceLROnPlateau callback to counter overfitting.
* Output layer contains 7 units to classify the 7 emotion classes.
* Output layer uses softmax activation function for probabilistic class outputs.
* Optimizer: Nesterov-accelerated Adaptive Moment Estimation (Nadam), which combines Adam and Nesterov Momentum.

<img width="893" alt="Screenshot 2023-08-11 at 19 48 45" src="https://github.com/ACM40960/project-22200226/assets/114998243/2649b795-8765-429c-9eb4-4187c2fe39ac">

![image](https://github.com/ACM40960/project-22200226/assets/114998243/d9ea45b0-c369-49d6-be73-110f55983187)


### Model Evaluation
1. Confusion Matrix
![confusion_matrix](https://github.com/ACM40960/project-22200226/assets/114998243/2995ba89-eb20-448c-95c1-c6ccdb2ea341)

2. Classification Report

| Classes       | Precision | Sensitivity (Recall) | Specificity | F1 Score | Accuracy |
| ------------- | --------- | -------------------- | ----------- | -------- | -------- |
| 0 - Anger     | 0.604     | 0.646                | 0.933       | 0.624    | 0.894    |
| 1 - Disgust   | 0.750     | 0.436                | 0.998       | 0.552    | 0.989    |
| 2 - Fear      | 0.592     | 0.422                | 0.950       | 0.493    | 0.872    |
| 3 - Happy     | 0.886     | 0.902                | 0.962       | 0.894    | 0.948    |
| 4 - Neutral   | 0.571     | 0.539                | 0.920       | 0.555    | 0.857    |
| 5 - Sad       | 0.784     | 0.793                | 0.971       | 0.789    | 0.951    |
| 6 - Surprise  | 0.610     | 0.759                | 0.897       | 0.676    | 0.837    |

**Overall Accuracy = 69.16%**\
Based on the performance metrics, the emotion category with the highest overall performance is "Happy". This category exhibits both high precision, recall, and F1-score while the lowest performing emotion category is "Disgust". While the model demonstrates a moderate precision, the recall and F1-score are relatively low. This could be because there are very few samples available for this category.


*Additional Information: This dataset is employed in the context of a Kaggle Challenge, where the first winning entry achieved an accuracy of 71.16%, while our own model attained an accuracy of 69.16%.*

## Credits
The facial emotion recognition algorithm was adapted from the following sources:
* [mayurmadnani](https://github.com/mayurmadnani/fer.git)
* [greatsharma](https://github.com/greatsharma/Facial_Emotion_Recognition.git)

## Authors

- [@ClementineSurya_22200226](https://github.com/ACM40960/project-22200226.git)

- [@LiuYe_22200868](https://github.com/ACM40960/project-YeLiu.git)

*For Inquiry:*\
If you have any questions with this project, feel free to reach out to us. You can contact us at:
* [suryaclementine@gmail.com](mailto:suryaclementine@gmail.com)
* [liuyeirtj@gmail.com](mailto:liuyeirtj@gmail.com)

