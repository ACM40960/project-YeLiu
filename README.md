# Facial Emotion Recognition
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/python-v3.11.4+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)

## Getting Started
Get ready for fun! Follow instructions for real-time facial emotion recognition through your webcam. Let's go! 🚀😃

## Instructions
Follow the [main file](fer_main.ipynb) for neural network training.

Files Structure:
- fer_main.ipynb - Tutorial to train the CNN
- fer_webcam.ipynb - Uses the pre-trained model to predict emotions via webcam
- model.json - Neural network architecture
- model.h5 - Trained model weights

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

Clone the repository using the command:  
```
https://github.com/ACM40960/project-22200226.git
```

Download and extract the dataset from the Kaggle link: 
```
https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge
```
Download Haar Cascade classifier from this repository:
```
https://github.com/opencv/opencv/tree/master/data/haarcascades
```

Run fer.ipynb and modify to your needs!


## Analysis & Findings
The classes in the dataset show imbalance,  where 'Happy' is predominant and 'Disgust' is minority. 
![newplot 20 22 41](https://github.com/ACM40960/project-22200226/assets/114998243/e2264db8-3437-4e9f-ba6c-3f169085ff20)

## Our CNN Build Model and Model Summary
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


## Model Evaluation
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


*Additional Information: This dataset is employed in the context of a Kaggle Challenge, where the first winning entry achieved an accuracy of 71.16%, while our own model attained an accuracy of 69.16%.*

## Authors

- [@ClementineSurya_22200226](https://github.com/ACM40960/project-22200226.git)

- [@LiuYe_22200868](https://github.com/ACM40960/project-YeLiu.git)
