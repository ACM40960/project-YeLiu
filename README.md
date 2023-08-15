# Facial Emotion Recognition (FER) using Convolution Neural Networks
![Python](https://img.shields.io/badge/python-v3.11.4+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)

![ezgif com-crop](https://github.com/ACM40960/project-22200226/assets/114998243/8e0d0e6a-a864-4616-9b60-b7a16abd7937)


## Introduction
In this project, we build a Convolutional Neural Network (CNN) model with the ability to categorize seven different emotional states in human faces: Anger, Disgust, Fear, Happy, Sad, Surprise, and Neutral. Furthermore, we put this model into practical use by integrating it with a webcam for real-time application.

The benefits of this project reach different areas. For example, in healthcare, it can make it easier for people with Alexithymia to understand how others feel and to express their own emotions. It also has uses in public safety, education, jobs, and more.

## Getting Started - Instructions
Get ready for fun! Follow instructions for real-time facial emotion recognition through your webcam, all done using Python in Jupyter Notebook (Anaconda). Let's go! 🚀😃

Follow the [main file](fer_main.ipynb) for neural network training. Files Structure:
- [fer_main.ipynb](fer_main.ipynb) - Main project file about the dataset, train the CNN, and analysis
- [fer_webcam.ipynb](fer_webcam.ipynb) - Uses the pre-trained model to predict emotions via webcam
- [haarcascade_frontalface_default](haarcascade_frontalface_default.xml) - Face detection algorithm
  (we obtain this from this [repository](https://github.com/opencv/opencv/tree/master/data/haarcascades))
- [model_final.json](model_final.json) - Neural network architecture
- [weights_final.h5](weights_final.h5) - Trained model weights
- [requirements.txt](requirements.txt) - Version of each dependency
- [Literature Review](literature_review.pdf) - Synopsis of Facial Emotion Recognition project
- CNN Visualization folder - Source code for generating visual representation of CNN architecture (created using LaTeX format)
- [gitattributes](gitattributes) - Source code to upload large file more than 25mb to Github (to be ignored)

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

No need to train from scratch! Just use [fer_webcam.ipynb](fer_webcam.ipynb) with pre-trained [model_final.json](model_final.json) and [weights_final.h5](weights_final.h5) to predict facial emotions real time on your webcam. Customize it for your needs! 🤩

### Method 2 : Start from scratch
Let's get started with building the model from scratch! Follow these steps:

1. Clone the repository using the command:  
```
https://github.com/ACM40960/project-22200226.git
```

2. Download and extract [FER2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge).

This dataset contains 35,887 facial image extracts where emotion labels are linked to the corresponding pixel values in each image. It covers 7 distinct emotions/classes: Angry (0), Disgust (1), Fear (2), Happy (3), Sad (4), Surprise (5), and Neutral (6). The dataset is split into sections for training, validation, and testing purposes. All images are in grayscale and have dimensions of 48 x 48 pixels.

3. Run [fer_main.ipynb](fer_main.ipynb) and modify to your needs!


## Our Work
### Methodology


![image](https://github.com/ACM40960/project-22200226/assets/114998243/cb2da4f9-88c0-458c-bde0-8d6dffbc0103)


For the real-time facial expression recognition, we employed the **Haar Cascade classifier**, a feature-based object detection algorithm, implemented through **OpenCV**. This approach allowed us to detect faces in live video streams from the laptop's webcam. Subsequently, the selected CNN model was applied to recognize facial expressions in real-time, providing instantaneous emotion detection.

### Analysis & Findings
The classes in the dataset show imbalance,  where 'Happy' is predominant and 'Disgust' is minority. 
<img width="734" alt="Class Distribution" src="https://github.com/ACM40960/project-22200226/assets/114998243/d14b09d5-e6cc-4934-a508-219b02799d34">

### CNN Build Model and Model Summary
> :rocket: **Alert!** Buckle up, because the training process for our model takes around *16.39 hours*! :hourglass_flowing_sand: (We use a 1.4 GHz Quad-Core Intel Core i5 processor) 
* Images reshaped to 48 x 48 and normalized before entering the first convolutional layer.
* Data augmentation techniques (such as rotations, shifts, and flips) are employed to enhance the model's ability to generalize
* Utilizes 6 convolutional layers. 
* Includes Batch Normalization and Dropout Layers after each convolution layers, Early Stopping, and ReduceLROnPlateau callback to counter overfitting.
* Feature maps from convolutional layers go through Exponential Linear Unit (ELU) activation function with HeNormal kernel initializer.
* Output layer contains 7 units to classify the 7 emotion classes.
* Output layer uses softmax activation function for probabilistic class outputs.
* Loss function using categorical cross-entropy.
* Batch size: 32, Epochs: 100.
* Optimizer: Nesterov-accelerated Adaptive Moment Estimation (Nadam), which combines Adam and Nesterov Momentum.

<img width="667" alt="cnn visualization" src="https://github.com/ACM40960/project-22200226/assets/114998243/33c4698a-be55-4e8f-ae1a-425f94f514d0">



![image](https://github.com/ACM40960/project-22200226/assets/114998243/d9ea45b0-c369-49d6-be73-110f55983187)


### Model Evaluation
1. **Training vs Validation Loss & Accuracy** - Progressively improving, the model reaches ~0.95 for loss and ~70% accuracy at around 60 to 70 epochs. However, beyond this, signs of overfitting start to emerge.


   <img width="792" alt="Screenshot 2023-08-13 at 13 53 49" src="https://github.com/ACM40960/project-22200226/assets/114998243/e43b91ea-6119-44f2-b533-688193873db4">



2. **Normalized Confusion Matrix** - Model Evaluation on the Test Set\
Disgust images frequently predicted as Anger. Notably, Happy demonstrated exceptional classification performance, with 787 accurate predictions across all images, the highest among all emotion categories.


<img width="663" alt="Screenshot 2023-08-14 at 09 59 21" src="https://github.com/ACM40960/project-22200226/assets/114998243/5f897314-531e-459d-9ce0-f4d5f8f6050d">



3. **Classification Report** - Model Evaluation on the Test Set
   

| Classes       | Precision | Sensitivity (Recall) | Specificity | F1 Score | Accuracy |
| ------------- | --------- | -------------------- | ----------- | -------- | -------- |
| 0 - Anger     | 0.570     | 0.658                | 0.921       | 0.611    | 0.885    |
| 1 - Disgust   | 0.833     | 0.545                | 0.998       | 0.659    | 0.991    |
| 2 - Fear      | 0.584     | 0.439                | 0.946       | 0.502    | 0.872    |
| 3 - Happy     | 0.895     | 0.895                | 0.966       | 0.895    | 0.949    |
| 4 - Sad       | 0.594     | 0.544                | 0.926       | 0.568    | 0.863    |
| 5 - Surprise  | 0.799     | 0.767                | 0.975       | 0.783    | 0.951    |
| 6 - Neutral   | 0.615     | 0.754                | 0.900       | 0.678    | 0.875    |

> **Overall Accuracy = 69.27%**

4. **One-VS-Rest Multiclass ROC** - Model Evaluation on the Test Set

<img width="488" alt="Screenshot 2023-08-13 at 13 55 30" src="https://github.com/ACM40960/project-22200226/assets/114998243/56dead64-5f6d-4812-bf36-0e888fabcd49">

### Conclusion
The model's performance on the test set achieves an overall accuracy of approximately 69.3%. Given the class imbalance present, evaluating the model through metrics such as F1 score and ROC-AUC becomes more appropriate. Notably, both the F1 score and ROC-AUC give the highest score to the "Happy" class, while the "Fear" class has the lowest score. Looking at the images in the dataset again, it's tough for even people to tell the difference between "Fear" and other emotions like "Anger" or being "Sad". This is also true in real life – detecting the "Fear" emotion is not easy.

### Future Work
Exploring transfer learning methods with pre-trained models, facial landmark alignment, additional data augmentation, addressing class imbalance and expanding the dataset to include more varied examples could improve the model's classification capabilities.

*Additional Information: This dataset is employed in the context of a Kaggle Challenge, where the first winning entry achieved an accuracy of 71.16%, while this model attained an accuracy of **69.27%**.*

## Acknowledgments
The facial emotion recognition algorithm was adapted from the following sources:
* [mayurmadnani](https://github.com/mayurmadnani/fer.git)
* [greatsharma](https://github.com/greatsharma/Facial_Emotion_Recognition.git)

## Authors
If you have any questions with this project, feel free to reach out to us at:

- [Clementine Surya - 22200226](https://github.com/ACM40960/project-22200226.git) - [clementine.surya@ucdconnect.ie](mailto:clementine.surya@ucdconnect.ie)

- [Liu Ye - 22200868](https://github.com/ACM40960/project-YeLiu.git) - [ye.liu1@ucdconnect.ie](mailto:ye.liu1@ucdconnect.ie)




