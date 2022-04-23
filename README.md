# Emotion detection using deep learning

## Introduction

This project aims to classify the emotion on a person's face into one of **seven categories**, using deep convolutional neural networks. This dataset consists of 35887 grayscale which are inserted as a dataset, 48x48 sized face images with **seven emotions** - angry, disgusted, fearful, happy, neutral, sad and surprised.

## Software Requirements
* [Python 3](https://www.python.org/downloads/), [OpenCV](https://opencv.org/), [Tensorflow](https://www.tensorflow.org/)
* To install the required packages, run `pip install -r requirements.txt`.

## Core Library

The repository is currently compatible with `tensorflow-2.0` and makes use of the Keras API using the `tensorflow.keras` library.

* First, clone the repository and enter the folder

```bash
git clone https://github.com/atulapra/Emotion-detection.git
cd Emotion-detection
```

* Download the FER-2013 dataset from [here](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer) and unzip it inside the `src` folder. This will create the folder `data`.

* If you want to train this model, use:  

```bash
cd src
python emotions.py --mode train
```

```bash
cd src
python emotions.py --mode display
```

* The folder structure is of the form:  
  src:
  * data (folder)
  * `emotions.py` (file)
  * `haarcascade_frontalface_default.xml` (file)
  * `model.h5` (file)

* This implementation by default detects emotions on all faces in the webcam feed. With a simple 4-layer CNN, the test accuracy reached 63.2% in 50 epochs.
* 
![Accuracy plot](https://github.com/devanshukla/Emotion_detection-dev/blob/main/emot.png)

* In case you are looking to experiment with new datasets, you may have to deal with data in the csv format. I have provided the code I wrote for data preprocessing in the `dataset_prepare.py` file which can be used for reference.

## Algorithm

* First, the **haar cascade** method is used to detect faces in each frame of the webcam feed.

* The region of image containing the face is resized to **48x48** and is passed as input to the CNN.

* The network outputs a list of **softmax scores** for the seven classes of emotions.

* The emotion with maximum score is displayed on the screen.

## Example Output

![Mutiface]( https://github.com/devanshukla/Emotion_detection-dev/blob/main/IMG-20220111-WA0014.jpg)
