# SpottleAIthon
## Code README
This is a Python file that contains code to train a neural network model for emotion recognition on images. The model is built using TensorFlow and Keras using a Convolutional neural Network (CNN) architecture.

##Functionality
The code, present in "source/classification.py", is organized into several functions:

**data(trainfile):** Loads the data from a CSV file and processes it into numpy arrays that can be used to train a neural network model. The function also performs normalization and one-hot encoding of the labels.

**build_net():** Builds a convolutional neural network model using TensorFlow and Keras.

**train_a_model(trainfile):** Trains the neural network model using the data loaded from a CSV file.

The dataset used to train the model includes 48x48 grayscale images of faces with one of seven emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral. The emotions of this dataset are converted into three classes (fear, happy, and sad).

The script first preprocesses the data by reading the CSV file containing the data and converting the emotions to classes. Then, the script splits the data into training and validation sets and normalizes the data by dividing each pixel value by 255. It also applies data augmentation techniques like rotation, zooming, and shifting.

##Usage
To use the code, run the train_a_model(trainfile) function with the path to the CSV file containing the data as the argument. The model will be trained on the data and the results will be printed.
