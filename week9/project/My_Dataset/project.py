import numpy as np
import pandas as pd
import tensorflow as tf
import os

from tensorflow.keras.preprocessing.image import load_img
from numpy import asarray
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Regularization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout

#Ignore Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Loading Images, Converting to Arrays and Splitting
X = []
y = []
classes = ['empty', 'bird', 'mug', 'sunglass']
base_path = 'imageclassifier/data/'

for i, target in enumerate(classes):
      files = os.listdir(base_path+target)  #gets the list of all files and directories in the specified directory
      for file in files:
         # load the image
         img = load_img(base_path+target+'/'+file)
         # convert it to an array
         img_array = asarray(img)
         # append the array to X
         X.append(img_array)
         # append the numeric target to y
         y.append(i)


X = np.array(X)
y = np.array(y)

# shuffle the data
shuffler = np.random.permutation(len(X))
X = X[shuffler]
y = y[shuffler]

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(X[i], cmap=plt.cm.Greys)
    plt.axis('off')
plt.show()

print(f'''\n shape for X: {X.shape}, and shape for y: {y.shape} \n''')

#spliting the data
xtrain = X[:1855,:,:,:]
xtest = X[1855:,:,:,:]

ytrain = y[:1855]
ytest = y[1855:]
print(f'''After splitting, shape for xtrain: {xtrain.shape}, shape for xtest: {xtest.shape}''')

ytest_true = ytest.copy()


#Normalizing the data to help with the training (Scale these values to a range of 0 to 1)
xtrain = xtrain / 255.0
xtest = xtest / 255.0
print('\n Data is normalized \n')

# One-Hot-Encoding the labels
ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)
print('\n labels are one hot encoded \n ')

#Building a CNN with 8 Layers
#Defining the model
from tensorflow.keras import backend as K
K.clear_session()
model = Sequential([
    
    # first convolutional and max pooling layer
    Conv2D(filters=6,
          kernel_size=(5,5),
          strides=(1, 1),
          padding="valid",
          activation="relu",
          input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2,2),
                strides=(2, 2),
                padding="valid"),
    #BatchNormalization(), # optional
    
    # Second convolutional and max pooling layer
    Conv2D(filters=16,
          kernel_size=(5,5),
          strides=(1, 1),
          padding="valid",
          activation="relu"),
    MaxPooling2D(pool_size=(2,2),
                strides=(2,2),
                padding="valid"),
    #BatchNormalization(), # optional
    
    # Flatten
    Flatten(),
    
    # Fully connected
    # layer 1
    Dense(120, activation="relu"),
    #Dropout(0.2), # optional
    
    # layer 2
    Dense(84, activation="relu"),
    #Dropout(0.2), # optional
    
    # Output layer
    Dense(4, activation="softmax")
])

print(model.summary())

# Compile the model
model.compile(optimizer=Adam(), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

#Fit the model
print('\n Fitting will start now \n')
fit_history = model.fit(xtrain, ytrain, batch_size=32, epochs=10, validation_split=0.2)
#Evaluate model
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.family"] = "monaco"
plt.rcParams["font.size"] = 12
plt.rcParams["font.weight"] = "bold"
plt.subplot(1, 2, 1)
plt.plot(fit_history.history["accuracy"], label="train data")
plt.plot(fit_history.history["val_accuracy"], label="test data")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(fit_history.history["loss"], label="train data")
plt.plot(fit_history.history["val_loss"], label="test data")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()

df_metrics = pd.DataFrame(fit_history.history)
print(df_metrics)

#Prediction
ypred = model.predict(xtest)
print(f''''the prediction for the first test image is {np.argmax(ypred[0])} 
and the real image label is {ytest_true[0]}''')


#Create a confusion matrix
y_pred_labels = np.argmax(ypred, axis=1)
plt.rcParams["figure.figsize"] = (8, 8)

cm = confusion_matrix(y_true=ytest_true, y_pred=y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(ytest_true))
disp.plot()
print(ytest.shape)

