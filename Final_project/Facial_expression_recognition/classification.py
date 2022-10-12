import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2

from tensorflow.keras.preprocessing.image import load_img
from numpy import asarray
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow import keras
from PIL import Image

#ML Models
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras import backend as K


plt.rcParams['figure.figsize'] = (10, 10)

#######################################################################
#Preparing the data
#######################################################################

X = []
y = []
classes = ['empty', 'sad', 'happy', 'anger', 'disgust', 'neutral', 'fear']
base_path = 'imagecapture/data/'

for i, target in enumerate(classes):
    files = os.listdir(base_path+target)  #gets the list of all files and directories in the specified directory
    for file in files:
        # load the image
        img = load_img(path=base_path+target+'/'+file,target_size=(224,224))
        # convert it to an array
        img_array = np.array(img)
        #processing the images
        processed_image = preprocess_input(img_array)
        # append the array to X
        X.append(processed_image)
        # append the numeric target to y
        y.append(i)

X = np.array(X)
y = np.array(y)
print('\nData is prepared\n')

#######################################################################
#Transfer learning from VGG 16 Model
#######################################################################
K.clear_session()
#Create a base model
# vgg_model = keras.applications.vgg16.VGG16(weights='imagenet')
base_model = keras.applications.vgg16.VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)

# Freeze all layers
for layer in base_model.layers:
    layer.trainable = False

# Feature extraction
out_features_vector = base_model.predict(X)


# Flatten out_features_vector
X_reshaped = out_features_vector.reshape((2388,7*7*512))

print('\nTransfer learning from VGG 16 Model is done\n')

#######################################################################
#Train a Logistic Regression model
#######################################################################

# Train Test Split
Xtrain,Xtest,ytrain,ytest = train_test_split(X_reshaped, y, test_size=.20,random_state=42)

logi = LogisticRegression()
print('\nFitting LogisticRegression is started ...\n')
logi.fit(Xtrain,ytrain)
print('\nFitting is finished ...\n')
ypred_train=logi.predict(Xtrain)
ypred_test = logi.predict(Xtest)

#Validation
scores_train = cross_val_score(logi,Xtrain,ytrain,cv=10)
print(f'\nvalidation scores: {scores_train}')

print(f"\naccuracy score on train: {accuracy_score(ytrain,ypred_train)}")
print(f"\naccuracy score on test: {accuracy_score(ytest,ypred_test)}")

########################################################################################
#What I got as a result (6 min run time in colab)
########################################################################################
# validation_scores = [1, 0.984, 0.989, 1, 0.989, 0.995, 0.995, 1, 0.9947644,1]
# accuracy_score_train = 1.0
# accuracy_score_test= 1.0

#######################################################################
# Real-time predictions
#######################################################################

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?)")
        break
    # Reverse color channels to RGB
    rgb_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert it to an array
    img_array = np.array(rgb_im)
    # Preprocess
    processed_image = preprocess_input(img_array)
    # Flatten out 
    X_reshaped = processed_image.reshape((1,7*7*512))
    pred = logi.predict(X_reshaped)
    # Display the resulting frame
    cv2.imshow('frame', rgb_im)
    cv2.putText(frame, str(pred))
    # Capture finishing with q
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture

cap.release()
cv2.destroyAllWindows()

##############################
#Taking photo on Colab
##############################

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
from IPython.display import Image

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename



try:
  filename = take_photo()
  print('Saved to {}'.format(filename))
  # Convert it to an array
  img_array = np.array(filename)
  # Preprocess
  processed_image = preprocess_input(img_array)
  # Flatten out 
  X_reshaped = processed_image.reshape((1,7*7*512))
  pred = logi.predict(X_reshaped)

  # Show the image which was just taken.
  display(Image(filename))

except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))

