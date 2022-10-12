import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pickle
import warnings


from tensorflow.keras.preprocessing.image import load_img
from numpy import asarray
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow import keras
from PIL import Image
from PIL import ImageDraw 
from PIL import ImageFont

#ML Models
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras import backend as K
import glob
import cv2


plt.rcParams['figure.figsize'] = (10, 10)
warnings.filterwarnings('ignore')
#######################################################################
#Preparing the data
#######################################################################

if __name__ == '__main__':

  mode = 'test'

  img_orig = []
  X = []
  y = []
  classes = ['empty', 'sad', 'happy', 'anger', 'disgust', 'neutral', 'fear']

  if mode == 'train':
    base_path = 'imagecapture/data/'
  else:
    base_path = '/content/drive/MyDrive/Facial_expression_recognition/Test_data'


  if mode == 'train':
    for i, target in enumerate(classes):
        files = os.listdir(base_path+target)  #gets the list of all files and directories in the specified directory
        for f in files:
            # load the image
            img = load_img(path=base_path+target+'/'+f,target_size=(224,224))
            
            # convert it to an array
            img_array = np.array(img)
            #processing the images
            processed_image = preprocess_input(img_array)
            # append the array to X
            X.append(processed_image)
            # append the numeric target to y
            y.append(i)
  else:
    files = glob.glob(base_path+'/*.png')
    for f in files:
      # load the image
      img = load_img(path=f,target_size=(224,224))
      img_orig.append(img)
      # convert it to an array
      img_array = np.array(img)
      #processing the images
      processed_image = preprocess_input(img_array)
      # append the array to X
      X.append(processed_image)
      # append the numeric target to y
      y.append(0)

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
  # pdb.set_trace()

  # Flatten out_features_vector
  X_reshaped = out_features_vector.reshape((-1,7*7*512))

  print('\nTransfer learning from VGG 16 Model is done\n')

  #######################################################################
  #Train a Logistic Regression model
  #######################################################################
  if mode == 'train':

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

    #savingg the model
    print('\nSaving the model.\n')
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(logi, file)
  
  else:
    #loading the model
    with open('pickle_model.pkl', 'rb') as f:
        logi = pickle.load(f)
    # import pdb
    # pdb.set_trace()
    print('\nPredicting the emotion ...\n')
    test_pred = logi.predict(X_reshaped)

    # font = ImageFont.truetype("sans-serif.ttf", 16)
    
    for i, pred in enumerate(test_pred):
      # converting the prediction to text

      predict = int(pred)
      result = classes[predict]
      # Saving mage with prediction
      img = img_orig[i]
      # import pdb
      # pdb.set_trace()
      draw = ImageDraw.Draw(img)

      draw.text((10, 10),str(result),(255,255,0))
      img.save(files[i].replace('Test_data', 'Test_result'))

      # cv2.putText(img, str(result))
      # cv2.imwrite(files[i].replace('demo', 'expression_result'), img)













########################################################################################
#What I got as a result (6 min run time in colab)
########################################################################################
# validation_scores = [1, 0.984, 0.989, 1, 0.989, 0.995, 0.995, 1, 0.9947644,1]
# accuracy_score_train = 1.0
# accuracy_score_test= 1.0

  #######################################################################
  # Real-time predictions
  #######################################################################
  print('\nstarting with the real-time prediction\n')
  # cap = cv2.VideoCapture(-1)
  # if not cap.isOpened():
  #     print("Cannot open camera")
  #     exit()
  # while True:
  #     # Capture frame-by-frame
  #     ret, frame = cap.read()
  #     # if frame is read correctly ret is True
  #     if not ret:
  #         print("Can't receive frame (stream end?)")
  #         break
  #     # Reverse color channels to RGB
  #     rgb_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  #     # Convert it to an array
  #     img_array = np.array(rgb_im)
  #     # Preprocess
  #     processed_image = preprocess_input(img_array)
  #     # Flatten out 
  #     X_reshaped = processed_image.reshape((1,7*7*512))
  #     pred = logi.predict(X_reshaped)
  #     # Display the resulting frame
  #     cv2.imshow('frame', rgb_im)
  #     cv2.putText(frame, str(pred))
  #     # Capture finishing with q
  #     if cv2.waitKey(1) == ord('q'):
  #         break
  # # When everything done, release the capture

  # cap.release()
  # cv2.destroyAllWindows()






