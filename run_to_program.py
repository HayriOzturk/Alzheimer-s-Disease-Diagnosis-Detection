###############################################################
#      --- Testing With New Data And Showing Result ---       #
#                                                             #
###############################################################

import numpy as np
import cv2
import keras
import tensorflow as tf
import os
import random
from keras.layers import Conv2D, MaxPooling2D ,Dense, Flatten,Dropout
from keras.models import Sequential,load_model
from matplotlib import pyplot as plt


############################## Random Classes For Test ##############################
testdata = ["mild","moderate","non","verymild"]
randomtype = random.choice(testdata)
randomnumber =str (random.randint(1,60))

############################## Loading Model ##############################
new_model = load_model(os.path.join('model','cnnforalzheimer.h5'))



############################## Getting Random Test Data ##############################
filenames = ('testdata/'+randomtype+'_'+randomnumber+'.jpg')
print(filenames)
img = cv2.imread(filenames)
resize_img = tf.image.resize(img,(256,256))

############################## Prediction Of Model ##############################
result = new_model.predict(np.expand_dims(resize_img/255,0))

print(result)
print(result.argmax())

############################## Printing Prediction Of Model ##############################
if result.argmax() == 0:
    prediction = "Prediction is mild demented"
    
elif result.argmax() == 1:
    prediction = "Prediction is moderate demented" 
    
elif result.argmax() == 2:
    prediction = "Prediction is non demented" 
    
else:
    prediction = "Prediction is verymild demented"
    

 ############################## Showing Prediction And True Diagnosis ##############################
plt.title("Test Brain MRI image"+"\nResemblance: %"+ str(result.max()*100))
plt.xlabel(prediction+"\nTrue diagnosis is "+randomtype+" demented")
plt.imshow(img)
plt.show()
   