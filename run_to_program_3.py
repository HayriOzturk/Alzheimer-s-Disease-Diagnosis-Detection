###############################################################
# --- Testing With New Data And Saving Them To A Txt File --- #
#            --- With TP-FP-TN-FN Values ---                  #
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


falsenumber = 0
count = 0
main_folder_path = 'testdata'
image_filenames = []
new_model = load_model(os.path.join('model','cnnforalzheimer.h5'))

tp = 0  # true positive
fp = 0  # false positive
fn = 0  # false negative
tn = 0  # true negative

for root, dirs, files in os.walk(main_folder_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            image_filenames.append(os.path.join(root, file))



for file in image_filenames:
    img = cv2.imread(file)
    resize_img = tf.image.resize(img,(256,256))
    result = new_model.predict(np.expand_dims(resize_img/255,0))
    truedia = file.split('\\')[1]
    truediagnosis = str(truedia)
    count+=1


    if result.argmax() == 0:
        prediction = "Mild"
        if prediction == truediagnosis:
            tp += 1
        else:
            fp += 1
        
    elif result.argmax() == 1:
        prediction = "Moderate"
        if prediction == truediagnosis:
            tp += 1
        else:
            fp += 1
        
    elif result.argmax() == 2:
        prediction = "Non" 
        if prediction == truediagnosis:
            tn += 1
        else:
            fn += 1
        
    else:
        prediction = "VeryMild"
        if prediction == truediagnosis:
            tp += 1
        else:
            fp += 1

    if prediction != truediagnosis:
        falsenumber+= 1
        with open("result.txt", "a") as f:    
            f.write("Prediction is "+prediction+" True diagnosis is "+truediagnosis+"          Resemblance: %"+ str(result.max()*100)+"!!!!!!!!!!!!!!!!!!!!"+"\n")
   
    else:
        with open("result.txt", "a") as f:   
            f.write("Prediction is "+prediction+" True diagnosis is "+truediagnosis+"          Resemblance: %"+ str(result.max()*100)+"\n")


recall = tp / (tp + fn)
precision = tp / (tp + fp)
accuracy = (tp + tn) / (tp + tn + fp + fn)
f1 = (2*precision*recall)/(precision+recall)

strfalsenumber = str(falsenumber) 
strcount = str(count)
strrecall = str(recall)
strprecision = str(precision)
straccuracy = str(accuracy)
strf1 = str(f1)
with open("result.txt", "a") as f:    
        f.write("Number of total false: "+strfalsenumber+"          "+"Number of all test images: "+strcount+"\n"+"Recall: "+strrecall+
        "   Precision: "+strprecision+"   Accuracy: "+straccuracy+"   F1 Score: "+strf1)

print(tp)
print(tn)
print(fp)
print(fn)
print("Recall:"+strrecall)
print("Precision:"+strprecision)
print("Accuracy:"+straccuracy)
print("F1 Score:"+strf1)