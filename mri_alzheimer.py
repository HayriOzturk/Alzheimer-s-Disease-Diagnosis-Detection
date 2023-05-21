###############################################################
#      --- Getting Data, Building CNN Architecture   ---      #
#                                                             #
###############################################################


############################## Import Libraries ##############################
import numpy as np
import cv2
import keras
import tensorflow as tf
import os
from keras.layers import Conv2D, MaxPooling2D ,Dense, Flatten,Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.metrics import Precision, Recall

############################## Using GPU ##############################
ourgpu = tf.config.experimental.list_physical_devices('GPU')
for gpu in ourgpu:
    tf.config.experimental.set_memory_growth(gpu,True)

############################## Getting Dataset From Local Disk ##############################
dataset = tf.keras.utils.image_dataset_from_directory("dataset_MRI",label_mode='categorical')
dataset = dataset.map(lambda x,y: (x/255,y))
scdataset_iterator = dataset.as_numpy_iterator()
batch = scdataset_iterator.next()
# print(batch[1])
# fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx, img in enumerate(batch[0][:4]):
#    ax[idx].imshow(img)
#    ax[idx].title.set_text(batch[1][idx])
# plt.show()
#print(len(dataset))
############################## Splitting Dataset For Train-Test-Validation ##############################
train_number = int(len (dataset)*.7)
val_number = int(len (dataset)*.2)
test_number = int(len (dataset)*.1)+1
#print(train_number+val_number+test_number)

train = dataset.take(train_number)
val = dataset.skip(train_number).take(val_number)
test = dataset.skip(train_number+val_number).take(val_number)

############################## Building CNN Architecture ##############################
cnn_model = tf.keras.Sequential()

cnn_model.add(Conv2D(16,(2,2),1,activation='relu',input_shape=(256,256,3)))
cnn_model.add(MaxPooling2D())

cnn_model.add(Conv2D(32,(2,2),1,activation='relu'))
cnn_model.add(MaxPooling2D())

cnn_model.add(Conv2D(64,(2,2),1,activation='relu'))
cnn_model.add(MaxPooling2D())

cnn_model.add(Conv2D(32,(2,2),1,activation='relu'))
cnn_model.add(MaxPooling2D())

cnn_model.add(Conv2D(16,(2,2),1,activation='relu'))
cnn_model.add(MaxPooling2D())

cnn_model.add(Flatten())
cnn_model.add(Dense(2048, activation='relu'))

cnn_model.add(Dense(4, activation='softmax'))

############################## Compiling ##############################
cnn_model.compile('adam', loss='categorical_crossentropy',metrics=['accuracy'])

############################## Log Files ##############################
logdir = 'log_dir'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

############################## Fitting ##############################
fit_model = cnn_model.fit(train,epochs=200,validation_data=val,callbacks=[tensorboard_callback])

############################## Creating Loss and Accuracy Graphics ##############################
fig = plt.figure()
plt.plot(fit_model.history['loss'], color='teal', label='loss')
plt.plot(fit_model.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig2 = plt.figure()
plt.plot(fit_model.history['accuracy'], color='teal', label='accuracy')
plt.plot(fit_model.history['val_accuracy'], color='orange', label='val_accuracy')
fig2.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

############################## Getting Precision-Recall-F1_Score Values ##############################
pre = Precision()
re = Recall()

for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = cnn_model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
print(pre.result(), re.result())

a = pre.result()
b = re.result()

F1 = 2 * (a * b) / (a + b)
print(F1)

############################## Saving Model ##############################
cnn_model.save(os.path.join('model','cnnforalzheimer.h5'))

cv2.waitKey(0)