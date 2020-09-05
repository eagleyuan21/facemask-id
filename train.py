import cv2, os
import numpy as np
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

data_path = '../../Downloads/dataset' # Path here
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]

label_dict = dict(zip(categories,labels))

img_size = 100
data = []
target = []

for category in categories:
    folder_path = os.path.join(data_path,category)
    img_names = os.listdir(folder_path)
        
    for img_name in img_names:
        img_path = os.path.join(folder_path,img_name)
        img = cv2.imread(img_path)

        try:
            #image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)         
            #Coverting the image into gray scale
            image = img
            #keep in color

            resized = cv2.resize(image,(img_size,img_size))
            #resizing, since we need a fixed common size for all the images in the dataset
            data.append(resized)
            target.append(label_dict[category])
            #appending the image and the label(categorized) into the list (dataset)

        except Exception as e:
            print('Exception:',e)

data = np.array(data)/255.0
data = np.reshape(data,(data.shape[0],img_size,img_size,3))
target = np.array(target)

new_target = np_utils.to_categorical(target)

model = Sequential()

model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(50,activation='relu'))
#Dense layer of neurons
model.add(Dense(2,activation='softmax'))
#The Final layer with two outputs for two categories

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

train_data,test_data,train_target,test_target=train_test_split(data,new_target,test_size=0.1)

checkpoint = ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history = model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)

model.save("model-final.model")