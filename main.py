from keras.models import load_model
import cv2
import numpy as np

model = load_model('new200px.model') # Model here

face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

source = cv2.VideoCapture(0)

labels_dict = {1:'No Mask', 0:'Mask'}
color_dict = {1:(0,0,255), 0:(0,255,0)}
img_size = 100

while(True):

    ptr,img = source.read()
    image = img
    faces = face_clsfr.detectMultiScale(image,1.3,5)  

    for (x,y,w,h) in faces:
    
        face_img = image[y:y+w,x:x+w]
        resized = cv2.resize(face_img,(img_size,img_size))
        normalized = resized/255.0
        reshaped = np.reshape(normalized,(1,img_size,img_size,3))
        result = model.predict(reshaped)

        label = np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key = cv2.waitKey(1)
    
    if(key == 113):
        break
        
cv2.destroyAllWindows()
source.release()