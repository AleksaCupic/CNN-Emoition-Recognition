import numpy as np
import cv2
import matplotlib.image as mpimg
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models

def cropImage():
    image = cv2.imread('Folder1/image.jpg')
    print("MY IMAGE")
    print(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))

    print("[INFO] Found {0} Faces.".format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        print("[INFO] Object found. Saving locally.")
        cv2.imwrite('detectedFaces/image.jpg', roi_color)
        cv2.imwrite('detectedFaces/images/image.jpg', roi_color)


def predictImage():

    print("Loading model....")
    model_kaggle = models.load_model('clean_kaggle.h5')


    # SCALE IMAGE AND PREDICT

    imasd=image.load_img('detectedFaces/image.jpg', target_size=(64,64))
    imgArr=image.img_to_array(imasd)/255.
    image1 = np.expand_dims(imgArr, axis=0)
    result=model_kaggle.predict(image1)
    print("Prediction: ")
    print(result)
    maxPred=np.argmax(result,-1)
    print(maxPred)


    emotions=['Angry','Happy','Sad','Surprised']


    plt.title(emotions[maxPred[0]])
    #image = x_batch[i]
    img = mpimg.imread('detectedFaces/image.jpg')
    plt.imshow(img)
    plt.show();




#Start video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)

while cap.isOpened():
    flags, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Emotion Classifier', gray)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cv2.imwrite('Folder1/image.jpg',frame)
        cap.release()
        cv2.destroyAllWindows()
        cropImage()
        predictImage()
        break
