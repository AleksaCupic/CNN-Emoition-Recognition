import numpy as np
import cv2
import matplotlib.image as mpimg
import os


def cropImage():
    image = cv2.imread('{folder_path}/{image.jpg}')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
    
    print("[INFO] Found {0} Faces.".format(len(faces)))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        print("[INFO] Object found. Saving locally.")
        cv2.imwrite('{folder_for_detected_faces}/{image.jpg}', roi_color)



#May need to be deleted

def augumentImage():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    my_test_datagen  = ImageDataGenerator( rescale = 1.0/255, )
        
    my_test_generator=my_test_datagen.flow_from_directory('{folder_where_you_stored_detected_face}',
                                                        class_mode  = 'categorical',
                                                        #color_mode='grayscale',
                                                        target_size = (64, 64),
                                                        batch_size=128)
    return my_test_generator

def displayImage():
    import matplotlib.pyplot as plt

    from tensorflow.keras import models
    my_test_generator=augumentImage()
    model_kaggle = models.load_model('clean_kaggle.h5')
    y_img_batch, y_class_batch = my_test_generator[0]
    y_pred = np.argmax(model_kaggle.predict(y_img_batch),-1)
    
    
    x_batch, y_batch = my_test_generator[0]
    
    emotions=['Angry','Happy','Sad','Surprised']
    
    for i in range (0,1):
        plt.title(emotions[y_pred[i]])
        #image = x_batch[i]
        img = mpimg.imread('{folder_with_detected_face}/{image.jpg}')
        plt.imshow(img)
        plt.show()





#Start video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)

while cap.isOpened():
    flags, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Emotion Classifier',gray)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cv2.imwrite('{folder_path}/{image.jpg}',frame)
        cap.release()
        cv2.destroyAllWindows()
        cropImage()
        displayImage()
        break

