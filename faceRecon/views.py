from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.models import User
from .forms import UserSelection

import cv2
import numpy as np
import os
from PIL import Image

BASE_DIR = getattr(settings, 'BASE_DIR')

def index(request):
    context = {'forms': UserSelection }
    return render(request, 'index.html', context)

def create_dataset(request):
    if request.method == "POST":
        face_id = int(request.POST['selected_user'])
        #print("Face ID->", face_id, type(face_id))

        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height
        face_detector = cv2.CascadeClassifier(BASE_DIR + '/ml/haarcascade_frontalface_default.xml')  # For each person, enter one numeric face id
        print("[INFO] Initializing face capture. Look the camera and wait ...")  # Initialize individual sampling face count
        count = 0
        while (True):
            ret, img = cam.read()
            # img = cv2.flip(img, -1) # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            # Skip the process if multiple faces detected:
            if len(faces) == 1:
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    count += 1
                    # Save the captured image into the datasets folder
                    cv2.imwrite(BASE_DIR+"/ml/dataset/User." + str(face_id) + '.' +
                                str(count) + ".jpg", gray[y:y + h, x:x + w])
                    cv2.waitKey(250)

                cv2.imshow('Face', img)
                k = cv2.waitKey(1) & 0xff  # Press 'ESC' for exiting video
                if k == 27:
                    break
                elif count >= 30:  # Take 30 face sample and stop video
                    break  # Do a bit of cleanup
                print(count)
            else:
                print("\n multiple faces detected")

        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()

        messages.success(request, 'Face successfully registered.')

    else:
        print("Its a GET method.")

    return redirect('/')

def detect(request):
    faceDetect = cv2.CascadeClassifier(BASE_DIR + '/ml/haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)
    # creating recognizer
    rec = cv2.face.LBPHFaceRecognizer_create();
    # loading the training data
    rec.read(BASE_DIR + '/ml/recognizer/trainer.yml')
    getId = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    userId = 0
    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            getId, conf = rec.predict(gray[y:y + h, x:x + w])  # This will predict the id of the face
            print(getId, conf)
            confidence = "  {0}%".format(round(100 - conf))
            # print conf;
            if conf < 35:
                try:
                    user = User.objects.get(id=getId)
                except  User.DoesNotExist:
                    pass

                print("User Name", user.username)

                userId = getId
                if user.username:
                    cv2.putText(img, user.username, (x+5, y+h-10), font, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(img, "Detected", (x, y + h), font, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Unknown", (x, y + h), font, 1, (0, 0, 255), 2)

            cv2.putText(img, str(confidence), (x + 5, y - 5), font, 1, (255, 255, 0), 1)
            # Printing that number below the face
            # @Prams cam image, id, location,font style, color, stroke

        cv2.imshow("Face", img)
        if (cv2.waitKey(1) == ord('q')):
            break
        #elif (userId != 0):
        #    cv2.waitKey(1000)
        #    cam.release()
        #    cv2.destroyAllWindows()
        #    return redirect('/records/details/' + str(userId))

    cam.release()
    cv2.destroyAllWindows()
    return redirect('/')

def trainer(request):
    '''
        In trainer.py we have to get all the samples from the dataset folder,
        for the trainer to recognize which id number is for which face.

        for that we need to extract all the relative path
        i.e. dataset/user.1.1.jpg, dataset/user.1.2.jpg, dataset/user.1.3.jpg
        for this python has a library called os
    '''


    # Path for face image database
    path = BASE_DIR + '/ml/dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(BASE_DIR+"/ml/haarcascade_frontalface_default.xml");  # function to get the images and label data

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')  # grayscale
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            #faces = detector.detectMultiScale(img_numpy)
            #for (x, y, w, h) in faces:
            #    faceSamples.append(img_numpy[y:y + h, x:x + w])
            #    ids.append(id)
            faceSamples.append(img_numpy)
            ids.append(id)
            # print ID
            cv2.imshow("training", img_numpy)
            cv2.waitKey(10)
        return np.array(faceSamples), np.array(ids)
        #return faceSamples, ids

    print("[INFO] Training faces. It will take a few seconds. Wait ...")

    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, ids)  # Save the model into trainer/trainer.yml
    recognizer.save(BASE_DIR+'/ml/recognizer/trainer.yml')  # Print the numer of faces trained and end program
    print("[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    cv2.destroyAllWindows()
    messages.success(request, "{0} faces trained successfully".format(len(np.unique(ids))) )

    return redirect('/')