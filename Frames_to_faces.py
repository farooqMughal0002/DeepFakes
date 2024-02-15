import cv2
import os
path = "G:/test/*.*"
#img = cv2.imread('G:/frame/fake/0.jpg') #Path of an image
#faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img_number=1

img_list=glob.glob(path)
for file in img_list[0:5]:
    print(file)
    img=cv2.imread(file,1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(img,1.3,5)
    try:
        for (x,y,w,h) in faces:
            roi_color=img[y:y+h,x:x+w]
        resized=cv2.resize(roi_color, (128,128))
        cv2.imwrite("G:/"+str(img_number)+".jpg",resized)
    except:
        print("No faces")
    img_number+=1

#directory = os.getcwd()+r''
