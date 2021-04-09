from cv2 import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)#prints list of students
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')#to read our current image
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])#to get the names without the file format(.jpg)
print(classNames)
#encoding function
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]  # encodes the image
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')#separate the line at comma
            nameList.append(entry[0])#entry[0] i.e the name would be appended to the list
        if name not in nameList: #if name is not present in list(used to avoid repetition)
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(images)
#print(len(encodeListKnown))
print('Encoding Complete')

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25) #reduces the size of image
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)  # detects the faces and gives them location
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)# encodes the image of webcam
    #iterates from all the faces and finds the face that we require in case of multiple faces
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        # compares these lists and returns one with minimum difference
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)#returns index with minimum values
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)


    cv2.imshow('Webcam',img) #shows the image in webcame
    cv2.waitKey(1)

"""
faceLoc = face_recognition.face_locations(imgElon)[0] #detects the faces

cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#print(faceLoc) #gives 4 values-top,right,bottom and left

faceLocTest = face_recognition.face_locations(imgTest)[0] #detects the faces
encodeTest = face_recognition.face_encodings(imgTest)[0] #encodes the image
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodeTest)
"""