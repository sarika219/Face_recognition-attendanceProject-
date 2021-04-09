from cv2 import cv2
import numpy as np
import face_recognition
import os
#step 1: Loading the images and converting them into RGB(we recieve images as bgr but the library understands only RGB format)
imgElon = face_recognition.load_image_file('ImagesBasic/elonMusk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/billgates.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
#step 2:Finding faces in our image and its encodings
faceLoc = face_recognition.face_locations(imgElon)[0] #detects the faces
encodeElon = face_recognition.face_encodings(imgElon)[0] #encodes the image
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#print(faceLoc) #gives 4 values-top,right,bottom and left

faceLocTest = face_recognition.face_locations(imgTest)[0] #detects the faces
encodeTest = face_recognition.face_encodings(imgTest)[0] #encodes the image
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)#if faces have similar properties then lower this value more is accuracy
#faceDis is an array
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)#we want first element of faceDis

cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)