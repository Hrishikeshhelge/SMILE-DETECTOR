# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:32:13 2021

@author: Hrishi_rich
"""
# import computer vision library
import cv2


# Face & Smile Classifier
face_detector = cv2.CascadeClassifier('frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')


# Grab Webcam
webcam = cv2.VideoCapture(0) #'Why_so_serious.mp4'


# Show the current frame
while True:
    
    # Read current frame from webcam
    successful_frame_read, frame = webcam.read()
    
    # If there's an error abort
    if not successful_frame_read:
        break
    
    # Change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_detector.detectMultiScale(frame_grayscale)
        
    
    # Run face detection within each of those faces
    for (x, y, w, h) in faces:
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 2)
        
        # Get the sub frame (using numpy N-dimensional array slicing)
        the_face = frame[y:y+h, x:x+w]
    
        # Change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)                
  
        smile = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)    

        # Lable this faces as smiling
        if len(smile) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))


# Show the current frame
cv2.imshow('Smile Detector', frame)


# Display 
cv2.waitkey(1)


#cleanup
webcam.release()
cv2.destroyAllWindows()







                         
print("code complete")