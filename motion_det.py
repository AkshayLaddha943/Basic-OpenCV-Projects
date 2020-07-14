import cv2 
import numpy as np
import imutils
from imutils.video import VideoStream
import datetime
import time

vs = VideoStream(src=0).start()
time.sleep(2.0)

firstframe = None

while True:
	#Grab current frame and initialize movement/no_movement text
	frame = vs.read()
	text = "No movement"

	#if frame is not caught, then end
	if frame is None:
		break

	#resizing the frame, convert to grayscale, blur
	frame = imutils.resize(frame, width = 500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21,21), 0)

	#if first frame is None, initialize it
	if firstframe is None:
		firstframe = gray
		continue

	#Computing difference between current frame and next frame
	#Lower the threshold value, higher the sensitivity of the model to capture motion & vice-versa
	framedelta = cv2.absdiff(firstframe, gray)
	thresh = cv2.threshold(framedelta, 130, 255, cv2.THRESH_BINARY)[1]

	#dilate threshold image to fill in holes, find Contours
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts) 

	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x,y), (x + w, y + h), (255, 0, 0), 2)
		text = "Movement Spotted"

	cv2.putText(frame,"Status: {}".format(text), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 255, 25), 2)

	
	
	cv2.imshow("Frame Delta view", framedelta)
	cv2.imshow("Threshold view", thresh)
	cv2.imshow("Room view", frame)

	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

vs.stop()
cv2.destroyAllWindows()

