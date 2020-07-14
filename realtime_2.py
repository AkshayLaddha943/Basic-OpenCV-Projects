from PIL import Image, ImageDraw
import face_recognition
#import argparse
import cv2
import numpy as np
import face_recognition

#ap = argparse.ArgumentParser()
#ap.add_argument("-i","--image",required = True, help = "Path to image")
#args = vars(ap.parse_args())

cap = cv2.VideoCapture(0)

rit_image = face_recognition.load_image_file("ritz.jpg")
rit_face_encod = face_recognition.face_encodings(rit_image)[0]

#rit_image = face_recognition.load_image_file("rit.jpg")
#rit_face_encod = face_recognition.face_encodings(rit_image)[0]

known_face_encodings = [rit_face_encod]
known_face_names = ["Ritika Laddha"]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


while True:
	
	ret, frame = cap.read()

	#Resizing frame of video for faster face recognition
	small_frame = cv2.resize(frame, (0, 0) , fx = 0.25, fy = 0.25)

	#Convert img from BGR (which openCV uses) to RGB (for face_recog)
	rgb_small_frame = small_frame[:, :, ::-1]

	if process_this_frame:
		#Find all faces and face_encodings in current frame
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

		face_names = []
		for face_encoding in face_encodings:
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
			name = "Unknown"

			if True in matches:
				first_match_index = matches.index(True)
				name = known_face_names[first_match_index]

			face_names.append(name)

	process_this_frame = not process_this_frame

	for(top, right, bottom, left), name in zip(face_locations,face_names):
		top *= 4
		right *= 4
		bottom *= 4
		left *= 4

		cv2.rectangle(frame, (left, top), (right, bottom),(0, 0, 255), 2)

		cv2.rectangle(frame, (left, bottom - 35), (right, bottom),(0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

	cv2.imshow('Video', frame)

	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()