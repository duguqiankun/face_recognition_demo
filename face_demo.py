import os
import sys
sys.path.append('./face_detect/')
from detector import Retinaface_Detector
import cv2
import time
import numpy as np

detector=Retinaface_Detector(use_gpu=True)

cap = cv2.VideoCapture(0)
ret=True
while(ret):
	# Capture frame-by-frame
	ret, frame = cap.read()
	if frame is not None:
		t0=time.time()


		results= detector.detect(frame)

		print(results)
		print(time.time()-t0)

		for result in results:
			face = result[0]
			landmark = result[1]

			color = (0, 0, 255)
			cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), color, 2)

			for l in range(landmark.shape[0]):
				color = (0, 0, 255)
				if l == 0 or l == 3:
					color = (0, 255, 0)
				cv2.circle(frame, (landmark[l][0], landmark[l][1]), 1, color, 2)
		# Display the resulting frame
		cv2.imshow('frame',cv2.resize(frame,(1280,960)))
		print(frame.shape)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
print('finish')
cap.release()
cv2.destroyAllWindows()
