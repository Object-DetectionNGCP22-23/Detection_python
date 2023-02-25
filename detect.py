import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils


def run() -> None:
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	
	#CHANGE NAME OF FILE YOU'RE USING HERE AND SET USE_CORAL TO TRUE IF USING EDGETPU LINE 16
	base_options = core.BaseOptions(file_name="android_edgetpu.tflite", use_coral=True, num_threads=4)
	detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)
	options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
	detector = vision.ObjectDetector.create_from_options(options)

	while cap.isOpened():
		success, image = cap.read()

		input_tensor = vision.TensorImage.create_from_array(image)
		detection_result = detector.detect(input_tensor)
		#image = utils.visualize(image, detection_result)
		numObj = len(detection_result.detections)
		for i  in range(numObj):
			label_halfx = detection_result.detections[i].bounding_box.width//2
			label_halfy = detection_result.detections[i].bounding_box.height//2
			label_x = label_halfx + detection_result.detections[i].bounding_box.origin_x
			label_y = label_halfy + detection_result.detections[i].bounding_box.origin_y

			# Format xmin, xmax, ymin, ymax, and class ground truth labeled data for DisNET.tflite 
			# put that ^ data into a numpy array and DisNET.predict(np.array), and DONE
			 
			cv2.putText(image, "L", (label_x,label_y), cv2.FONT_HERSHEY_PLAIN,10,(0,0,255),10)
			# In this ^ add a distance: ‘12.3456789 inches’ statement

		if cv2.waitKey(1) == 27:
			break
		cv2.imshow('object_detector', image)

	cap.release()
	cv2.destroyAllWindows()

run()

