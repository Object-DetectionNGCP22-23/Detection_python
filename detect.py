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
	detection_options = processor.DetectionOptions(max_results=5, score_threshold=0.3)
	options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
	detector = vision.ObjectDetector.create_from_options(options)
	
	#declare visualization parameters
	# Variables to calculate FPS
	counter, fps = 0, 0
	start_time = time.time()
	row_size = 20  # pixels
	left_margin = 24  # pixels
	text_color = (0, 0, 255)  # red
	font_size = 1
	font_thickness = 1
	fps_avg_frame_count = 10

	while cap.isOpened():
		success, image = cap.read()
		counter += 1
		image = cv2.flip(image, 1)

		# Convert the image from BGR to RGB as required by the TFLite model.
		rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Create a TensorImage object from the RGB image.
		input_tensor = vision.TensorImage.create_from_array(rgb_image)

		# Run object detection estimation using the model.
		detection_result = detector.detect(input_tensor)

		# Draw keypoints and edges on input image
		image = utils.visualize(image, detection_result)

		# Calculate the FPS
		if counter % fps_avg_frame_count == 0:
		  end_time = time.time()
		  fps = fps_avg_frame_count / (end_time - start_time)
		  start_time = time.time()

		# Show the FPS
		fps_text = 'FPS = {:.1f}'.format(fps)
		text_location = (left_margin, row_size)
		cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
					font_size, text_color, font_thickness)

		# Stop the program if the ESC key is pressed.
		if cv2.waitKey(1) == 27:
		  break
		cv2.imshow('object_detector', image)

	cap.release()
	cv2.destroyAllWindows()

run()
