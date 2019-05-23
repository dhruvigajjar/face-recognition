import dlib
import cv2
from align_dlib import *

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "E:/SEM 2/ML/shape_predictor_68_face_landmarks.dat"

# Take the image file name from the command line
file_name = "E:/SEM 2/ML/Face Detection Data/Dataset/1641038_Female_Sad_RGB.png"

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = AlignDlib(predictor_model)
win = dlib.image_window()
'''
# Take the image file name from the command line
file_name = sys.argv[1]
'''
# Load the image
image = cv2.imread(file_name)

# Run the HOG face detector on the image data
detected_faces = face_detector(image, 1)

print("Found {} faces in the image file {}".format(len(detected_faces), file_name))
win.set_image(image)
# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):

	# Detected faces are returned as an object with the coordinates 
	# of the top, left, right and bottom edges
	print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

	# Get the the face's pose
	pose_landmarks = face_pose_predictor(image, face_rect)

	# Use openface to calculate and perform the face alignment
	alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices= AlignDlib.OUTER_EYES_AND_NOSE)

	# Save the aligned image to a file
	cv2.imwrite("E:/SEM 2/ML/Face Detection Data/Dataset/1641038_Female_Sad_RGB_aligned_face.jpg".format(i), alignedFace)
    