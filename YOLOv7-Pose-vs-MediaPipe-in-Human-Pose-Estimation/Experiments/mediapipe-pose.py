import cv2
import time
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5)
videos = ['dance', 
        'dark', 
        'far-away', 
        'occlusion-example', 
        'skydiving', 
        'yoga-1']

for i in range(len(videos)):
	file_name = videos[i] + '.mp4'
	path = 'Media/New/' + file_name

	cap = cv2.VideoCapture(path)
	fps = int(cap.get(cv2.CAP_PROP_FPS))

	ret, frame = cap.read()
	h,w,c = frame.shape

	out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

	while True:
		ret, frame = cap.read()

		if not ret:
			print('Can\'t read frames. Exiting..')
			break

		# cv2.imshow('Original Image', frame)

		img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		t1 = time.time()
		results = pose.process(img)
		# print(dir(results))
		print(dir(results.index.__sizeof__))
		t2 = time.time()

		fps_org = 1/(t2 - t1)

		ann_img = frame.copy()

		mp_drawing.draw_landmarks(ann_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
			landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
			connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 200, 0), 
			thickness=3, 
			circle_radius=3))

		cv2.putText(ann_img, 'FPS : {:.2f}'.format(fps_org), (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 8)
		cv2.putText(ann_img, 'MediaPipe', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 8)

		out.write(ann_img)

		cv2.imshow('Annotated Image', ann_img)
		key = cv2.waitKey(1)

		if key == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
