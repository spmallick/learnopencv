import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

keypnt_68 = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
             285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385, 387,
             466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14, 178,
             162, 54, 67, 10, 297, 284, 389]

# For static images:
IMAGE_FILES = []
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            print('face_landmarks:', face_landmarks)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
        cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image1 = image.copy()
        # image2 = image.copy()
        if results.multi_face_landmarks:
            contour_point_idxs = []
            for point in mp_face_mesh.FACEMESH_CONTOURS:
                contour_point_idxs.append(point[0])
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    # if idx in contour_point_idxs:
                    if idx in keypnt_68:
                        # if True:
                        height, width = image.shape[:2]
                        point = (int(landmark.x * width), int(landmark.y * height))
                        cv2.circle(image, point, 1, (255, 0, 0), -1)
                        cv2.putText(image, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
                        # if idx % 2 == 0:
                        #     cv2.circle(image2, point, 1, (255, 0, 0), -1)
                        #     cv2.putText(image2, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
                        # elif (idx+1) % 2 == 0:
                        #     cv2.circle(image1, point, 1, (255, 0, 0), -1)
                        #     cv2.putText(image1, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)

        # Flip the image horizontally for a selfie-view display.
        # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        cv2.imshow('MediaPipe Face Mesh', image)
        # cv2.imshow('image2', image2)
        # cv2.imshow('image1', image1)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
