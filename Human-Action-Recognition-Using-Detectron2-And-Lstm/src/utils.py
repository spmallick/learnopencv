import cv2

WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 255, 0)


def draw_line(image, p1, p2, color):
    cv2.line(image, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)


def find_person_indicies(scores):
    return [i for i, s in enumerate(scores) if s > 0.9]


def filter_persons(outputs):
    persons = {}
    p_indicies = find_person_indicies(outputs["instances"].scores)
    for x in p_indicies:
        desired_kp = outputs["instances"].pred_keypoints[x][:].to("cpu")
        persons[x] = desired_kp
    return (persons, p_indicies)


def draw_keypoints(person, img):
    l_eye = person[1]
    r_eye = person[2]
    l_ear = person[3]
    r_ear = person[4]
    nose = person[0]
    l_shoulder = person[5]
    r_shoulder = person[6]
    l_elbow = person[7]
    r_elbow = person[8]
    l_wrist = person[9]
    r_wrist = person[10]
    l_hip = person[11]
    r_hip = person[12]
    l_knee = person[13]
    r_knee = person[14]
    l_ankle = person[15]
    r_ankle = person[16]

    draw_line(img, (l_shoulder[0], l_shoulder[1]),
              (l_elbow[0], l_elbow[1]), GREEN_COLOR)
    draw_line(img, (l_elbow[0], l_elbow[1]),
              (l_wrist[0], l_wrist[1]), GREEN_COLOR)
    draw_line(img, (l_shoulder[0], l_shoulder[1]),
              (r_shoulder[0], r_shoulder[1]), GREEN_COLOR)
    draw_line(img, (l_shoulder[0], l_shoulder[1]),
              (l_hip[0], l_hip[1]), GREEN_COLOR)
    draw_line(img, (r_shoulder[0], r_shoulder[1]),
              (r_hip[0], r_hip[1]), GREEN_COLOR)
    draw_line(img, (r_shoulder[0], r_shoulder[1]),
              (r_elbow[0], r_elbow[1]), GREEN_COLOR)
    draw_line(img, (r_elbow[0], r_elbow[1]),
              (r_wrist[0], r_wrist[1]), GREEN_COLOR)
    draw_line(img, (l_hip[0], l_hip[1]), (r_hip[0], r_hip[1]), GREEN_COLOR)
    draw_line(img, (l_hip[0], l_hip[1]), (l_knee[0], l_knee[1]), GREEN_COLOR)
    draw_line(img, (l_knee[0], l_knee[1]),
              (l_ankle[0], l_ankle[1]), GREEN_COLOR)
    draw_line(img, (r_hip[0], r_hip[1]), (r_knee[0], r_knee[1]), GREEN_COLOR)
    draw_line(img, (r_knee[0], r_knee[1]),
              (r_ankle[0], r_ankle[1]), GREEN_COLOR)

    cv2.circle(img, (l_eye[0], l_eye[1]), 4, WHITE_COLOR, -1)
    cv2.circle(img, (r_eye[0], r_eye[1]), 4, WHITE_COLOR, -1)
    cv2.circle(img, (l_wrist[0], l_wrist[1]), 4, WHITE_COLOR, -1)
    cv2.circle(img, (r_wrist[0], r_wrist[1]), 4, WHITE_COLOR, -1)
    cv2.circle(img, (l_shoulder[0], l_shoulder[1]), 4, WHITE_COLOR, -1)
    cv2.circle(img, (r_shoulder[0], r_shoulder[1]), 4, WHITE_COLOR, -1)
    cv2.circle(img, (l_elbow[0], l_elbow[1]), 4, WHITE_COLOR, -1)
    cv2.circle(img, (r_elbow[0], r_elbow[1]), 4, WHITE_COLOR, -1)
    cv2.circle(img, (l_hip[0], l_hip[1]), 4, WHITE_COLOR, -1)
    cv2.circle(img, (r_hip[0], r_hip[1]), 4, WHITE_COLOR, -1)
    cv2.circle(img, (l_knee[0], l_knee[1]), 4, WHITE_COLOR, -1)
    cv2.circle(img, (r_knee[0], r_knee[1]), 4, WHITE_COLOR, -1)
    cv2.circle(img, (l_ankle[0], l_ankle[1]), 4, WHITE_COLOR, -1)
    cv2.circle(img, (r_ankle[0], r_ankle[1]), 4, WHITE_COLOR, -1)
