
import os
import cv2
import numpy as np
import argparse
import time
import imutils
import dlib
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from keras.models import load_model
from pygame import mixer

# Initialize sound alarm
from pygame import mixer


mixer.init()
sound = mixer.Sound(r"C:\Users\julah\Downloads\1\Driver_Drowsiness_Detection-main\alarm.wav")


# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(r"C:\Users\julah\Downloads\1\Driver_Drowsiness_Detection-main\haar cascade files\haarcascade_frontalface_alt.xml")
leye_cascade = cv2.CascadeClassifier(r"C:\Users\julah\Downloads\1\Driver_Drowsiness_Detection-main\haar cascade files\haarcascade_lefteye_2splits.xml")
reye_cascade = cv2.CascadeClassifier(r"C:\Users\julah\Downloads\1\Driver_Drowsiness_Detection-main\haar cascade files\haarcascade_righteye_2splits.xml")

# Load pre-trained model for eye state prediction
model = load_model(r"C:\Users\julah\OneDrive\Desktop\major project\inspection_v3.h5")

# Function to trigger the alarm
def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        print('call')
        s = 'espeak "' + msg + '"'
        os.system(s)

    if alarm_status2:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False

# Eye Aspect Ratio Calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Calculate final EAR for both eyes
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

# Lip distance for yawning detection
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier(r"C:\Users\julah\Downloads\2\Yawn-detection-using-machine-learning-main\haarcascade_frontalface_default.xml")  # Faster but less accurate
predictor = dlib.shape_predictor(r"C:\Users\julah\Downloads\2\Yawn-detection-using-machine-learning-main\shape_predictor_68_face_landmarks.dat")

print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# Initialize score variables for eye state model
score = 0
thickness = 2
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
lbl = ['Close', 'Open']
# Initialize variables for eye state
val1 = 0  # Default value for right eye (0: closed, 1: open)
val2 = 0  # Default value for left eye (0: closed, 1: open)
# Initialize variables for eye state
val1 = 0  # Default value for right eye (0: closed, 1: open)
val2 = 0  # Default value for left eye (0: closed, 1: open)

# Main loop
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get the height and width of the frame
    height, width = frame.shape[:2]  # Extract the dimensions of the frame

    faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye_cascade.detectMultiScale(gray)
    right_eye = reye_cascade.detectMultiScale(gray)

    # Detect faces and eyes
    for (x, y, w, h) in faces:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        # Check for drowsiness
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if alarm_status == False:
                    alarm_status = True
                    t = Thread(target=alarm, args=('wake up sir',))
                    t.deamon = True
                    t.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            alarm_status = False

        # Check for yawning
        if distance > YAWN_THRESH:
            cv2.putText(frame, "Yawn Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if alarm_status2 == False and saying == False:
                alarm_status2 = True
                t = Thread(target=alarm, args=('take some fresh air sir',))
                t.deamon = True
                t.start()
        else:
            alarm_status2 = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Process right eye for state prediction
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (52, 52))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(52, 52, 1)
        rpred = model.predict(np.expand_dims(r_eye, axis=0))
        val1 = 1 if rpred[0][1] > rpred[0][0] else 0
        break

    # Process left eye for state prediction
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (52, 52))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(52, 52, 1)
        lpred = model.predict(np.expand_dims(l_eye, axis=0))
        val2 = 1 if lpred[0][1] > lpred[0][0] else 0
        break

    # Eye state prediction logic
    if val1 == 0 and val2 == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Trigger alarm if score exceeds threshold
    if score > 15:
        cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame)
        try:
            sound.play()
        except Exception as e:
            print(str(e))
        score = 0

    cv2.imshow("Frame", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()


