# IN THIS PROJECT WE ARE GOING TO PRESENT PREMISES ANALYSIS USING YOLO MODEL OF MACHINE LEARNING WITH OBJECT DETECTION
# SO HERE ARE THE FILES THAT YOU NEED TO UNDERSTAND THE CONCEPT OF YOLO AND OPEN CV WITH TRAINED DATA
# HERE ARE THE REQUIRED FILES.... HOPE YOU LIKE THIS AND OUR EFFORTS
#THANK YOU

from ultralytics import YOLO
import cv2


# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
video_path = '../hackathon/persons.mp4'
cap = cv2.VideoCapture(video_path)

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:

        # detect objects
        # track objects
        results = model.track(frame, persist=True)

        # plot results
        # cv2.rectangle
        # cv2.putText
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break