 # IN THIS PROJECT WE ARE GOING TO PRESENT PREMISES ANALYSIS USING YOLO MODEL OF MACHINE LEARNING WITH OBJECT DETECTION
# SO HERE ARE THE FILES THAT YOU NEED TO UNDERSTAND THE CONCEPT OF YOLO AND OPEN CV WITH TRAINED DATA
# HERE ARE THE REQUIRED FILES.... HOPE YOU LIKE THIS AND OUR EFFORTS
#THANK YOU

import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt

#read image
image_path = 'stop.jpg'

img = cv2.imread(image_path)

#instance text detector
reader = easyocr.Reader(['en'], gpu=True)

#detect text on image
text_ = reader.readtext(img)

threshold = 0.25
#draw bbox and text
for t_,t in enumerate(text_):
    print(t)

    bbox, text, score = t
    if score>threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0,255,0), 5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,255,0), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
