import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import cv2
from tqdm import tqdm_notebook as tqdm
import sys
import os
import numpy as np
from PIL import Image

newmod = load_model('model.h5')

background = None

accumulated_weight = 0.5


roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600



def calc_accum_avg(frame, accumulated_weight):


    global background


    if background is None:
        background = frame.copy().astype("float")
        return None


    cv2.accumulateWeighted(frame, background, accumulated_weight)



def segment(frame, threshold=35):
    global background


    diff = cv2.absdiff(background.astype("uint8"), frame)

    _, thresholded = cv2.threshold(diff, threshold, 188, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment)



def thres_display(img):
    width = 64
    height = 64
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    test_img = image.img_to_array(resized)
    test_img = np.expand_dims(test_img, axis=0)
    result = newmod.predict(test_img)
    val = [index for index, value in enumerate(result[0]) if value == 1]
    return val


cam = cv2.VideoCapture(0)


num_frames = 0


while True:

    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame


    roi = frame[roi_top:roi_bottom, roi_right:roi_left]


    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)


    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "WAIT!", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

    else:

        cv2.putText(frame_copy, "Place your hand in side the box", (330, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)


        hand = segment(gray)

        if hand is not None:
            thresholded, hand_segment = hand

            fff = cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0), 1)


            cv2.imshow("Thresholded Image", thresholded)
            cv2.imshow("Thresholded Image", thresholded)


            directory = "LiveData/Sample02/"
            print(directory)
            imagecount = int(100)
            print(imagecount)

            filename = 0
            count = 0

            pbar = tqdm(total=imagecount + 1)
            while True and count < imagecount:
                filename += 1
                count += 1

                path = directory + "//" + str(filename) + ".jpg"
                cv2.imwrite(path,   thresholded)


            res = thres_display(thresholded)



    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 2)


    num_frames += 500


    cv2.imshow("Hand Gestures", frame_copy)






    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break


cam.release()
cv2.destroyAllWindows()