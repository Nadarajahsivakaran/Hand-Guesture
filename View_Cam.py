import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import cv2
from audioplayer import AudioPlayer

Vtype = input("Please Enter Area")
print("you have selected" , Vtype)



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



def segment(frame, threshold=40):
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
            cv2.putText(frame_copy, "WAIT!", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    else:

        cv2.putText(frame_copy, "Place your hand in side the box", (330, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)


        hand = segment(gray)


        if hand is not None:
            thresholded, hand_segment = hand


            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0), 1)



            cv2.imshow("Thresholded Image", thresholded)
            res = thres_display(thresholded)

            if len(res) == 0:
                cv2.putText(frame_copy, str('None'), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:

                x = str(res[0])
                if (x == "0"):
                    cv2.putText(frame_copy, "A", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/a.mp3").play(block=True)
                elif (x == "1"):
                    cv2.putText(frame_copy, "AA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/aa.mp3").play(block=True)
                elif (x == "2"):
                    cv2.putText(frame_copy, "AH", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/ah.mp3").play(block=True)
                elif (x == "3"):
                    cv2.putText(frame_copy, "CHA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Cha.mp3").play(block=True)
                elif (x == "4"):
                    cv2.putText(frame_copy, "DA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Da.mp3").play(block=True)
                elif (x == "5"):
                    cv2.putText(frame_copy, "E", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/E.mp3").play(block=True)
                elif (x == "6"):
                    cv2.putText(frame_copy, "EA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/ea.mp3").play(block=True)
                elif (x == "7"):
                    cv2.putText(frame_copy, "EAA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Ae.mp3").play(block=True)
                elif (x == "8"):
                    cv2.putText(frame_copy, "EE", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Aee.mp3").play(block=True)
                elif (x == "9"):
                    cv2.putText(frame_copy, "EIGHT", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/8.mp3").play(block=True)
                elif (x == "10"):
                    cv2.putText(frame_copy, "FIST", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Da.mp3").play(block=True)
                elif (x == "11"):
                    cv2.putText(frame_copy, "FIVE", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/5.mp3").play(block=True)
                elif (x == "12"):
                    cv2.putText(frame_copy, "FOUR", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/4.mp3").play(block=True)
                elif (x == "13"):
                    cv2.putText(frame_copy, "FRIEND", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Nanban.mp3").play(block=True)
                elif (x == "14"):
                    cv2.putText(frame_copy, "GA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Ga.mp3").play(block=True)
                elif (x == "15"):
                    cv2.putText(frame_copy, "GNA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Na.mp3").play(block=True)
                elif (x == "16"):
                    cv2.putText(frame_copy, "HELLO", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Hello.mp3").play(block=True)
                elif (x == "17"):
                    cv2.putText(frame_copy, "I", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Ai.mp3").play(block=True)
                elif (x == "18"):
                    cv2.putText(frame_copy, "KA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Ka.mp3").play(block=True)
                elif (x == "19"):
                    cv2.putText(frame_copy, "LA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Na.mp3").play(block=True)
                elif (x == "20"):
                    cv2.putText(frame_copy, "LLA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Cha.mp3").play(block=True)
                elif (x == "21"):
                    cv2.putText(frame_copy, "MA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Ma.mp3").play(block=True)
                elif (x == "22"):
                    cv2.putText(frame_copy, "NA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Na.mp3").play(block=True)
                elif (x == "23"):
                    cv2.putText(frame_copy, "NAA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Nna.mp3").play(block=True)
                elif (x == "24"):
                    cv2.putText(frame_copy, "NINE", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/9.mp3").play(block=True)
                elif (x == "25"):
                    cv2.putText(frame_copy, "NONE", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/0.mp3").play(block=True)
                elif (x == "26"):
                    cv2.putText(frame_copy, "O", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/o.mp3").play(block=True)
                elif (x == "27"):
                    cv2.putText(frame_copy, "OKAY", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/ok.mp3").play(block=True)
                elif (x == "28"):
                    cv2.putText(frame_copy, "ONE", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/1.mp3").play(block=True)
                elif (x == "29"):
                    cv2.putText(frame_copy, "OO", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/oo.mp3").play(block=True)
                elif (x == "30"):
                    cv2.putText(frame_copy, "OV", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/oo.mp3").play(block=True)
                elif (x == "31"):
                    cv2.putText(frame_copy, "PA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/1.mp3").play(block=True)
                elif (x == "32"):
                    cv2.putText(frame_copy, "PEACE", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/1.mp3").play(block=True)
                elif (x == "33"):
                    cv2.putText(frame_copy, "RA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/1.mp3").play(block=True)
                elif (x == "34"):
                    cv2.putText(frame_copy, "RAD", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/1.mp3").play(block=True)
                elif (x == "35"):
                    cv2.putText(frame_copy, "SANTHOSAM", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/1.mp3").play(block=True)
                elif (x == "36"):
                    cv2.putText(frame_copy, "SEVEN", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/7.mp3").play(block=True)
                elif (x == "37"):
                    cv2.putText(frame_copy, "SIX", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/6.mp3").play(block=True)
                elif (x == "38"):
                    cv2.putText(frame_copy, "STARAIGHT", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Ga.mp3").play(block=True)
                elif (x == "39"):
                    cv2.putText(frame_copy, "TA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Tha.mp3").play(block=True)
                elif (x == "40"):
                    cv2.putText(frame_copy, "TEN", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Pa.mp3").play(block=True)
                elif (x == "41"):
                    cv2.putText(frame_copy, "THA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Tha.mp3").play(block=True)
                elif (x == "42"):
                    cv2.putText(frame_copy, "THREE", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/3.mp3").play(block=True)
                elif (x == "43"):
                    cv2.putText(frame_copy, "THUMBS", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/ok.mp3").play(block=True)
                elif (x == "44"):
                    cv2.putText(frame_copy, "TWO", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/2.mp3").play(block=True)
                elif (x == "45"):
                    cv2.putText(frame_copy, "U", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/U.mp3").play(block=True)
                elif (x == "46"):
                    cv2.putText(frame_copy, "UU", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Uoo.mp3").play(block=True)
                elif (x == "47"):
                    cv2.putText(frame_copy, "VA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Ya.mp3").play(block=True)
                elif (x == "48"):
                    cv2.putText(frame_copy, "VANAKKAM", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Vanakkam.mp3").play(block=True)
                elif (x == "49"):
                    cv2.putText(frame_copy, "VEEDU", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Veedu.mp3").play(block=True)
                elif (x == "50"):
                    cv2.putText(frame_copy, "YA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Ya.mp3").play(block=True)
                elif (x == "51"):
                    cv2.putText(frame_copy, "ZHA", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # AudioPlayer("Audio_Files/" + Vtype + "/Cha.mp3").play(block=True)












    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 2)


    num_frames += 1


    cv2.imshow("Hand Gestures", frame_copy)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break


cam.release()
cv2.destroyAllWindows()
