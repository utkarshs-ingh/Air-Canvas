import cv2
import numpy as np
from sklearn.metrics import pairwise

background = None

accumulated_weight = 0.5
kernel = np.ones((5,5), np.uint8)

roi_top = 10
roi_bottom = 300
roi_right = 300
roi_left = 600

x1 = []
y1 = []
fgbg = cv2.createBackgroundSubtractorMOG2()

def calc_accum_avg(frame, accumulated_weight):
    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)
    
    # foreground extraction 
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    _, contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment)


def get_marker(thresholded, hand_segment):
    
    conv_hull = cv2.convexHull(hand_segment)
    
    top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

    
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2

    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
    max_distance = distance.max()
    
    # Create a circle with 90% radius of the max euclidean distance
    radius = int(0.8 * max_distance)
    
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    
    cv2.circle(circular_roi, (cX, cY), radius, (255, 0, 255), 10)
    
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    _, contours, _ = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    marker = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    ((x, y), r) = cv2.minEnclosingCircle(marker)
    
    return top, x, y, r



cam = cv2.VideoCapture(0)
num_frames = 0

while True:
    ret, frame = cam.read()

    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    roi = frame[roi_top:roi_bottom, roi_right:roi_left]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            cv2.imshow("Air-Canvas",frame_copy)
            
    else:
        hand = segment(gray)
        if hand is not None:
            thresholded, hand_segment = hand
            
            extTop, x, y, radius = get_marker(thresholded, hand_segment)
            
            x1.append(extTop[0])
            y1.append(extTop[1])

        key = cv2.waitKey(1)
        if key == ord('z'):
            del x1[-50::1]
            del y1[-50::1]


        for i in range(len(x1)):
            cv2.circle(frame_copy, (x1[i], y1[i]), 1, (255, 155, 100), 5)

    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)
    num_frames += 1

    cv2.imshow("Air-Canvas", frame_copy)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()