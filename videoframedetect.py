
import cv2
import numpy as np
import time
def yello_extract(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([90,0,0])
    high_hsv = np.array([255,255,255])
    mask = cv2.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)
    # cv2.imshow("find_yellow",mask)
    return mask
    
# 分解视频图片
cap = cv2.VideoCapture(r'video/230913100646_2.5V.mp4')
is_opened = cap.isOpened()
print(is_opened)
# 获取视频属性
fps = cap.get(cv2.CAP_PROP_FPS)   #帧数
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(fps, width, height,fc)

i = 0
(flag, frame_init) = cap.read()
# flag = True

while is_opened and flag:
    if i == int(4000):    # 获取2s帧图片
        break
    else:
        i += 1
    (flag, frame) = cap.read()
    frame = yello_extract(frame) # 得到阈值划分后的二值图像


    contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    cv2.imshow("frame",frame)
    waitKey = cv2.waitKey(1) & 0xFF
    if waitKey == ord(
        "q"
    ):  # if Q pressed you could do something else with other keypress
        print("closing video and exiting")
        cv2.destroyWindow("123")
        # video.release()
        break

cv2.waitKey(0) 
cv2.destroyAllWindows()


print('end')
cap.release()

