import cv2
import numpy as np

def extract_background(video_path):
    # 打开视频
    cap = cv2.VideoCapture(video_path)

    # 读取第一帧图像
    ret, frame = cap.read()

    # 将第一帧图像转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用中值滤波去除噪声
    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.medianBlur(gray, 5)

    # 保存背景
    background = gray

    # 关闭视频
    cap.release()

    return background

def detect_yellow_line_movement(video_path):
    # 打开视频
    cap = cv2.VideoCapture(video_path)

    # 读取第一帧图像并提取背景
    background = extract_background(video_path)

    # 初始化黄色区域的边界
    border = np.zeros_like(background)

    # 循环处理视频中的每帧图像
    while True:
        # 读取帧图像
        ret, frame = cap.read()

        # 将帧图像转换为 HSV 颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 使用 Otsu 阈值算法自动设定阈值
        threshold, mask = cv2.threshold(hsv[:, :, 2], 150, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

        # 使用中值滤波去除噪声
        # kernel = np.ones((5, 5), np.uint8)
        # mask = cv2.medianBlur(mask, 5)

        # 使用背景减除方法减去背景噪声
        # mask = cv2.absdiff(mask, background)

        # 使用形态学变换来去除噪声
        # kernel = np.ones((5, 5), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 计算黄色区域的边界
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            border[y:y + h, x:x + w] = 255

        # 显示黄色区域的边界
        cv2.imshow("Border", border)

        # 按键退出
        if cv2.waitKey(1) == 27:
            break

    # 关闭视频
    cap.release()

# 测试代码
video_path = r'230913100646_2.5V.mp4'
detect_yellow_line_movement(video_path)