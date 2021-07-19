import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from sklearn.externals._pilutil import imresize
from tensorflow.keras.models import load_model

# Load Keras model
model = load_model(r'E:\Administrator\Desktop\Lane_line_detection\model\model.h5')


def road_lines(image):
    """
    Takes in a road image, re-sizes for the model,predicts the lane to be drawn from the model in G color,recreates an
    RGB image of a lane and merges with the original road image.
    """
    recent_fit = []

    # 准备一张图片并输入到模型中
    small_img = np.array(imresize(image, (80, 160, 3)))
    small_img = small_img[None, :, :, :]

    prediction = model.predict(small_img)[0] * 255
    recent_fit.append(prediction)

    # 只用最后五个求平均
    if len(recent_fit) > 5:
        recent_fit = recent_fit[1:]
    avg_fit = np.mean(np.array([i for i in recent_fit]), axis=0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, avg_fit, blanks))

    # 调整大小以匹配原始图像
    lane_image = imresize(lane_drawn, (720, 1280, 3))
    # 合并车道绘图到原始图像
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)
    return result


if __name__ == "__main__":
    video = VideoFileClip(r"E:\BaiduNetdiskDownload\road_line.mp4")
    vid_clip = video.fl_image(road_lines)
    vid_clip.write_videofile(r'E:\Administrator\Desktop\Lane_line_detection\output\test_video.mp4', audio=False)
