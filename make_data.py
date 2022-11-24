# make data for training model
# python make_data.py --data

import argparse
import cv2
import numpy as np
import os
import time
import pandas as pd
import mediapipe as mp
import threading
import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *



# point_landmark_toget = [0, 11 ,12 ,13, 14,15 ,16, 19, 20, 23, 24]

def get_frame_landmarks(results):
    size_landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark[:23]]).flatten() if results.pose_landmarks else np.zeros(4*23)
    return size_landmarks

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def get_scaled_landmarks(landmarks, dimenson):
    landmarks_2d = []
    if dimenson == '2d':
        for landmark in landmarks:
            x, y = int(landmark.x*1280), int(landmark.y*720)
            landmarks_2d.append([x, y])
        return landmarks_2d



def show_fps(image, prev_frame_time):
    new_frame_time = time.time()
    fps = int(1/(new_frame_time-prev_frame_time))
    cv2.putText(image, f"fps: {fps}", (1000, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 2, cv2.LINE_AA)
    return new_frame_time
        
def draw_landmark_on_image( results, image):
    lmks = results.pose_landmarks.landmark
    pose_landmarks = [lmks[0], lmks[11], lmks[12], lmks[13], lmks[14], lmks[15], lmks[16], lmks[23], lmks[24], lmks[19], lmks[20]] 
    pose_landmarks = get_scaled_landmarks(pose_landmarks, '2d')
    
    cv2.line(image, tuple(pose_landmarks[0]), tuple(pose_landmarks[4]), (0, 144, 255), 2)
    cv2.line(image, tuple(pose_landmarks[0]), tuple(pose_landmarks[1]), (255, 205, 0), 2)
    cv2.line(image, tuple(pose_landmarks[4]), tuple(pose_landmarks[6]), (0, 144, 255), 2)
    cv2.line(image, tuple(pose_landmarks[6]), tuple(pose_landmarks[8]), (0, 144, 255), 2)
    cv2.line(image, tuple(pose_landmarks[1]), tuple(pose_landmarks[3]), (255, 205, 0), 2)
    cv2.line(image, tuple(pose_landmarks[3]), tuple(pose_landmarks[7]), (255, 205, 0), 2)
    cv2.line(image, tuple(pose_landmarks[9]), tuple(pose_landmarks[10]), (255, 255, 255), 2)

    cv2.line(image, tuple(pose_landmarks[11]), tuple(pose_landmarks[12]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[11]), tuple(pose_landmarks[13]), (255, 205, 0), 2)
    cv2.line(image, tuple(pose_landmarks[11]), tuple(pose_landmarks[23]), (255, 205, 0), 2)

    cv2.line(image, tuple(pose_landmarks[24]), tuple(pose_landmarks[23]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[24]), tuple(pose_landmarks[12]), (0, 144, 255), 2)

    cv2.line(image, tuple(pose_landmarks[14]), tuple(pose_landmarks[16]), (0, 144, 255), 2)
    cv2.line(image, tuple(pose_landmarks[14]), tuple(pose_landmarks[12]), (0, 144, 255), 2)

    cv2.line(image, tuple(pose_landmarks[16]), tuple(pose_landmarks[22]), (0, 144, 255), 2)
    cv2.line(image, tuple(pose_landmarks[16]), tuple(pose_landmarks[20]), (0, 144, 255), 2)
    cv2.line(image, tuple(pose_landmarks[16]), tuple(pose_landmarks[18]), (0, 144, 255), 2)
    cv2.line(image, tuple(pose_landmarks[18]), tuple(pose_landmarks[20]), (0, 144, 255), 2)

    cv2.line(image, tuple(pose_landmarks[15]), tuple(pose_landmarks[13]), (255, 205, 0), 2)
    cv2.line(image, tuple(pose_landmarks[15]), tuple(pose_landmarks[21]), (255, 205, 0), 2)
    cv2.line(image, tuple(pose_landmarks[15]), tuple(pose_landmarks[19]), (255, 205, 0), 2)
    cv2.line(image, tuple(pose_landmarks[15]), tuple(pose_landmarks[17]), (255, 205, 0), 2)
    cv2.line(image, tuple(pose_landmarks[19]), tuple(pose_landmarks[17]), (255, 205, 0), 2)


    for lm in pose_landmarks:

        cv2.circle(image, (int(lm[0]), int(lm[1])), 5, (255,255,255), 4)
        cv2.circle(image, (int(lm[0]), int(lm[1])), 4, (255,255,141), -1)
        
        
        

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#     parser.add_argument('', type=str, default='yolo7.pt', help='initial weights path')
#     parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data','-d', type=str, default='./data', help='path')
    parser.add_argument('--type','-t', type=str, default='train', help='training or testing data')
    opt = parser.parse_args()
    
    
    path_data = opt.data
    path_video = os.path.join(path_data)
    path_csv = os.path.join(path_data,'landmarks',opt.type)
    name_actions = ['cheat','non_cheat']

    # Khởi tạo thư viện mediapipe
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    mp_drawing = mp.solutions.drawing_utils

    prev_frame_time = 0
    
    for name_action in name_actions:
        direction_path = os.path.join(path_video,name_action)
        lm_list = []
        frame_num = 0

        for id_sequence, video_name in enumerate(os.listdir(direction_path)):
            video_path = os.path.join(direction_path, video_name)
            cap = cv2.VideoCapture(video_path)


            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Video end")
                    break

                frame_num += 1
                print(frame_num)

                frame = cv2.resize(frame,(1300,750)) 
                frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = pose.process(frameRGB)

                if results.pose_landmarks:
                    # Ghi nhận thông số khung xương
                    # Vẽ khung xương lên ảnh
                    draw_landmark_on_image( results, frame)

                    lm = get_frame_landmarks(results)

                    lm_list.append(lm)
                    print(lm)
                prev_frame_time = show_fps(frame, prev_frame_time)
                cv2.imshow("image", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            cap.release()

        df  = pd.DataFrame(lm_list)
       
        csv_per_action = os.path.join(path_csv, name_action +'.csv')
        with open(csv_per_action, mode='w') as f:
            df.to_csv(f)
        
    