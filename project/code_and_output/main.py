
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob
import pandas as pd

files = glob.glob('output/*.png')
for f in files:
    os.remove(f)

from sort import *
cross_check = []
tracker = Sort()
memory = {}
time_test = {}
time_for_speed = []
df = pd.DataFrame(columns= ["TrackingID","FrameID","LaneID"])
df4 = pd.DataFrame(columns= ["TrackingID","Speed"])

dict_id_speed = {}
line1 = [(585,462), (212, 462)]
line_speed_end = [(585,462), (212, 462)]
line_speed_start = [(555,530), (100, 530)]

counter1 = 0


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False,
    help="path to input video", default="./input.mp4")
ap.add_argument("-o", "--output", required=False,
    help="path to output video", default="outp8ut.mp4")
ap.add_argument("-y", "--yolo", required=False,
    help="base path to YOLO directory", default="./yolo")
ap.add_argument("-c", "--confidence", type=float, default=0.40,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.40,
    help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


def adjust_gamma(image, gamma=1.0):

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 

    return cv2.LUT(image, table)


f1, g1 = 0, 720 
f2, g2 = 622, 720 
f3, g3 = 622, 330
f4, g4 = 0, 330

top_left_x = min([f1,f2,f3,f4])
print(top_left_x)
top_left_y = min([g1,g2,g3,g4])
print(top_left_y)
bot_right_x = max([f1,f2,f3,f4])
print(bot_right_x)
bot_right_y = max([g1,g2,g3,g4])
print(bot_right_y)

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


labelsPath = os.path.sep.join([args["yolo"] , "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
toDetect = ["car", "truck", "bus"]

weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])


print("[INFO] loading YOLO")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


vs = cv2.VideoCapture(args["input"])

writer = None
(W, H) = (None, None)

frameIndex = 0

try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))


except:
    print("[INFO] could not determine # of frames")
    print("[INFO] no approx. completion time can be provided")
    total = -1


if writer is None:
    
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(args["output"], fourcc, 30,
        (1080, 720), True)
while True:

    (grabbed, frame) = vs.read()
    if grabbed == False:
        break
    
    
    lane1 = np.array([[[0, 580], [375, 370], [450, 370], [40, 720]]], np.int32)
    lane2 = np.array([[[40, 720], [450, 370], [525, 370], [255, 720]]], np.int32)
    lane3 = np.array([[[255,720], [525,370], [605, 370], [500, 720]]], np.int32)
    cv2.polylines(frame, [lane1], True, (0,255,0), thickness=1)
    cv2.polylines(frame, [lane2], True, (0,0,255), thickness=1)
    cv2.polylines(frame, [lane3], True, (255,0,0), thickness=1)
    cv2.line(frame, line_speed_start[0], line_speed_start[1], (255, 255, 255), 1)
    cv2.line(frame, line_speed_end[0], line_speed_end[1], (255, 255, 255), 1)
    

    
    if W is None or H is None:
        (H, W) = frame.shape[:2]


    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (256, 256),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    center = []
    confidences = []
    classIDs = []

    
    for output in layerOutputs:
        
        for detection in output:
            
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            
            if confidence > args["confidence"] and LABELS[classID] in toDetect:
                
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                
                center.append(int(centerY))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                

    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    
    dets = []
    if len(idxs) > 0:
        
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x+w, y+h, confidences[i]])
            
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)
    
    boxes = []
    indexIDs = []
    c = []
    
    previous = memory.copy()
    
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            
            if (int(top_left_x) <= int(box[0]) <= int(bot_right_x)) :
                if int(box[3])-int(box[1]) < 120:

                    (x, y) = (int(box[0]), int(box[1]))
                    (w, h) = (int(box[2]), int(box[3]))
        
                    
                    p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                    ct1 = cv2.pointPolygonTest(lane1, p0, False)
                    ct2 = cv2.pointPolygonTest(lane2, p0, False)
                    ct3 = cv2.pointPolygonTest(lane3, p0, False)
                    
                        
                    color = (255,0,0) if ct1==1 else (0,255,0) if ct2==1 else (255,0,255) if ct3==1 else (0,0,255)
                    cv2.rectangle(frame, (x, y), (w, h), color, 4)
        
                    if indexIDs[i] in previous:
                        previous_box = previous[indexIDs[i]]
                        (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                        (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                        p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                        p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                        cv2.line(frame, p0, p1, color, 3)
                        id = indexIDs[i]    

                        if intersect(p0,p1,line_speed_start[0],line_speed_start[1]):
                            time_start = np.round(time.time(),3)
    
                            time_test.update({id:time_start})
                        elif intersect(p0,p1,line_speed_end[0],line_speed_end[1]):
                            if id in time_test:
                                time_taken = np.round(time.time(),3)-time_test.get(id)
                                time_for_speed.append(time_taken)
                                print(time_taken)
                                speed = 180/time_taken
                                print("SPEED", speed)
                                df3 = pd.DataFrame([[id,speed]], columns= ["TrackingID","Speed"])
    
                                df4=df4.append(df3,ignore_index=True)
    
                                del time_test[id]
                                
                        
                            
                            
                        
    
                        cv2.putText(frame, str(id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 5)
                        if intersect(p0, p1, line1[0], line1[1]) and indexIDs[i] not in cross_check:
                            counter1 += 1
                            cross_check.append(indexIDs[i])
                            lane = 1 if ct1==1 else 2 if ct2==1 else 3 if ct3==1 else "NA"
                            df2 = pd.DataFrame([[id,frameIndex,lane]], columns= ["TrackingID","FrameID","LaneID"])
                            df = df.append(df2,ignore_index=True)
            i += 1

    frameIndex +=  1

    
    cv2.line(frame, line1[0], line1[1], (0, 255, 255), 3)

    
    counter_text = "cars:{}".format(counter1)
    cv2.putText(frame, counter_text, (50,100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)
    

    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    new_dim = (1080,720)
    writer.write(cv2.resize(frame,new_dim, interpolation = cv2.INTER_AREA))

    

final_df1 = pd.merge_ordered(df, df4, how='outer', on='TrackingID')
final_df1 = final_df1[final_df1.FrameID.notnull()]
print("[INFO] Done")
writer.release()
vs.release()
final_df1.to_csv("./report.csv",index=False)