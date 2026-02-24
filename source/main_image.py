#Import relevant modules
import os
import cv2

#Read in the image file
path = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        'config_files',
        'bulky2.jpg'
    )
)

weightsPath = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        'config_files',
        'frozen_inference_graph.pb'
    )
)

image = cv2.imread(path) #Read in the image file


#Import the class names
classNames = []
classFile = path

#### Extracts the classes into a list
with open(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'coco.names'), "r", encoding="utf-8") as f:
    classNames = f.read().rstrip('\n').split('\n')

#Import the config and weights file
configPath = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        'config_files',
        'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    )
)

weightsPath = os.path.realpath(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        'config_files',
        'frozen_inference_graph.pb'
    )
)
#Set relevant parameters and initiate model
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

#Extract, ClassIds, confidence and bounding box info
classIds, confs, bbox = net.detect(image, confThreshold=0.5) #if 50% confident, predict
print(classIds, confs,bbox)

#For loop for three different list led to the use of list
#The output from clasids.flatten, confs.flatten and bbox is [1] [0.6724][[60 40 373 461]]
#If you had multiple objects, the first object and list of box coordinates are all in line.
#The object, confidence and co-ordinates can know all be used in the opencv packages

for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
    cv2.rectangle(image,box,color=(0,255,0), thickness=2)
    cv2.putText(image,classNames[classId-1],(box[0]+10,box[1]+30), 
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

#Open
cv2.imshow("Output", image)
cv2.waitKey(0)