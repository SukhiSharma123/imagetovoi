from imageai.Detection import ObjectDetection
from gtts import gTTS
import os
import cv2
from playsound import playsound
from time import sleep
key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
sleep(2)
while True:

    try:
        check, frame = webcam.read()
        print(check) #prints true as long as the webcam is running
        print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            cv2.imwrite(filename='saved_img.jpg', img=frame)
            webcam.release()
            print("Image saved!")
            
            break
        
        elif key == ord('q'):
            webcam.release()
            cv2.destroyAllWindows()
            break
    
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "saved_img.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

import csv
os.remove('myfile.txt')
for eachObject in detections:
    #print(eachObject["name"] , " : " , eachObject["percentage_probability"] )   #it works....
    hi = eachObject["name"]
    file1 = open("myfile.txt","a")#append mode
    file1.write(hi) 
    file1.write("\n") 
    file1.close()
    print(hi)

##############################


#######################################
file = open("myfile.txt", "r").read().replace("\n", " ")
language='en'
speech = gTTS(text = str(file), lang = language, slow = True)
speech.save("voice.mp3")
#os.system("start voice.mp3") 
playsound("voice.mp3")