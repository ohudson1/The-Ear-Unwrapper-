import cv2
import numpy as np
import time
import usb.core
import usb.util
import sys
from PIL import Image
import glob
from datetime import datetime
import argparse

#user sets initial corn sample ID number on startup with command-line argument
parser = argparse.ArgumentParser()
parser.add_argument('--idnum', type=int)
args = parser.parse_args()
corn_id_num = args.idnum

#initialization code to interface with USB device
# we must put the camera in external trigger mode 
# to allow shutter control by the microcontroller,
# the camera manufacturer provided no API, so we have to
# assemble the packet manually into the msg[] array,
# and write it to the device 

#search available USB devices for the correct device ID
dev=usb.core.find(idVendor=0x2560, idProduct=0xc128)
dev_in=dev[0][(2,0)][0] #interrupt in
dev_out=dev[0][(2,0)][1] #interrupt out
if dev is None:
    raise ValueError('Device not found')
cfg=-1
i = dev[0].interfaces()[2].bInterfaceNumber
cfg = dev.get_active_configuration()
intf = cfg[(2,0)]
if dev.is_kernel_driver_active(i):
    try:
        reattach = True
        dev.detach_kernel_driver(i)
    except usb.core.USBError as e:
        sys.exit("Could not detatch kernel driver from interface({0}): {1}".format(i, str(e)))

#print(dev)

#assemble config command and write to device USB buffer
msg = [0] * 64
#msg = [0xA0, 0xc1, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]
msg[0] = 0xA8
msg[1] = 0x1c
msg[2] = 0x01
msg[3] = 0x00
msg[6] = 0x00
dev.write(0x6,msg,1000)

#initialize and begin image capture
mtx= np.array([[686.902, 0, 652.704],
                              [0, 686.492, 354.634],
                              [0, 0, 1]])
dist=np.array([[-0.33881202,  0.13453343, -0.00057266, -0.00039356, -0.0267943 ]])
cap = cv2.VideoCapture(0)
img_counter=0

# confirm the webcam is opened correctly for capture
if not cap.isOpened():
    raise IOError("Cannot open webcam")
cap.set(3,1280)
cap.set(4,720)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)    #set at 1 default. 0 to turn off autoexposure.  
cap.set(cv2.CAP_PROP_EXPOSURE, 20)       #Exposure time
index=0
slit=np.empty([800,1280,3])

#allows the user to provide as estimated diameter for later scaling reference
diameter = input("Session Diameter: ")

#main application loop
while True:
    ret, frame = cap.read()
    if not ret: break
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (1280,720), cv2.CV_16SC2)
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    #display live feed
    cv2.imshow('Input', frame)
    keypress = cv2.waitKey(1)
    #press ESC to exit
    if keypress == 27:
        break
    #press 's' to skip an ID number / sample    
    elif keypress == 115:
        corn_id_num += 1
        continue
    #press spacebar to initiate acquisition
    elif keypress == 32:
        timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        for index in range(800):
            keypress = cv2.waitKey(1)
            #press ESC to exit
            if keypress == 27:
                break
            ret, frame = cap.read()
            if not ret: break
            #correct for 'fisheye' lens effect 
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (1280,720), cv2.CV_16SC2)
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            cv2.imshow('Input', frame)
            slice=frame[360,0:1279]

            #capture the slitwise image
            for x in range(1279):
                temp = []
                for i in range(3):
                    temp.append(((slice[x][i]))/1)
                slit[index][x]=temp
            cv2.imshow('slit', slit/255)
            
            #write out full images at each 90 degree interval for reference / QA
            if index == 0 or index == 200 or index == 400 or index == 600:
                corn_profile_name = timestamp + "_" + str(corn_id_num) + "_" + str(diameter) + "_" + str(index) + ".png"
                cv2.imwrite(corn_profile_name, frame)
            index += 1
        outname = timestamp + "_" + str(corn_id_num) + "_" + str(diameter) + ".png"
        corn_id_num += 1
        #write out unwrapped image
        cv2.imwrite(outname, slit)

#release device and close windows on termination         
cap.release()
cv2.destroyAllWindows()
