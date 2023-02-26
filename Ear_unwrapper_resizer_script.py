##corn diameter estimation script. Written and executed from Pycharm 2022.2.1
##Tested/runs in Python 3.1 installed via homebrew on macOS. If any errors, ensure running in the correct environment.
##Script expects /inputs and /output directory in parent folder.
​
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
import threading
from datetime import datetime
​
os.chdir('inputs')
images = os.listdir(os.getcwd())
bounding_box=(278, 152, 828, 400)
x_pt, y_pt = -1, -1
active_drawing = False
drawing = False
enable_threading = False
manual_selection = False
​
​
def draw_bounding_box(click, x, y, flag_param, parameters):
    global x_pt, y_pt, drawing, top_left_point, bottom_right_point, original_image_200_temp1, original_image_200_temp2, bounding_box
    if click == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_pt, y_pt = x, y
​
    elif click == cv2.EVENT_MOUSEMOVE:
        if drawing:
            original_image_200_temp1 = original_image_200_temp2.copy()
            top_left_point, bottom_right_point = (x_pt,y_pt), (x,y)
            image[y_pt:y, x_pt:x] = 255 - original_image_200_temp1[y_pt:y, x_pt:x]
            cv2.rectangle(original_image_200_temp1, top_left_point, bottom_right_point, (0,255,0), 2)
    elif click == cv2.EVENT_LBUTTONUP:
        drawing = False
        top_left_point, bottom_right_point = (x_pt,y_pt), (x,y)
        image[y_pt:y, x_pt:x] = 255 - image[y_pt:y, x_pt:x]
        cv2.rectangle(image, top_left_point, bottom_right_point, (0,255,0), 2)
        bounding_box = (x_pt, y_pt, x-x_pt, y-y_pt)
        print(bounding_box)
        #active_drawing = False
        #grabcut_algorithm(original_image, bounding_box)
count = 0
def grabcut_algorithm(original_image, bounding_box, cell, show_images):
    original_image=hsv_subtract(original_image)
    greyImg = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    ret, greyImg = cv2.threshold(greyImg, 30, 255, cv2.THRESH_BINARY)
    segment = np.zeros(greyImg.shape[:2], np.uint8)
    #segment = greyImg.copy()
    x,y,width,height = bounding_box
    segment[y:y+height, x:x+width] = 1
    img_mult = greyImg*segment
    background_mdl = np.zeros((1,65), np.float64)
    foreground_mdl = np.zeros((1,65), np.float64)
​
    new_mask = np.where((img_mult==2)|(img_mult==0),0,255).astype('uint8')
    new_mask2 = np.where((img_mult == 2) | (img_mult == 0), 0, 1).astype('uint8')
    masked_original_image = original_image*new_mask[:,:,np.newaxis]
    segment = np.zeros(greyImg.shape[:2], np.uint8)
    x, y, width, height, mask = centroid(new_mask, original_image, cell)
    segment[y:y + height, x:x + width] = 1
    img_mult = mask * segment
    new_mask = np.where((img_mult==2)|(img_mult==0),0,255).astype('uint8')
    new_mask2 = np.where((img_mult == 2) | (img_mult == 0), 0, 1).astype('uint8')
    corn_sum_of_columns = np.sum(new_mask2, 0)
    corn_sum_of_columns_temp = corn_sum_of_columns.copy()
    corn_sum_of_rows = np.sum(new_mask2, 0)
    corn_sum_of_columns_temp[corn_sum_of_columns_temp<20] = 1
    corn_sum_of_columns[corn_sum_of_columns < 20] = 0
    corn_num_of_valid_columns = np.divide(corn_sum_of_columns, corn_sum_of_columns_temp)
    corn_num_of_valid_columns = np.sum(corn_num_of_valid_columns)
    corn_avg_diameter_px = np.sum(corn_sum_of_columns)/corn_num_of_valid_columns
​
    #corn_avg_diameter_px = np.average(corn_sum_of_columns)
    est_dia_array[cell] = round(corn_avg_diameter_px/44.3, 2)
    if(show_images):
        img = cv2.rectangle(original_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        while(True):
            cv2.imshow("Frame", img)
            cv2.imshow("newmask", new_mask)
            cv2.imshow("mask", mask)
            if cv2.waitKey(10) == 13:
                break
    #cv2.imwrite('../outputs/' + "est_dia"+str(est_dia_array[cell])+ "_" + corn_id + "_" + str(cell) + "_" 'mask.png', new_mask)
    #print(est_dia_array[cell]) #for manual review of individual masked images
​
def centroid(new_mask, OG_image, cell):
    thresh2 = new_mask
    #tmpArea = np.zeros(new_mask.shape)
    #tmpArea = OG_image.copy()
    tmpArea = cv2.cvtColor(OG_image, cv2.COLOR_BGR2GRAY)
    ret, tmpArea = cv2.threshold(tmpArea, 127, 255, cv2.THRESH_BINARY)
    # Contours
    contours = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    c = max(contours, key=cv2.contourArea)
    cv2.drawContours(thresh2,[c],-1,(255),-1)
    #cv2.drawContours(OG_image,c,-1,(255, 255, 0),2)
    x, y, w, h = cv2.boundingRect(c)
    img = cv2.rectangle(OG_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #rect = cv2.minAreaRect(c)
    #box = cv2.boxPoints(rect)
    #box = np.int0(box)
    #img = cv2.drawContours(OG_image, [box], 0, (0, 255, 255), 2)
    #print(box)
    print("x= " + str(x) + " y= " + str(y) + " w= " + str(w) + " h= " + str(h))
    return x, y, w, h, thresh2
    # Centroid
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    #cv2.circle(tmpArea, (cx, cy), 5, (0, 0, 255), -1)
    #cv2.imwrite('../outputs/' + corn_id + "_" + str(cell) + "_" 'tmpArea.png', tmpArea)
    # Ellipse
    e = cv2.fitEllipse(c)
    cv2.ellipse(tmpArea, e, (0, 255, 0), 2)
​
    # Principal axis
    x1 = int(np.round(cx + e[1][1] / 2 * np.cos((e[2] + 90) * np.pi / 180.0)))
    y1 = int(np.round(cy + e[1][1] / 2 * np.sin((e[2] + 90) * np.pi / 180.0)))
    x2 = int(np.round(cx + e[1][1] / 2 * np.cos((e[2] - 90) * np.pi / 180.0)))
    y2 = int(np.round(cy + e[1][1] / 2 * np.sin((e[2] - 90) * np.pi / 180.0)))
    if (x1>x2):
        xtemp = x1
        x1=x2
        x2=xtemp
    if (y1>y2):
        ytemp = y1
        y1=y2
        y2=ytemp
    corn_length=np.sum(new_mask[y1:y2, x1:x2],)/255
    print("corn length= " + str(corn_length))
    cv2.line(new_mask, (x1, y1), (x2, y2), (255, 255, 0), 2)
    while True:
        cv2.imshow('mask', OG_image)
        cv2.imshow('contour', OG_image)
        c = cv2.waitKey(1)
        if c == 13:
            break
    return corn_length
def hsv_subtract(image):
    result = image.copy()
​
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
​
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([10, 00, 0])
    upper1 = np.array([100, 255, 255])
​
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160, 100, 20])
    upper2 = np.array([179, 255, 255])
​
    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)
​
    full_mask = lower_mask + upper_mask
​
    result = cv2.bitwise_and(result, result, mask=full_mask)
​
    return result
​
​
if __name__=='__main__':
    active_drawing = True
    top_left_point, bottom_right_point = (-1,-1), (-1,-1)
    path=Path(os.getcwd())
    est_dia_array = [0.0, 0.0, 0.0, 0.0]
​
    for image in images:
​
        if image.endswith("200.png"): #all the images end in *0.png, this gets is a unique identifier for the group.
        #if image.endswith("534_3.1_200.png"):  # optional line/template if you just want to pull in a single corn ID.
            time_stamp = image.split("_")[0].strip(".png")
            corn_id = image.split("_")[1].strip(".png")
            diameter = image.split("_")[2].strip(".png")
            for file_path_200 in glob.glob("*"+"_"+corn_id+"_"+diameter+"_200.png"):
                print("running " + file_path_200)
            #slit = cv2.imread(file_path)
            for file_path_400 in glob.glob("*"+"_"+corn_id+"_"+diameter+"_400.png"):
                print("found " + file_path_400)
            for file_path_600 in glob.glob("*"+"_"+corn_id+"_"+diameter+"_600.png"):
                print("found " + file_path_600)
            for file_path_0 in glob.glob("*"+"_"+corn_id+"_"+diameter+"_0.png"):
                print("found " + file_path_0)
            for file_path_slit in glob.glob("*"+"_"+corn_id+"_"+diameter+".png"):
                print("found " + file_path_slit)
            original_image_200 = cv2.imread(file_path_200)
            original_image_200_temp1 = hsv_subtract(original_image_200)
            original_image_200_temp2 = original_image_200_temp1.copy()
            original_image_400 = cv2.imread(file_path_400)
            original_image_600 = cv2.imread(file_path_600)
            original_image_000 = cv2.imread(file_path_0)
            slit_scan = cv2.imread(file_path_slit)
            image = slit_scan.copy()
​
            if(manual_selection):
                cv2.namedWindow('Frame')   #Set of functions to enable manual corn crop/selection for corn diameter.
                cv2.setMouseCallback('Frame', draw_bounding_box)
                while(True):
                    cv2.imshow("Frame", original_image_200_temp1)
                    if cv2.waitKey(10) == 13:
                        break
                cv2.waitKey(1)
​
​
            # multithreaded option
            if(enable_threading==True):
                print("starting multi threading analysis")
                t1 = threading.Thread(target=grabcut_algorithm, args=(original_image_200, bounding_box, 1, False))
                t2 = threading.Thread(target=grabcut_algorithm, args=(original_image_400, bounding_box, 2, False))
                t3 = threading.Thread(target=grabcut_algorithm, args=(original_image_600, bounding_box, 3, False))
                t4 = threading.Thread(target=grabcut_algorithm, args=(original_image_000, bounding_box, 0, False))
​
                t1.start()
                t2.start()
                t3.start()
                t4.start()
​
                t1.join()
                t2.join()
                t3.join()
                t4.join()
​
            #single threading option
            if(enable_threading==False):
                print("starting single threaded analysis")
                grabcut_algorithm(original_image_200, bounding_box, 1, True) #last parameter is whether or not to manually review masks prior to pixel summation.
                grabcut_algorithm(original_image_400, bounding_box, 2, True) #if enabled, after picture shows, press ENTER to proceed. There will be four pictures
                grabcut_algorithm(original_image_600, bounding_box, 3, True) #to review for each image set.
                grabcut_algorithm(original_image_000, bounding_box, 0, True)
​
            est_diameter = sum(est_dia_array)/len(est_dia_array)
            est_diameter = round(est_diameter, 2)
            est_diameter_str = str(est_diameter)
            print("est_diameter=" + est_diameter_str)
​
            f_diameter = float(diameter)
            if (abs(est_diameter - f_diameter) > 0.5):
                print("WARNING: LARGE SIZE DEVIATION. Owen diameter: " + diameter + " Est diameter= " + est_diameter_str)
​
            cv2.imwrite('../outputs/' + "estDia_" + str(est_diameter)+"_" + file_path_slit, slit_scan)
print("done")
cv2.destroyAllWindows()