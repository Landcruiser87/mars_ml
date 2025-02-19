#Import libraries
import numpy as np
import cv2
import os
import logging
import time
import requests
import json
import datetime
from pathlib import Path, PurePath
from PIL import Image
from support import logger


#FUNCTION horizon test
def horizon_test(img:np.array) -> bool:
    ######## Horizon test ############
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Edge Detection (Canny or Sobel)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # Adjust thresholds as needed
        # OR, if Canny is too sensitive:
        # sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # Adjust ksize
        # sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        # edges = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)

        # 2. Hough Line Transform (Probabilistic)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=200, maxLineGap=50) # Tweak parameters
        # Horizon Confirmation (Check for near-horizontal lines)
        if lines is not None:
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) < 15:  # Adjust angle tolerance (15 degrees)
                    horizontal_lines.append(line[0])
            # Check if we found any near-horizontal lines
            if horizontal_lines:  
                return False # Found a horizon (not invalid)
            else:
                return True # No near-horizontal lines, is invalid

        else:
            return True  # No lines detected

    except Exception as e: #Invalid due to errors
        logger.debug(f"Error processing image: {e}")
        return True

def horizon_dos(img, display_res:bool=False):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #Blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Adjust kernel size as needed

        #Calculate vertical gradients (Sobel)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5) # vertical gradient
        abs_sobel_y = np.absolute(sobel_y)
        scaled_sobel_y = np.uint8(abs_sobel_y)

        #Threshold the gradient image (find strong horizontal edges)
        _, thresh = cv2.threshold(scaled_sobel_y, 50, 255, cv2.THRESH_BINARY)  # Adjust threshold

        #Find the "most prominent" horizontal line (horizon candidate)
        histogram = np.sum(thresh, axis=1)  # Sum across rows (horizontal direction)
        max_intensity_row = np.argmax(histogram) # Row with the most white pixels (strongest edge)

        #Confidence measure (how distinct is the horizon?)
        confidence = histogram[max_intensity_row] / np.sum(histogram) if np.sum(histogram) > 0 else 0

        if confidence < 0.1:  # Adjust this confidence threshold
            logger.warning(f"{confidence} Low confidence horizon detection.")
            return None

        if display_res:
            # Draw the detected horizon line
            cv2.line(img, (0, max_intensity_row), (img.shape[1], max_intensity_row), (0, 0, 255), 2)
            cv2.namedWindow("Mars Horizon")
            cv2.setWindowProperty("Mars Horizon", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Mars Horizon', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return (max_intensity_row, confidence)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def horizon_tres(image, plot_res:bool=False):
    def generate_plot(pl_name:str, plot_data:np.array):
        cv2.namedWindow(pl_name, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(pl_name, plot_data)
        cv2.resizeWindow(pl_name, 1200, 800)
        cv2.waitKey(0)
        cv2.destroyWindow(pl_name)
    
    #Pulled from here
    #https://github.com/DevashishPrasad/CascadeTabNet/blob/master/Table%20Structure%20Recognition/Functions/line_detection.py

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 1)
    bw = cv2.bitwise_not(bw)
    ## To visualize image after thresholding ##
    # if plot_res:
    #     cv2.imshow("bw",bw)
    #     cv2.waitKey(0)
    ###########################################
    horizontal = bw.copy()
    vertical = bw.copy()
    img = image.copy()
    # [horizontal lines]
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    horizontal = cv2.dilate(horizontal, (1,1), iterations=5)
    horizontal = cv2.erode(horizontal, (1,1), iterations=5)

    ## Uncomment to visualize highlighted Horizontal lines
    if plot_res:
        generate_plot("horizontal", horizontal)
        
    # HoughlinesP function to detect horizontal lines
    hor_lines = cv2.HoughLinesP(horizontal,rho=1,theta=np.pi/180,threshold=100,minLineLength=30,maxLineGap=3)
    if hor_lines is None:
        return None,None
    temp_line = []
    for line in hor_lines:
        for x1,y1,x2,y2 in line:
            temp_line.append([x1,y1-5,x2,y2-5])

    # Sorting the list of detected lines by Y1
    hor_lines = sorted(temp_line,key=lambda x: x[1])

    ## Uncomment this part to visualize the lines detected on the image ##
    logger.info(f"number of lines {(len(hor_lines))}")
    for x1, y1, x2, y2 in hor_lines:
        cv2.line(image, (x1,y1), (x2,y2), (0, 255, 0), 1)
    
    logger.info(f"Shape of image {image.shape}")
    if plot_res:
        generate_plot("image", image)
 
    # ####################################################################

    ## Selection of best lines from all the horizontal lines detected ##
    lasty1 = -111111
    lines_x1 = []
    lines_x2 = []
    hor = []
    i=0
    for x1,y1,x2,y2 in hor_lines:
        if y1 >= lasty1 and y1 <= lasty1 + 10:
            lines_x1.append(x1)
            lines_x2.append(x2)
        else:
            if (i != 0) and (len(lines_x1) != 0):
                hor.append([min(lines_x1),lasty1,max(lines_x2),lasty1])
            lasty1 = y1
            lines_x1 = []
            lines_x2 = []
            lines_x1.append(x1)
            lines_x2.append(x2)
            i+=1
    hor.append([min(lines_x1),lasty1,max(lines_x2),lasty1])
    #####################################################################

    # [vertical lines]
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    vertical = cv2.dilate(vertical, (1,1), iterations=8)
    vertical = cv2.erode(vertical, (1,1), iterations=7)

    ######## Preprocessing Vertical Lines ###############
    if plot_res:
        generate_plot("vertical", vertical)
    #####################################################

    # HoughlinesP function to detect vertical lines
    # ver_lines = cv2.HoughLinesP(vertical,rho=1,theta=np.pi/180,threshold=20,minLineLength=20,maxLineGap=2)
    ver_lines = cv2.HoughLinesP(vertical, 1, np.pi/180, 20, np.array([]), 20, 2)
    if ver_lines is None:
        return None,None
    temp_line = []
    for line in ver_lines:
        for x1,y1,x2,y2 in line:
            temp_line.append([x1,y1,x2,y2])

    # Sorting the list of detected lines by X1
    ver_lines = sorted(temp_line,key=lambda x: x[0])

    ## Uncomment this part to visualize the lines detected on the image ##
    logger.info(len(ver_lines))
    for x1, y1, x2, y2 in ver_lines:
        cv2.line(image, (x1,y1-5), (x2,y2-5), (0, 255, 0), 1)

    # print(image.shape)
    if plot_res:
        generate_plot("image", image)
    ####################################################################

    ## Selection of best lines from all the vertical lines detected ##
    lastx1 = -111111
    lines_y1 = []
    lines_y2 = []
    ver = []
    count = 0
    lasty1 = -11111
    lasty2 = -11111
    for x1,y1,x2,y2 in ver_lines:
        if x1 >= lastx1 and x1 <= lastx1 + 15 and not (((min(y1,y2)<min(lasty1,lasty2)-20 or min(y1,y2)<min(lasty1,lasty2)+20)) and ((max(y1,y2)<max(lasty1,lasty2)-20 or max(y1,y2)<max(lasty1,lasty2)+20))):
            lines_y1.append(y1)
            lines_y2.append(y2)
            # lasty1 = y1
            # lasty2 = y2
        else:
            if (count != 0) and (len(lines_y1) != 0):
                ver.append([lastx1,min(lines_y2)-5,lastx1,max(lines_y1)-5])
            lastx1 = x1
            lines_y1 = []
            lines_y2 = []
            lines_y1.append(y1)
            lines_y2.append(y2)
            count += 1
            lasty1 = -11111
            lasty2 = -11111
    ver.append([lastx1,min(lines_y2)-5,lastx1,max(lines_y1)-5])
    #################################################################


    ############ Visualization of Lines After Post Processing ############
    if plot_res:
        for x1, y1, x2, y2 in ver:
            cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 1)

        for x1, y1, x2, y2 in hor:
            cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 1)
        generate_plot("image", img)
        cv2.destroyAllWindows()
    return hor,ver

def classify_res(hor:np.array, ver:np.array):
    
    pass

def training_images() -> list:
    valid = [
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217291_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217146_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217180_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217204_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217228_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217249_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217270_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217291_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217312_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217333_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217353_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217373_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217393_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217414_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217462_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217486_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217510_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217530_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217550_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217574_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217600_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217624_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217648_000IOF_N0010052AUT_04096_034085A03.png",
        "",
        "",
        "",
        "",

    ]
    invalid = [
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667216999_445IOF_N0010052ZCAM00013_0630LUA02.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217696_000IOF_N0010052AUT_04096_034085A03.png",
        "data\mars2020_mastcamz_sci_calibrated\data\0003\iof\ZL0_0003_0667217717_000IOF_N0010052AUT_04096_034085A03.png",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ]


    #######################################################################
def main():
    for idx, file in enumerate(os.scandir(PurePath(Path().cwd(), Path("./data/mars2020_mastcamz_sci_calibrated/data/0003/iof/horizon/")))):
        if idx > 0:
            if file.path.endswith(".png"):
                img = cv2.imread(file)
                hor, ver = horizon_tres(img, True)
                result = ""# classify_res(hor, ver)
                
                if result:
                    logger.info(f"{idx} valid photo {file.name} ")
                else:
                    logger.info(f"{idx} invalid photo {file.name}")

                # answer = horizon_test(img)
                # if answer:
                #     logger.info(f"{idx} Invalid photo {file.name}")
                # else:
                #     logger.info(f"{idx} Valid photo {file.name}")


if __name__ == "__main__":
    main()

    
# How to detect horizontal lines in an image using OpenCV and the Hough Transform: 



# https://stackoverflow.com/questions/62305018/challenge-how-to-get-the-4-sided-polygon-with-minimum-area
# https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python
