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
from rich.progress import (
    Progress,
    BarColumn,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn
)
from rich.logging import RichHandler
from rich.console import Console
from PIL import Image
from support import logger


#FUNCTION horizon test
def horizon_test(img:np.array) -> bool:
    ######## Horizon test ############
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Edge Detection (Canny or Sobel)
        # edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # Adjust thresholds as needed
        # OR, if Canny is too sensitive:
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # Adjust ksize
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edges = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)

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

        # Enhance contrast (important for Mars images)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Adjust parameters as needed
        enhanced_gray = clahe.apply(gray)

        # Improved edge detection (adjust parameters based on image characteristics)
        edges = cv2.Canny(enhanced_gray, 50, 150, apertureSize=3)

        # Hough Line Transform (tune parameters)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150) # Reduced threshold

        horizontal_lines = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                # More robust horizontal line detection using cosine and tolerance
                if np.cos(theta) > 0.95: # cosine close to 1 means horizontal
                    horizontal_lines.append((rho, theta))

        if display_res:
            # Draw detected lines (more efficient drawing)
            for rho, theta in horizontal_lines:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red lines

            cv2.namedWindow("floorlava")
            cv2.setWindowProperty("floorlava", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Horizontal Lines', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return horizontal_lines
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def main():
    for idx, file in enumerate(os.scandir(PurePath(Path().cwd(), Path("./data/mars2020_mastcamz_sci_calibrated/data/0003/iof/")))):
        if file.path.endswith(".png"):
            img = cv2.imread(file)
            lines_dos = horizon_dos(img, True)
            if lines_dos is not None:
                logger.info(f"{idx} valid photo {file.name}")
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
