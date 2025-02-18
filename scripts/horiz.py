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
                return False # Found a horizon
            else:
                return True # No near-horizontal lines

        else:
            return True  # No lines detected

    except Exception as e: #Invalid due to errors
        logger.debug(f"Error processing image: {e}")
        return True
def main():
    for idx, file in enumerate(os.scandir("/home/andyh/github/mars_ml/mars2020_mastcamz_sci_calibrated/data/0003/iof/")):
        if file.path.endswith(".png"):
            img = cv2.imread(file)
            answer = horizon_test(img)
            if answer:
                logger.info(f"{idx} Invalid photo {file.name}")
            else:
                logger.info(f"{idx} Valid photo {file.name}")


if __name__ == "__main__":
    main()
