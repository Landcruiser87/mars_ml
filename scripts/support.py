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


################################# Globals ####################################
HEADERS = {
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36',
    'sec-ch-ua': '"Not)A;Brand";v="99", "Google Chrome";v="122", "Chromium";v="122"',
    'accept': 'application/json',
    'content-type': 'application/x-www-form-urlencoded',
}
POST_URL = 'https://pds-imaging.jpl.nasa.gov/api/search/atlas/_search?filter_path=hits.hits._source.archive,hits.hits._source.uri,hits.total,aggregations'
NAPTIME = 0.5

################################# Timing Func ####################################
def log_time(fn):
    """Decorator timing function.  Accepts any function and returns a logging
    statement with the amount of time it took to run. DJ, I use this code everywhere still.  Thank you bud!

    Args:
        fn (function): Input function you want to time
    """	
    def inner(*args, **kwargs):
        tnow = time.time()
        out = fn(*args, **kwargs)
        te = time.time()
        took = round(te - tnow, 2)
        if took <= 60:
            logging.warning(f"{fn.__name__} ran in {took:.2f}s")
        elif took <= 3600:
            logging.warning(f"{fn.__name__} ran in {(took)/60:.2f}m")		
        else:
            logging.warning(f"{fn.__name__} ran in {(took)/3600:.2f}h")
        return out
    return inner
################################# Size Funcs ############################################

def sizeofobject(totalsize)->str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(totalsize) < 1024:
            return f"{totalsize:4.1f} {unit}"
        totalsize /= 1024.0
    return f"{totalsize:.1f} PB"

################################# Logging funcs ####################################

def get_file_handler(log_dir:Path)->logging.FileHandler:
    """Assigns the saved file logger format and location to be saved

    Args:
        log_dir (Path): Path to where you want the log saved

    Returns:
        filehandler(handler): This will handle the logger's format and file management
    """	
    LOG_FORMAT = "%(asctime)s|%(levelname)-8s|%(lineno)-3d|%(funcName)-14s|%(message)-175s|" 
    current_date = time.strftime("%m-%d-%Y_%H-%M-%S")
    log_file = log_dir / f"{current_date}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, "%m-%d-%Y %H:%M:%S"))
    return file_handler

def get_rich_handler(console:Console):
    """Assigns the rich format that prints out to your terminal

    Args:
        console (Console): Reference to your terminal

    Returns:
        rh(RichHandler): This will format your terminal output
    """
    FORMAT_RICH = "|%(funcName)-14s|%(message)-175s "
    rh = RichHandler(level=logging.INFO, console=console)
    rh.setFormatter(logging.Formatter(FORMAT_RICH))
    return rh

def get_logger(log_dir:Path, console:Console)->logging.Logger:
    """Loads logger instance.  When given a path and access to the terminal output.  The logger will save a log of all records, as well as print it out to your terminal. Propogate set to False assigns all captured log messages to both handlers.

    Args:
        log_dir (Path): Path you want the logs saved
        console (Console): Reference to your terminal

    Returns:
        logger: Returns custom logger object.  Info level reporting with a file handler and rich handler to properly terminal print
    """	
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_file_handler(log_dir)) 
    logger.addHandler(get_rich_handler(console))  
    logger.propagate = False
    return logger

console = Console(color_system="truecolor")
logger = get_logger(Path("./data/logs"), console)


#FUNCTION sleep progbar
def mainspinner(console:Console, totalstops:int):
    """Load a rich Progress bar for however many categories that will be searched

    Args:
        console (Console): reference to the terminal
        totalstops (int): Amount of categories searched

    Returns:
        my_progress_bar (Progress): Progress bar for tracking overall progress
        jobtask (int): Job id for the main job
    """    
    my_progress_bar = Progress(
        SpinnerColumn("pong"),
        TextColumn("{task.description}"),
        BarColumn(),
        "time elapsed:",
        TextColumn("*"),
        TimeElapsedColumn(),
        TextColumn("*"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("*"),
        
        transient=True,
        console=console,
        refresh_per_second=10
    )
    jobtask = my_progress_bar.add_task("[green]Downloading Images", total=totalstops + 1)
    return my_progress_bar, jobtask

def add_spin_subt(prog:Progress, msg:str, howmany:int):
    """Adds a secondary job to the main progress bar that will take track a secondary job to the main progress should you need it. 

    Args:
        prog (Progress): Main progress bar
        msg (str): Message to update secondary progress bar
        howmany (int): How many tasks to add to sub spinner
    """
    #Add secondary task to progbar
    liltask = prog.add_task(f"[magenta]{msg}", total = howmany)
    return liltask
