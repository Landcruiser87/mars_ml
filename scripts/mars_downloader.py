#Import libraries
import numpy as np
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
NAPTIME = 0.75

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


################################# Rich Spinner Control ####################################

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
        TextColumn("{task.description}"),
        SpinnerColumn("pong"),
        BarColumn(),
        TextColumn("*"),
        "time elapsed:",
        TextColumn("*"),
        TimeElapsedColumn(),
        TextColumn("*"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
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

################################  Saving functions ############################################

#CLASS Numpy encoder
class NumpyArrayEncoder(json.JSONEncoder):
    """Custom numpy JSON Encoder.  Takes in any type from an array and formats it to something that can be JSON serialized.
    Source Code found here.  https://pynative.com/python-serialize-numpy-ndarray-into-json/
    Args:
        json (object): Json serialized format
    """	
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, str):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return datetime.datetime.strftime(obj, "%m-%d-%Y_%H-%M-%S")
        else:
            return super(NumpyArrayEncoder, self).default(obj)
        
#FUNCTION Save Configs
def save_json(spath:str, nasa_j:dict):
    """This function saves the configs dictionary to a JSON file.     

    Args:
        spath (str): Path to save the json
        nasa_j (dict): Dictionary to save
    """
    out_json = json.dumps(nasa_j, indent=2, cls=NumpyArrayEncoder)
    with open(spath, "w") as out_f:
        out_f.write(out_json)


################################ API functions ############################################
#FUNCTION Ping nasa
def ping_that_nasa(parent_uri:str):
    """Function that pings the Atlas API to read the folder structure

    Args:
        parent_uri (str): as stated

    Returns:
        _type_: _description_
    """    
    # uri = "mars2020_mastcamz_sci_calibrated/data"
    # release_id = 0
    # resize = False
    # url = f"https://pds-imaging.jpl.nasa.gov/api/data/{uri}/::{release_id}?"
    # url = f"https://pds-imaging.jpl.nasa.gov/api/data/{uri}/::{release_id}?:{resize}?"
    #'https://pds-imaging.jpl.nasa.gov/api/search/atlas/_search?filter_path=hits.hits._source.archive,hits.hits._source.uri,hits.total,aggregations' -H 'accept: application/json' --data-raw '{"query":{"bool":{"must":[{"match":{"archive.parent_uri":"atlas:pds4:mars_2020:perseverance:/mars2020_cachecam_ops_calibrated/data"}}]}},"sort":[{"archive.name":"asc"}]}'

    data = {
        "query": {
            "bool": {
                "must":[
                    {"match":{"archive.parent_uri":f"atlas:pds4:mars_2020:perseverance:{parent_uri}"}}
                ]
            }
        },
        "sort":[{"archive.name":"asc"}],
        "size":10000
    }
    # data = '{"query":{"bool":{"must":[{"match":{"archive.parent_uri":"atlas:pds4:mars_2020:perseverance:/mars2020_mastcamz_sci_calibrated/data"}}]}},"sort":[{"archive.name":"asc"}],"size":10000}'

    response = requests.post(
        url = POST_URL,
        headers=HEADERS,
        data=json.dumps(data),
    )

    #Just in case we piss someone off
    if response.status_code != 200:
        # If there's an error, log it and return no data for that site
        logger.warning(f'Status code: {response.status_code}')
        logger.warning(f'Reason: {response.reason}')
        return None
    
    #Quick nap so we don't hammer servers
    time.sleep(NAPTIME)
    resp_json = response.json()
    return resp_json

#FUNCTION Download Image
def download_image(image_uri:str, save_path:Path, item_uri:str, release_id:int=0, resize:str="lg"):
    """This function will download the individual image to the directory

    Args:
        image_uri (str) : full uri for the image to be downloaded
        save_path (Path): full save path
        item_uri (str)  : partial folder location of the files
        release_id (int): Release version of the file (not working)
        resize (str)    : Whether to resize the file (not working)
    """
    # url = f"https://pds-imaging.jpl.nasa.gov/api/data/{image_uri}"
    # url = f"https://pds-imaging.jpl.nasa.gov/api/data/{image_uri}::{release_id}?"
    # url = f"https://pds-imaging.jpl.nasa.gov/api/data/{image_uri}::{release_id}:{resize}?"

    custom_headers = {
        'accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        'accept-language': 'en-US,en;q=0.9',
        'origin': 'https://pds-imaging.jpl.nasa.gov',
        'priority': 'u=1, i',
        'referer': 'https://pds-imaging.jpl.nasa.gov/',
        'sec-ch-ua': '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
        'sec-ch-ua-mobile': '?1',
        'sec-ch-ua-platform': '"Android"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'cross-site',
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Mobile Safari/537.36',
    }

    # /mars2020_mastcamz_sci_calibrated/data/0001/iof/ZL0_0001_0667035659_000IOF_N0010052AUT_04096_0260LMA03.IMG-
        #need to replace img with png and data with browse.
    # /mars2020_mastcamz_sci_calibrated/browse/0001/iof/ZL0_0001_0667035659_000IOF_N0010052AUT_04096_0260LMA03.png
        #  f"https://pds-imaging.jpl.nasa.gov/api/data/{image_uri}"

    #'referer': 'https://pds-imaging.jpl.nasa.gov/beta/archive-explorer?mission=mars_2020&instrument=mastcamz&bundle=mars2020_mastcamz_sci_calibrated&uri=atlas:pds4:mars_2020:perseverance:/mars2020_mastcamz_sci_calibrated/data/0001/iof/ZL0_0001_0667035659_000IOF_N0010052AUT_04096_0260LMA03.IMG-',
    # url = 'https://pds-imaging.jpl.nasa.gov/archive/m20/cumulative/' 
    

    archive_root = 'https://pds-imaging.jpl.nasa.gov/archive/m20/cumulative' 
    url = archive_root + item_uri
    logger.debug(f"requesting {url}")
    response = requests.get(url=url, headers=custom_headers, stream=True)

    # response = requests.get(url=url, stream=True)


    #Just in case we piss someone off
    if response.status_code != 200:
        # If there's an error, log it and return no data for that request
        logger.warning(f'Status code: {response.status_code}')
        logger.warning(f'Reason: {response.reason}')
        logger.warning(f"Image {image_uri} not downloaded")
        return 
    
    #Quick nap so we don't hammer NASA servers
    time.sleep(NAPTIME)
    
    #Save file
    with open(save_path, "wb") as f:
        f.write(response.content)
    
    #Open it and check if its blank. 
    try:
        img = Image.open(save_path)
        img_array = np.array(img)
        if img_array.ndim == 2: 
            all_zeros = np.all(img_array == 0)
        elif img_array.ndim == 3:  
            all_zeros = np.all(img_array[:, :, :3] == 0)
        else:
            return None

        return all_zeros
        
    except FileNotFoundError:
        logger.warning(f"Error: File not found at {save_path}")
        return None
    except Exception as e: 
        logger.warning(f"An error occurred: {e}")
        return None

#FUNCTION Intial Scan
def inital_scan(base_parent_uri:str):
    """This just does an initial scan to get a folder count in /data

    Args:
        base_parent_uri (str): starting URI point

    Raises:
        IndexError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """    
    try:
        nasa_json = ping_that_nasa(base_parent_uri)

    except IndexError:
        logger.warning("Gah something happened, run away!")
        raise IndexError("An Index of errors, I blame myself")

    else:
        total = nasa_json["hits"]["total"]["value"]
        question = f"\n\nDo you want to scan {total} folders and files?\nIf so enter a file path ie:\n./data/nasa\nOtherwise type no to exit\n"
        fold_choice = console.input(f"{question}")
        if fold_choice == "no":
            logger.warning("Run Away!!!!!")
            raise ValueError("Input Error!")
        else:
            save_fp = PurePath(Path.cwd(), Path(fold_choice)) #Path("./secret")
            if Path(save_fp).exists():
                logger.info("Let the downloads begin!")
                return save_fp, total
            else:
                logger.warning("File Location doesn't exist")
                raise ValueError("Select a location that exists")

#FUNCTION Recurse Tree
def recurse_tree(parent_uri:str):
    """Recursive internal function that descends the folder structure by file type

    Args:
        parent_uri (str): URI that you started with

    Returns:
        directory (dict): Don't really need this anymore, but not sure if the recursion needs it. 
    """
    global total_mem
    try:
        logger.info(f"ping {parent_uri}")
        data = ping_that_nasa(parent_uri)
        directory, files = {}, []
        pileofsomething = data["hits"]["hits"]
        prog.update(task_id=task, description=f"[green]searching [red]{parent_uri}[/red]", advance=1)
        make_path = PurePath(Path(save_path), Path(f"./{parent_uri}") )
        os.makedirs(make_path, exist_ok=True)
        logger.debug(f"new dir -> {make_path}")
        
        typecheck = all([item["_source"]["archive"]["fs_type"]=="file" for item in pileofsomething])
        if typecheck:
            liljob = add_spin_subt(prog, f"downloading images", len(pileofsomething) // 2) #Div by 2 because we don't want the xml files

        for item in pileofsomething:
            uri = item["_source"]["uri"]
            item_uri = uri.split(":")[-1]
            item_type = item["_source"]["archive"]["fs_type"]
            item_name = PurePath(item_uri).name if item_type=="file" else item["_source"]["archive"]["name"]
            item_ext = item["_source"]["archive"].get("file_extension")
            if not item_name:
                logger.warning(f"fname mising in {item_uri}")
                item_name = "file_from_uri"
            
            if item_type == "directory":
                logger.debug("descend w recursion")
                subdir = recurse_tree(item_uri)
                directory[item_name] = subdir

            elif item_type  == "file":
                if item_ext == "img":
                    item_name = item_name.replace(".IMG", ".png")
                    item_sp = PurePath(Path(make_path), Path(item_name))
                    if Path(item_sp).exists():
                        prog.update(liljob, description=f"[bold green]{item_name} stored locally[/bold green]", advance=1) 
                        logger.debug(f"skipping download on {item_name}")   
                        time.sleep(0.1)
                        continue
                    
                    total_mem += item["_source"]["archive"]["size"]
                    prog.update(liljob, description=f"[yellow]{sizeofobject(total_mem)}[/yellow] [green]downloaded[/green] [red]{item_name}[/red]", advance=1)
                    files.append(item_name)
                    #Try downloading the image
                    try:
                        item_uri = item_uri.replace("data", "browse").replace(".IMG", ".png")
                        invalid = download_image(uri, item_sp, item_uri)
                        if invalid:
                            logger.warning(f"{item_name} is blank")
                            item["_source"]["archive"]["valid"] = False
                        else:
                            logger.debug(f"downloaded file {item_name} from {parent_uri}")
                            item["_source"]["archive"]["valid"] = True

                    except Exception as e:
                        logger.warning(f"Error downloading {item_uri}: {e}")
                    
                    #Try saving the meta data
                    try:
                        save_json(PurePath(Path(make_path), Path(item_name.replace(".png", ".json"))), item["_source"]["archive"])
                        logger.debug(f"json saved {item_name}")

                    except Exception as e:
                        logger.warning(f"Error saving {item_uri}: {e}")

            else:
                logger.warning(f"unknown item type: {item_type}")

        directory[parent_uri] = files
        if typecheck:
            prog.update(liljob, visible=False)

        return directory

    except requests.exceptions.RequestException as e:
        logger.warning(f"error requesting data: {e}")
        return None  
    except json.JSONDecodeError as e:
        logger.warning(f"error decoding JSON: {e}")
        return None
    except Exception as e:
        logger.warning(f"a general error has occured {e}")
        return None
#################################### MAIN ##################################    

@log_time
def main():
    #load logger and funcs
    global logger, console
    console = Console(color_system="truecolor")
    logger = get_logger(Path().cwd(), console)
    
    #Load baseparent uri
    base_parent_uri = "/mars2020_mastcamz_sci_calibrated/data" #This is what changes
    global prog, task, save_path, total_mem
    total_mem = 0
    save_path, total_files = inital_scan(base_parent_uri)
    prog, task = mainspinner(console, total_files*2) #x2 because all our sub folders have 2 folders (iof, rad)
    with prog:
        directory = recurse_tree(base_parent_uri)
    
    #?Might need to unpack the directory dict.  Whoever runs this first let me know if this prints to the log.  Lol
    logger.debug(f"directory structure downloaded\n{directory}") 
    logger.warning("YOU'VE DONE IT.  All files downloaded.  TIME FOR A BEER")

if __name__ == "__main__":
    main()
