<h1 align="center">
  <b>Mars ML</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python->3.11-blue" /></a>    
</p>

## Purpose
This repo is meant to house various scripts to navigate the ATLAS database of images from the Mars 2020 mission.  

## Requirements
- Python >= 3.11

## Cloning and setting up environment.
Launch VSCode if that is IDE of choice.

```
`CTRL + SHIFT + ~` will open a terminal
Navigate to the directory where you want to clone the repo. 

$ git clone https://github.com/Landcruiser87/newsbyrob.git
$ cd newsbyrob
$ python -m venv .news_venv
(Or replace .news_venv with whatever you want to call your environment)	

On Windows
$ .venv\Scripts\activate.bat

On Mac / Linux
$ source .venv/bin/activate
```

# Project setup with Poetry

## How to check Poetry installation

In your terminal, navigate to your root folder.

If poetry is not installed, do so in order to continue
This will install version 1.7.0.  Adjust to your preference

```terminal
curl -sSL https://install.python-poetry.org | python3 - --version 1.7.0
```

To check if poetry is installed on your system. Type the following into your terminal

```terminal
poetry -V
```

if you see a `version` returned, you have Poetry installed.  The second command is to update poetry if its installed. (Always a good idea). If not, follow this [link](https://python-poetry.org/docs/) and follow installation commands for your systems requirements. If on windows, we recommend the `powershell` option for easiest installation. Using pip to install poetry will lead to problems down the road and we do not recommend that option.  It needs to be installed separately from your standard python installation to manage your many python installations.  `Note: Python 2.7 is not supported`

## Environment storage

Some prefer Poetry's default storage method of storing environments in one location on your system.  The default storage are nested under the `{cache_dir}/virtualenvs`.  

If you want to store you virtual environment locally.  Set this global configuration flag below once poetry is installed.  This will now search for whatever environments you have in the root folder before trying any global versions of the environment in the cache.

```terminal
poetry config virtualenvs.in-project true
```

For general instruction as to poetry's functionality and commands, please see read through poetry's [cli documentation](https://python-poetry.org/docs/cli/)
To select our poetry env.  If it doesn't have the path to your global python version, you can always make a native one.  Restart vscode and it will select that automatically after you have configured venv's to be stored locally above. 

```terminal
poetry env use python3.11

or 

python -m venv .venv
```

To install libraries

```terminal
poetry install
```

This will read from the poetry lock file that is included
in this repo and install all necessary packagage versions.  Should other
versions be needed, the project TOML file will be utilized and packages updated according to your system requirements.  

To view the current libraries installed

```terminal
poetry show
```

To view only top level library requirements

```terminal
poetry show -T

## File Setup
While in root directory run commands below
```
$ mkdir data
$ mkdir secret
```


## TODO
  