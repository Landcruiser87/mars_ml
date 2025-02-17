<h1 align="center">
  <b>News by Rob</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python->3.11-blue" /></a>    
</p>

## Purpose
A dear friend of my fiancee, Rob, used to circulate and currate his own email
listings of different and interesting changes in US immigration law. Sadly, he
recently passed away and as a present to my fiancee I wanted to recreate that
information exchange so that she could both remember/honor him and stay up to
date on any changes that may influence her work.  In general, it is difficult to
track these changes as they're spread across 4 different sites and have
different categories that all need manual inspection for quality. This software
has the capability to ingest that information from all 4 sites RSS feeds.  (via
xml).  It then emails you any new findings after it scans the targeted RSS
feeds.  

If there are other immigration lawyers who are in need of this information.
Feel free to clone and use the repo for your own purposes.  You will need to
create a dummy gmail account for utilizing the email function.  Information as
to how to do that can be found here.  

[Real Python Article](https://realpython.com/python-send-email/)


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
$ .news_venv\Scripts\activate.bat

On Mac
$ source .news_venv/bin/activate
```

Before next step, ensure you see the environment name to the left of your
command prompt.  If you see it and the path file to your current directory, then
the environment is activated.   If you don't activate it, and start installing
things.  You'll install all the `requirements.txt` libraries into your `base
python environment.` Which will lead to dependency problems down the road.  I
promise. After that has been activated, go to your terminal and type `pip list`
to check your base python libraries.  Now is a good time to upgrade pip and
setuptools. As those should be the only two libraries you see on a clean python
installation.  If not...  well.

![Screenshot 2023-03-28 144052](https://user-images.githubusercontent.com/16505709/228358535-3364e0ea-b273-40b8-ab59-4dddf2f92ee2.png)


Next install the required libraries with the below pip command!

```
$ pip install -r requirements.txt
```

Order of operations of above terminal commands. 
- Open Terminal
- Clone repo
- Change directories
- Create venv
- Activate venv
- Upgrade pip (because reasons)
- Install libraries

## File Setup
While in root directory run commands below
```
$ mkdir data
$ mkdir scripts
$ mkdir secret
```

Within the secret folder, make a file called `login.txt`
Enter the following on the first 3 lines separated by a `colon`
1. username:str of email
2. pwd:str of pwd
3. recipient emails:sep str of emails

## Sites Searched

- Aggregate news data from these sources.  Looks as though some have an RSS feed you can tap right into.  Easier than scraping so going that way.
- Sites to search are
  - [UCIS](https://www.uscis.gov/news/rss-feed/59144)
  - [CBP](https://www.cbp.gov/rss)
  - [travel](https://travel.state.gov/_res/rss/TAsTWs.xml#.html)
  - [ICE](https://www.ice.gov/rss)

## TODO

- [ ] All Done!
- [ ] Format email header so that when a form update happens, include a notification in the email subject such as "FORMS UPDATE!!"
  