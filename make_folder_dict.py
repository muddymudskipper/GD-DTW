# make dict for folders by concerty date
import os, re, pickle



SBDPATH = '/Volumes/gspeed1/thomasw/grateful_dead/lma_soundboards/sbd'
LMAPATH = '/Volumes/gspeed1/thomasw/grateful_dead/lma'

DIRS = [os.path.join(SBDPATH, d) for d in os.listdir(SBDPATH) if os.path.isdir(os.path.join(SBDPATH, d))] + [os.path.join(LMAPATH, d) for d in os.listdir(LMAPATH) if os.path.isdir(os.path.join(LMAPATH, d))]


datedict = {}
for d in DIRS:
    if 'flac24' in d: continue
    print(d)
    date = d.split('/')[-1].split('.')[0]
    try:
        date = re.findall(r'\d{2}-\d{2}-\d{2}', date)[0]
        if date not in datedict: datedict[date] = []
        datedict[date].append(d)
    except: pass

pickle.dump(datedict, open('date_folder_dict.pickle', 'wb'))

