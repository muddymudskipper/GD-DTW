# make dict for folders by concerty date
import os, re, pickle



SBDPATH = '/Volumes/gspeed1/thomasw/grateful_dead/lma_soundboards/sbd'
LMAPATH = '/Volumes/gspeed1/thomasw/grateful_dead/lma'

DIRS = [os.path.join(SBDPATH, d) for d in os.listdir(SBDPATH) if os.path.isdir(os.path.join(SBDPATH, d))] + [os.path.join(LMAPATH, d) for d in os.listdir(LMAPATH) if os.path.isdir(os.path.join(LMAPATH, d))]


datedict = {}


for d in DIRS:
    #if 'flac24' in d: continue
    


    date = d.split('/')[-1].split('.')[0]
    try:
        date = re.findall(r'\d{2}-\d{2}-\d{2}', date)[0]
        if date not in datedict: datedict[date] = []
        
        if d.replace('flac24', 'flac16') in datedict[date]:
            continue
        elif d.replace('flac1648', 'flac16') in datedict[date]:
            continue
        elif d.replace('flac24', 'flac1648') in datedict[date]:
            continue
        if d.replace('flac16', 'flac24') in datedict[date]:
            datedict[date].remove(d)
        elif d.replace('flac16', 'flac1648') in datedict[date]:
            datedict[date].remove(d)
        elif d.replace('flac1648', 'flac24') in datedict[date]:
            datedict[date].remove(d)


        datedict[date].append(d)
    except: pass

pickle.dump(datedict, open('date_folder_dict.pickle', 'wb'))

