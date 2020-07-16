
import os, re

SBDPATH = '/Volumes/gspeed1/thomasw/grateful_dead/lma_soundboards/sbd'
LMAPATH = '/Volumes/gspeed1/thomasw/grateful_dead/lma'


replace = { 
    #'flac16': ['flac24', 'flac2448', 'flac2496', 'flac1696', 'flac1648', 'shnf', 'flac1644', '1644'],
    'flac1644': ['flac24', 'flac2448', 'flac2496', 'flac1696', 'flac1648', 'shnf'],
    'shnf': ['flac24', 'flac2448', 'flac2496', 'flac1696', 'flac1648'],
    'flac1648': ['flac24', 'flac2448', 'flac2496', 'flac1696'],
    'flac1696': ['flac24', 'flac2448', 'flac2496'],
    'flac24': ['flac2448', 'flac2496'],
    'flac2448': ['flac2496']
    }

ignore = ['vobf', 'dvda', 'dvdf', 'sirmickflac1648', 'fkac16', 'na', '127416', '127417', '127418', 'chuckm', '127360']

DIRS = [os.path.join(SBDPATH, d) for d in os.listdir(SBDPATH) if os.path.isdir(os.path.join(SBDPATH, d))] + [os.path.join(LMAPATH, d) for d in os.listdir(LMAPATH) if os.path.isdir(os.path.join(LMAPATH, d))]


def dateDict():
    datedict = {}

    for d in DIRS:
        #print(d)
        etree = d.split('/')[-1].split('.')

        try:
            if etree[-1] in ignore: continue
            if etree[-1] == '1644': etree[-1] = 'flac1644'
            if etree[-1] == '2496': etree[-1] = 'flac2496'
            frmt = list(filter(lambda x: x.lower().startswith(('flac', 'shnf')), etree))[0]
        except:
            pass
            #print(etree)
            #sys.exit()
        
        sortname = '.'.join(filter(lambda x: not (x.isdigit() or x.lower().startswith(('shnf', 'flac'))), etree))


        if sortname in datedict and frmt.lower() in replace:
            if datedict[sortname]['format'] in replace[frmt.lower()]:
                datedict[sortname]['folder'] = d
                datedict[sortname]['format'] = frmt.lower()
        else:
            datedict[sortname] = { 'folder': d, 'format': frmt.lower()}
            

    newdict = {}

    for k, v in datedict.items():
        try:
            date = k.split('/')[-1].split('.')[0]
            date = re.findall(r'\d{2}-\d{2}-\d{2}', date)[0]
            if date not in newdict: 
                newdict[date] = []
            newdict[date].append(v['folder'])
        except: pass
    
    return newdict


