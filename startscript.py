
import sys
from make_folder_dict import dateDict
from subprocess import Popen, PIPE
from makedot import makeDotStart

YEAR = sys.argv[1]


date_dict = dateDict()

dates = sorted(list(filter(lambda x: x.split('-')[0] == YEAR, date_dict)))

try:
    START = dates.index(sys.argv[2])     # start with this date
except:
    START = 0


tmpl = 'python align_match.py {0}'


def checkEtreeNumbers(folders):
    # skip dates where dupliacte etree numbers exist in the list of recordings
    # e.g.  gd1990-03-29.123925.mk4.flac16
    #       gd1990-03-29.127385.mtx.eichorn.flac16
    # TODO: save unique etree number in files list
    es = []
    for r in folders:
        e = etreeNumber(r + '/')
        #print(e)
        if e in es:
            return False
        else:
            es.append(e)
    return True


def etreeNumber(e):
    for j in e.split('.'):
        try: return int(j)
        except: pass


for d in dates[START:]:    
    if not checkEtreeNumbers(date_dict[d]):
        print('SKIPPING: DUPLICATE IDS')
        continue
    cmd = tmpl.format(d)
    print(cmd)
    Popen(cmd, shell=True).communicate()

    makeDotStart(d)



