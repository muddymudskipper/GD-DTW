
import sys
from make_folder_dict import dateDict
from subprocess import Popen
from makedot import makeDotStart

YEAR = sys.argv[1]

try:
    START = int(sys.argv[2])     # start with this item in dates list
except:
    START = 0

dates = sorted(list(filter(lambda x: x.split('-')[0] == YEAR, dateDict())))
tmpl = 'python align_match.py {0}'

for d in dates[START:]:
    cmd = tmpl.format(d)
    print(cmd)
    Popen(cmd, shell=True).wait()
    makeDotStart(d)
    


