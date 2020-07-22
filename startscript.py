
import sys
from make_folder_dict import dateDict
from subprocess import Popen, PIPE
from makedot import makeDotStart

YEAR = sys.argv[1]


dates = sorted(list(filter(lambda x: x.split('-')[0] == YEAR, dateDict())))

try:
    START = dates.index(sys.argv[2])     # start with this date
except:
    START = 0


tmpl = 'python align_match.py {0}'

for d in dates[START:]:
    cmd = tmpl.format(d)
    print(cmd)
    stdout, std = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True).communicate()
    if stdout == b'SKIPPING: DUPLICATE IDS\n':
        print('SKIPPING: DUPLICATE IDS')
        continue
    makeDotStart(d)



