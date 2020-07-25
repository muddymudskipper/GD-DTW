import os, sys
from subprocess import Popen, DEVNULL, PIPE
from tqdm import tqdm
import multiprocessing as mp

THREADS = 24
FOLDER = 'results'

try:
    if sys.argv[1] == 'all':
        ALL = True
except:
    ALL = False


def getFiles():
    dots = []
    dirs = [d for d in os.listdir(FOLDER) if os.path.isdir(os.path.join(FOLDER, d))]
    for d in dirs:
        files = [f for f in os.listdir(os.path.join(FOLDER, d))]
        if ALL or len(list(filter(lambda x: x.endswith('gephi.pdf'), files))) == 0:
            dot = list(filter(lambda x: x.endswith('.dot'), files))
            if dot:
                df = os.path.join(FOLDER, d, dot[0])
                print(df)
                dots.append(df)
    return dots


def forceAtlas(d):
    cmd = f'jython ./gephi/gephi.py "{d}"'
    p = Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL).wait()


def start():
    dots = getFiles()
    pool = mp.Pool(min([len(dots), THREADS]))
    p = list(tqdm(pool.imap_unordered(forceAtlas, dots), total=len(dots)))
    pool.close()
    pool.join()


if __name__ == '__main__':
    start()
                