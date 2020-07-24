import os, sys
from subprocess import Popen, DEVNULL, PIPE
from tqdm import tqdm
import multiprocessing as mp

THREADS = 24


def getFiles():
    dots = []
    for root, dirs, files in os.walk('.'):
        for filename in files:
            if filename.endswith('.dot'):
                dots.append(os.path.join(root, filename))
    return dots

def forceAtlas(d):
    cmd = f'jython ./gephi/gephi.py "{d}"'
    p = Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL).wait()



def main():
    dots = getFiles()
    pool = mp.Pool(THREADS)
    p = list(tqdm(pool.imap_unordered(forceAtlas, dots), total=len(dots)))
    pool.close()
    pool.join()
    
main()
                