# https://essentia.upf.edu/essentia_python_examples.html

# 1. find longest performance
# 2. try alignemnet of all others with performance from 1.
# 3. store files with no alignment
# 4. repeat from 1. with unaligned files and reference next longest recording (TODO)

# needs 64GB memory


import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram
import itertools, os, json
import multiprocessing as mp
from subprocess import Popen, DEVNULL, PIPE
from test_subdtw_NEW_tuning import dtwstart
import numpy as np
from multiprocessing import shared_memory


#THREADS = 12
SCRIPT = '../../test_subdtw-NEW_tuning.command.py'
#SCRIPT = '../../test_subdtw-NEW_tuning_segments.command.py'
JSONFILE = 'test.json'
SR = 22050

RECSDIR = '/Volumes/Beratight2/SDTW/82-10-10'

CPUS = 24
THREADS_SIMILARITY = 24
THREADS_DTW = 6


class gl():
    shms = []


def loadRecordings():
    print('loading audio files')
    recordings = []
    folders = [os.path.join(RECSDIR, d) for d in os.listdir(RECSDIR) if os.path.isdir(os.path.join(RECSDIR, d))]
    for d in folders:
        content = []
        files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('flac', 'mp3'))]
        pool = mp.Pool(24)
        p = pool.map(loadFiles, files, chunksize=1)
        pool.close()
        pool.join()
        p.sort()
        recordings.append(p)
    recordings = sorted(recordings, key=lambda x: combinedLength(x))
    # shared memory for each audio file

    print('calculating hpcgrams')
    for i, rec in enumerate(recordings):
        etree_number = etreeNumber(rec[0][0])
        for j, f in enumerate(rec):
            #print(i, f[0])
            #gl.shms.append(shared_memory.SharedMemory(create=True, size=f[1].nbytes, name='{0}_{1}'.format(etree_number, j)))
            #s = np.ndarray(f[1].shape, dtype=np.float32, buffer=gl.shms[-1].buf)
            #s[:] = f[1][:]

            hpc = hpcpgram(f[1], sampleRate=SR)
            gl.shms.append(shared_memory.SharedMemory(create=True, size=hpc.nbytes, name='{0}_{1}'.format(etree_number, j)))
            s = np.ndarray(hpc.shape, dtype=np.float32, buffer=gl.shms[-1].buf)
            s[:] = hpc[:]

            recordings[i][j] = ('/'.join(f[0].split('/')[-2:]), j, f[1].shape) # store only filenames
    print('done')
    return recordings
    
def loadFiles(f):
    fs = estd.MonoLoader(filename=f, sampleRate=SR)()
    print(f)
    return (f, fs)

def combinedLength(x):
    l = 0
    for f in x: l += len(f[1])
    return l


def audioPairs(recs):
    rec1 = recs[0]
    rec2 = recs[1]
    files1 = [os.path.join(rec1, f) for f in os.listdir(rec1) if f.lower().endswith(('flac', 'mp3'))]
    files2 = [os.path.join(rec2, f) for f in os.listdir(rec2) if f.lower().endswith(('flac', 'mp3'))]
    if len(files2) > len(files1):
        temp = files1
        files1 = files2
        files2 = temp
    pairs = list(itertools.product(files1, files2))
    return pairs


def similarity(audiopair):
    #load audio from shared memory
    f1 = audiopair[0][0]
    f2 = audiopair[1][0]
    shmname1 = '{0}_{1}'.format(etreeNumber(audiopair[0][0]), audiopair[0][1])
    shmname2 = '{0}_{1}'.format(etreeNumber(audiopair[1][0]), audiopair[1][1])
    shm1 = shared_memory.SharedMemory(name=shmname1)
    shm2 = shared_memory.SharedMemory(name=shmname2)

    #file1 = np.ndarray(audiopair[0][2], dtype=np.float32, buffer=shm1.buf)
    #file2 = np.ndarray(audiopair[1][2], dtype=np.float32, buffer=shm2.buf)
    #file1_hpcp = hpcpgram(file1, sampleRate=SR)
    #file2_hpcp = hpcpgram(file2, sampleRate=SR)

    file1_hpcp = np.ndarray(audiopair[0][2], dtype=np.float32, buffer=shm1.buf)
    file2_hpcp = np.ndarray(audiopair[1][2], dtype=np.float32, buffer=shm2.buf)

    # Compute binary chroma cross similarity
    crp = estd.ChromaCrossSimilarity(frameStackSize=9,
                                    frameStackStride=1,
                                    binarizePercentile=0.095,
                                    oti=True)
    pair_crp = crp(file1_hpcp, file2_hpcp)
    #   Computing cover song similarity distance
    score_matrix, distance = estd.CoverSongSimilarity(disOnset=0.5,
                                                    disExtension=0.5,
                                                    alignmentType='serra09',
                                                    distanceType='asymmetric')(pair_crp)
    f1s = ('/').join(f1.split('/')[-2:])
    f2s = ('/').join(f2.split('/')[-2:])
    print(distance, f1s, f2s)
    return([f1, f2, distance])


def processResult(p):
    pdict = {}
    res = []
    for i in p:
        if i[0] not in pdict: pdict[i[0]] = []
        pdict[i[0]].append([i[2], i[1]])
    for k, v in pdict.items():
        #smin = min(v, key=lambda x: x[0])
        smin = sorted(v)[:2]
        #res.append([k] + smin)
        for i in smin: res.append([k] + i)
    return res


def runScript(f):
    file1 = f[0]
    file2 = f[2]
    #print(('/').join(file1.split('/')[-2:]), ('/').join(file2.split('/')[-2:]))
    print(file1, file2)
    #cmd = 'python {0} "{1}" "{2}"'.format(SCRIPT, file1, file2)
    resfiles = dtwstart(os.path.join(RECSDIR, file1), os.path.join(RECSDIR, file2))
    #print(resfiles)
    #print(cmd)
    #p = Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL).wait()
    return resfiles


def start():
    filenames = loadRecordings()
    # compare to longest
    audiopairs = []
    for n in range(0, len(filenames)-1):
        apairs = list(itertools.product(filenames[n], filenames[-1]))
        audiopairs += apairs
    process(audiopairs)


def process(apairs):
    manager = mp.Manager()
    count = manager.Value('i', 0)
    pool = mp.Pool(THREADS_SIMILARITY)
    #for i, a in enumerate(apairs):
    #    similarity(a)
    p = pool.map(similarity, apairs, chunksize=1)
    pool.close()
    pool.join()
    for s in gl.shms:
        s.close()
        s.unlink()
    res = processResult(p)
    pool = mp.Pool(THREADS_DTW)
    p = pool.map(runScript, res, chunksize=1)
    pool.close()
    pool.join()
    #json.dump(p, open(JSONFILE, 'w', encoding='utf-8'), sort_keys=True)


def etreeNumber(e):
    for j in e.split('/')[-2].split(('.')):
        try: return int(j)
        except: pass

    
if __name__ == '__main__':
    os.system('ulimit -n 30000')
    start()
    os.system('ulimit -n 256')
    os.system('stty sane')


