# https://essentia.upf.edu/essentia_python_examples.html

# 1. find longest performance
# 2. try alignemnet of all others with performance from 1.
# 3. store files with no alignment
# 4. repeat from 1. with unaligned files and reference next longest recording

# needs 64GB memory

import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram
import itertools, os, json, progressbar
import multiprocessing as mp
from multiprocessing import shared_memory
from subprocess import Popen, DEVNULL, PIPE
import numpy as np
from test_subdtw_NEW_tuning import dtwstart
from uuid import uuid4
import librosa
#import pickle

SR = 22050

RECSDIR = '/Volumes/Beratight2/SDTW/82-10-10'

CPUS = 24
THREADS_SIMILARITY = 24
THREADS_DTW = 6

DTWFRAMESIZE = 512


class gl():
    shms = []


def loadRecordings():
    print('loading audio files')
    recordings = []
    folders = [os.path.join(RECSDIR, d) for d in os.listdir(RECSDIR) if os.path.isdir(os.path.join(RECSDIR, d))]
    for i, d in enumerate(folders[:2]):
        content = []
        files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('flac', 'mp3'))]
        pool = mp.Pool(24)
        #p = pool.starmap(loadFiles, zip(files[:3], itertools.repeat(i)), chunksize=1)
        p = pool.map(loadFiles, files[:3], chunksize=1)
        pool.close()
        pool.join()
        p.sort()
        recordings.append(p)
    recordings = sorted(recordings, key=lambda x: combinedLength(x))
    # shared memory for each audio file
    for i, rec in enumerate(recordings):
        etree_number = etreeNumber(rec[0][0])
        for j, f in enumerate(rec):
            #print(i, f[0])
            gl.shms.append(shared_memory.SharedMemory(create=True, size=f[1].nbytes, name='{0}_{1}_audio'.format(etree_number, j)))
            s = np.ndarray(f[1].shape, dtype=np.float32, buffer=gl.shms[-1].buf)
            s[:] = f[1][:]

            gl.shms.append(shared_memory.SharedMemory(create=True, size=f[2].nbytes, name='{0}_{1}_hpcg'.format(etree_number, j)))
            s = np.ndarray(f[2].shape, dtype=np.float32, buffer=gl.shms[-1].buf)
            s[:] = f[2][:]

            #if i > 0:
            #    gl.shms.append(shared_memory.SharedMemory(create=True, size=f[3].nbytes, name='{0}_{1}_chroma'.format(etree_number, j)))
            #    s = np.ndarray(f[3].shape, dtype=np.float32, buffer=gl.shms[-1].buf)
            #    s[:] = f[3][:]
            #    recordings[i][j] = ('/'.join(f[0].split('/')[-2:]), j, f[1].shape, f[2].shape, f[3].shape)

            #else:
            #    recordings[i][j] = ('/'.join(f[0].split('/')[-2:]), j, f[1].shape, f[2].shape)
            recordings[i][j] = ('/'.join(f[0].split('/')[-2:]), j, f[1].shape, f[2].shape)
            # 0: filename
            # 1: index
            # 2: audio shape
            # 3: hpcg shape
            # (4: chroma shape)  # not first (shortest) because it will always be resampled in subdtw script

    return recordings

#def loadFiles(f, i):
def loadFiles(f):
    fs = estd.MonoLoader(filename=f, sampleRate=SR)()
    hpc = hpcpgram(fs, sampleRate=SR)
    #if i > 0:
    #    chroma = librosa.feature.chroma_cens(y=fs, sr=SR, hop_length=DTWFRAMESIZE, win_len_smooth=21)
    #    print(f)
    #    return (f, fs, hpc, chroma)
    #else:
    #    print(f)
    #    return (f, fs, hpc)
    print(f)
    return (f, fs, hpc)
    
    
def combinedLength(x):
    l = 0
    for f in x: l += len(f[2])
    return l


def similarity(audiopair):
    #load audio from shared memory
    f1 = audiopair[0][0]
    f2 = audiopair[1][0]
    shmname1 = '{0}_{1}_hpcg'.format(etreeNumber(f1), audiopair[0][1])
    shmname2 = '{0}_{1}_hpcg'.format(etreeNumber(f2), audiopair[1][1])
    shm1 = shared_memory.SharedMemory(name=shmname1)
    shm2 = shared_memory.SharedMemory(name=shmname2)

    file1_hpcp = np.ndarray(audiopair[0][3], dtype=np.float32, buffer=shm1.buf)
    file2_hpcp = np.ndarray(audiopair[1][3], dtype=np.float32, buffer=shm2.buf)

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

    #return([f1, f2, distance])
    return([audiopair[0], audiopair[1], distance])

def processResult(p):
    pdict = {}
    res = []
    for i in p:
        if i[0] not in pdict: pdict[i[0]] = []
        pdict[i[0]].append([i[2], i[1]])
    for k, v in pdict.items():
        #smin = sorted(v)[:2]        # store first 2 min distances
        smin = sorted(v)[:1]        # for testing
        for i in smin: res.append([k] + [i[1]]) # store recording pairs with all info
    print(res)
    return res


def runScript(f):
    file1 = f[0][0]
    file2 = f[1][0]
    print(file1, file2)
    #resfile = dtwstart(os.path.join(RECSDIR, file1), os.path.join(RECSDIR, file2))

    resfile = dtwstart(f[0], f[1], RECSDIR)
    return resfile


def start():
    filenames = loadRecordings()
    # compare to longest first

    matched_files = []
    for i in range(1,len(filenames)):
        audiopairs = []
        print('RUN', i)
        for n in range(0, len(filenames)-i):
            apairs = list(itertools.product(filenames[n], filenames[-i]))
            audiopairs += apairs
            audiopairs = list(filter(lambda x: x[0][0] not in matched_files, audiopairs))
        #print(audiopairs)
        if len(audiopairs) > 0: matched_files += process(audiopairs)
        else: break

        #break

    
def process(apairs):
    #manager = mp.Manager()
    pool = mp.Pool(THREADS_SIMILARITY)
    p = pool.map(similarity, apairs, chunksize=1)
    pool.close()
    pool.join()
    #for s in gl.shms:
    #    s.close()
    #    s.unlink()
    res = processResult(p)
    #res = pickle.load(open('similaritymin.pickle', 'rb'))
    #pickle.dump(res, open('similaritymin_test.pickle', 'wb'))
    #return
    pool = mp.Pool(THREADS_DTW)
    p = pool.map(runScript, res, chunksize=1)
    pool.close()
    pool.join()
    
    matched_files = list(set(filter(lambda x: x != None, p)))

    return matched_files
    


def etreeNumber(e):
    for j in e.split('/')[-2].split(('.')):
        try: return int(j)
        except: pass

    
if __name__ == '__main__':
    os.system('ulimit -n 30000')
    start()
    for s in gl.shms:
        s.close()
        s.unlink()
    os.system('ulimit -n 256')
    os.system('stty sane')


