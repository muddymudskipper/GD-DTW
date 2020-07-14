# https://essentia.upf.edu/essentia_python_examples.html

# 1. find longest performance
# 2. try alignemnet of all others with performance from 1.
# 3. store files with no alignment
# 4. repeat from 1. with unaligned files and reference next longest recording

# needs 64GB memory

import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram
import itertools, os, json, sys
import multiprocessing as mp
from multiprocessing import shared_memory
from subprocess import Popen, DEVNULL, PIPE
import numpy as np
from test_subdtw_NEW_tuning import dtwstart
from uuid import uuid4
from librosa.feature import chroma_cens
from tqdm import tqdm
from subprocess import Popen, DEVNULL, PIPE
import pickle
from task_processor import TaskProcessor
from multiprocessing.managers import SharedMemoryManager


SR = 22050

#RECSDIR = '/Volumes/Beratight2/SDTW/82-07-29'
#DIR = '/Volumes/Journal/Documents/OneDrive/OneDrive - Queen Mary, University of London/projects/SDTW/'
#TEMPDIR = DIR + 'temp/'


#MAX_MEM_GB = 50
MIN_MEM = 60
MAX_MEM = 70
DATE = sys.argv[1]
#DATE = '82-07-29'
TEMPDIR = 'temp'
DSTDIR = os.path.join('results', DATE)

THREADS_LOADING = 24
THREADS_CHROMA = 24
THREADS_SIMILARITY = 24
THREADS_DTW = 10

DTWFRAMESIZE = 512


class gl():
    shms = []


def loadRecordings():
    print('loading audio files')

    #folders = pickle.load(open('date_folder_dict.pickle', 'rb'))[DATE]
    #folders = [os.path.join(RECSDIR, d) for d in os.listdir(RECSDIR) if os.path.isdir(os.path.join(RECSDIR, d))]

    
    recordings = pickle.load(open('recordings.pickle', 'rb'))
    '''
    recordings = []
    for d in folders[:5]:
        print('loading files for', d.split('/')[-1])
        files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('flac', 'mp3', 'shn'))]
        pool = mp.Pool(THREADS_LOADING)
        p = list(tqdm(pool.imap(loadFiles, files[:5]), total=len(files[:5])))
        pool.close()
        pool.join()
        p = list(filter(lambda x: x != None, p)) # remove None type for unloadable files
        p.sort()
        recordings.append(p)
    recordings = sorted(recordings, key=lambda x: combinedLength(x))
    # shared memory for each audio file

    #pickle.dump(recordings, open('recordings.pickle', 'wb'))
    '''

    for i, rec in enumerate(recordings):
        etree_number = etreeNumber(rec[0][0])
        for j, f in enumerate(rec):
            #print(i, f[0])
            gl.shms.append(shared_memory.SharedMemory(create=True, size=f[1].nbytes, name='{0}_{1}_audio'.format(etree_number, j)))
            s = np.ndarray(f[1].shape, dtype=np.float32, buffer=gl.shms[-1].buf)
            s[:] = f[1][:]

            gl.shms.append(shared_memory.SharedMemory(create=True, size=f[2].nbytes, name='{0}_{1}_hpcp'.format(etree_number, j)))
            s = np.ndarray(f[2].shape, dtype=np.float32, buffer=gl.shms[-1].buf)
            s[:] = f[2][:]

            recordings[i][j] = ('/'.join(f[0].split('/')[-2:]), j, f[1].shape, f[2].shape)

            #recordings[i][j] = (f[0], j, f[1].shape, f[2].shape)
            # 0: filename
            # 1: index
            # 2: audio shape
            # 3: hpcp shape

    return recordings


def loadFiles(f):
    try:
        if f.endswith('.shn'):
            _f = os.path.join(TEMPDIR, str(uuid4()) + '.wav')
            cmd = 'shorten -x "{0}" "{1}"'.format(f, _f)
            p = Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL).wait()
        else: _f = f
        fs = estd.MonoLoader(filename=_f, sampleRate=SR)()
        if f.endswith('.shn'): os.remove(_f)
        hpc = hpcpgram(fs, sampleRate=SR)
        #print(f)
        return (f, fs, hpc)
    except: pass
    
    
def combinedLength(x):
    l = 0
    for f in x: l += len(f[2])
    return l


def similarity(audiopair, q):
    #load audio from shared memory
    f1 = audiopair[0][0]
    f2 = audiopair[1][0]
    shmname1 = '{0}_{1}_hpcp'.format(etreeNumber(f1), audiopair[0][1])
    shmname2 = '{0}_{1}_hpcp'.format(etreeNumber(f2), audiopair[1][1])
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
    #f1s = ('/').join(f1.split('/')[-2:])
    #f2s = ('/').join(f2.split('/')[-2:])
    #print(distance, f1s, f2s)
    q.put([audiopair[0], audiopair[1], distance])
    #return([audiopair[0], audiopair[1], distance])


def processResult(p):
    pdict = {}
    res = []
    for i in p:
        if i[0] not in pdict: pdict[i[0]] = []
        pdict[i[0]].append([i[2], i[1]])
    for k, v in pdict.items():
        smin = sorted(v)[:2]        # store first 2 min distances
        #smin = sorted(v)[:1]        # for testing
        for i in smin: res.append([k] + [i[1]]) # store recording pairs with all info

    # calculate chromas for all audio files that are second input for dtw
    audio_compared_to = []
    [audio_compared_to.append(f[1]) for f in res]
    audio_compared_to = list(set(audio_compared_to))
    chroma_shapes = getChromas(audio_compared_to)
    # append chroma shape
    for i in res: i.append(chroma_shapes[i[1][0]][1])
    #print(res)
    #print(audio_compared_to)
    return res


def runScript(f, q):
    file1 = f[0][0]
    file2 = f[1][0]
    #print(file1, file2)
    #resfile = dtwstart(os.path.join(RECSDIR, file1), os.path.join(RECSDIR, file2))
    # f[2] = chroma shape of f[1]
    resfile = dtwstart(f[0], f[1], f[2], DATE)
    q.put(resfile)
    #return resfile


def start():
    filenames = loadRecordings()
    # compare to longest first

    matched_files = []
    for i in range(1,len(filenames)):
        audiopairs = []
        print('Run {0} of {1}'.format(i, len(filenames)-1))
        for n in range(0, len(filenames)-i):
            apairs = list(itertools.product(filenames[n], filenames[-i]))
            audiopairs += apairs
            audiopairs = list(filter(lambda x: x[0][0] not in matched_files, audiopairs))
        #for p in audiopairs: print(p)
        if len(audiopairs) > 0: matched_files += process(audiopairs, filenames[-i])
        else: break
        
        #break

def unlinkShm(fs, ftype):
    print('cleaning memory ({0})'.format(ftype))
    for f in fs:
        try:                # chroma might not exist for each
            shmname = '{0}_{1}_{2}'.format(etreeNumber(f[0]), f[1], ftype)
            #print(shmname)
            shm = shared_memory.SharedMemory(name=shmname)
            shm.close()
            shm.unlink()
        except: pass



def getChromas(fs):
    print('getting chromas')
    pool = mp.Pool(THREADS_CHROMA)
    #p = pool.map(getChroma, fs, chunksize=1)
    p = list(tqdm(pool.imap(getChroma, fs), total=len(fs)))
    pool.close()
    pool.join()
    p.sort()
    chroma_shapes = {}
    for c in p:
        gl.shms.append(shared_memory.SharedMemory(create=True, size=c[2].nbytes, name='{0}_{1}_chroma'.format(etreeNumber(c[0]), c[1])))
        s = np.ndarray(c[2].shape, dtype=np.float32, buffer=gl.shms[-1].buf)
        s[:] = c[2][:]   
        chroma_shapes[c[0]] = (c[1], c[2].shape)
    return chroma_shapes


def getChroma(f):
    shmname = '{0}_{1}_audio'.format(etreeNumber(f[0]), f[1])
    shm = shared_memory.SharedMemory(name=shmname)
    a = np.ndarray(f[2], dtype=np.float32, buffer=shm.buf)
    c = chroma_cens(y=a, sr=SR, hop_length=DTWFRAMESIZE, win_len_smooth=21)
    #print(f[0])
    return (f[0], f[1], c)  # return filename, index, chroma


def process(apairs, filenames2):
    #manager = mp.Manager()
    print('measuring pairwise similarity')
    
    #pool = mp.Pool(THREADS_SIMILARITY)
    #p = list(tqdm(pool.imap(similarity, apairs), total=len(apairs)))
    #pool.close()
    #pool.join()

    '''
    qout_s = mp.Queue()
    tasks = [mp.Process(target=similarity, args=(a, qout_s)) for a in apairs]
    tp = TaskProcessor(n_cores=THREADS_SIMILARITY, min_mem=MIN_MEM, max_mem=MAX_MEM, tasks=tasks)
    tp.start()
    tp.join()
    p = getQueue(qout_s)
    qout_s.close()
    unlinkShm(filenames2, 'hpcp')
    '''
    
    
    p = pickle.load(open('similaritymin_test.pickle', 'rb'))
    #pickle.dump(p, open('similaritymin_test.pickle', 'wb'))
    
    res = processResult(p)
    print('calculating subsequence DTW paths')

    #pool = mp.Pool(THREADS_DTW)
    #p = list(tqdm(pool.imap(runScript, res), total=len(res)))
    #pool.close()
    #pool.join()

    
    qout_r = mp.Queue()
    tasks = [mp.Process(target=runScript, args=(r, qout_r)) for r in res]
    #tp = TaskProcessor(n_cores=THREADS_DTW, max_gb=MAX_MEM_GB, tasks=tasks)
    tp = TaskProcessor(n_cores=THREADS_DTW, min_mem=MIN_MEM, max_mem=MAX_MEM, tasks=tasks)
    tp.start()
    tp.join()
    p = getQueue(qout_r)
    qout_r.close()
    unlinkShm(filenames2, 'audio')
    unlinkShm(filenames2, 'chroma')
    
    matched_files = list(set(filter(lambda x: x != None, p)))
    #print(matched_files)
    return matched_files
    

def getQueue(q):
    res = []
    q.put(-1)
    for i in iter(q.get, -1):
        res.append(i)
    return res


def etreeNumber(e):
    for j in e.split('/')[-2].split(('.')):
        try: return int(j)
        except: pass


def remove_empty_folders():
    folders = list(os.walk(DSTDIR))[1:]
    for folder in folders:
        if not folder[2]: os.rmdir(folder[0])


if __name__ == '__main__':
    #os.system('ulimit -n 30000')
    start()
    for s in gl.shms:   # there shouldn't be any open ones, but just in case
        try:
            s.close()
            s.unlink()
        except: pass
    remove_empty_folders()
    #os.system('stty sane')

