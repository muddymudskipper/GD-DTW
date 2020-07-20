# https://essentia.upf.edu/essentia_python_examples.html

# 1. find longest performance
# 2. try alignemnet of all others with performance from 1.
# 3. store files with no alignment
# 4. repeat from 1. with unaligned files and reference next longest recording


import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram
import itertools, os, json, sys
import multiprocessing as mp
from multiprocessing import shared_memory
from subprocess import Popen, DEVNULL, PIPE
import numpy as np
from uuid import uuid4
from librosa.feature import chroma_cens
from tqdm import tqdm
from subprocess import Popen, DEVNULL, PIPE
import pickle
from multiprocessing.managers import SharedMemoryManager
from math import ceil, floor, log2
from random import shuffle
import vamp

#from test_subdtw_NEW_tuning import dtwstart
from match_module import matchStart
from make_folder_dict import dateDict

#RECSDIR = '/Volumes/Beratight2/SDTW/82-07-29'
#DIR = '/Volumes/Journal/Documents/OneDrive/OneDrive - Queen Mary, University of London/projects/SDTW/'
#TEMPDIR = DIR + 'temp/'

TOTAL_MEM = 64
HEADROOM_MEM = 10
FREE_MEM = TOTAL_MEM - HEADROOM_MEM
B_TO_GB = 1 / 2**30

DATE = sys.argv[1]
#DATE = '82-07-29'
TEMPDIR = 'temp'
DSTDIR = os.path.join('results', DATE)

CPUS = 24
THREADS_LOADING = 24
#THREADS_CHROMA = 24
THREADS_SIMILARITY = 12
#THREADS_DTW = 10
THREADS_TUNING = 24
THREADS_TUNINGDIFF = 24
THREADS_MATCH = 24

SR = 22050
DTWFRAMESIZE = 512

NUM_SIMILAR = 1


class gl():
    shms = []


def loadRecordings():
    print('loading audio files')

    #folders = pickle.load(open('date_folder_dict.pickle', 'rb'))[DATE]
    #folders = [os.path.join(RECSDIR, d) for d in os.listdir(RECSDIR) if os.path.isdir(os.path.join(RECSDIR, d))]

    #recordings = pickle.load(open('recordings.pickle', 'rb'))
    folders = dateDict()[DATE]

    recordings = []
    for d in folders:
        print('loading files for', d.split('/')[-1])
        files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('flac', 'mp3', 'shn'))]
        pool = mp.Pool(nThreads(files, THREADS_LOADING))
        p = list(tqdm(pool.imap(loadFiles, files), total=len(files)))
        pool.close()
        pool.join()
        p = list(filter(lambda x: x != None, p)) # remove None type for unloadable files
        p.sort()
        recordings.append(p)
    recordings = sorted(recordings, key=lambda x: combinedLength(x))
    # shared memory for each audio file

    #pickle.dump(recordings, open('recordings.pickle', 'wb'))
    

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


def similarity(audiopair):
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
    
    return([audiopair[0], audiopair[1], distance])


def processResult(p):
    pdict = {}
    res = []
    for i in p:
        if i[0] not in pdict: pdict[i[0]] = []
        pdict[i[0]].append([i[2], i[1]])
    for k, v in pdict.items():
        #smin = sorted(v)[:NUM_SIMILAR]        # store filenames with min distances
        sort_sim = sorted(v)
        smin = [sort_sim[0]]
        if sort_sim[1] <= 2 * sort_sim[0]:
            smin.append(sort_sim[1])
        #smin = sorted(v)[:1]        # for testing
        for i in smin: res.append([k] + [i[1]]) # store recording pairs with all info

    
    ## calculate chromas for all audio files that are second input for dtw
    # match: calculate tuning and tuning difference for all audio files that files are matched against
    audio_compared_to = []
    [audio_compared_to.append(f[1]) for f in res]
    audio_compared_to = list(set(audio_compared_to))
    tunings = getTunings(audio_compared_to)
    # append tuning freq
    for i in res: i.append(tunings[i[1][0]][1])
    print('getting tuning differences')
    pool = mp.Pool(nThreads(res, THREADS_TUNINGDIFF))
    res = list(tqdm(pool.imap(tuningDiffStart, res), total=len(res)))
    pool.close()
    pool.join()

    '''
    chroma_shapes = getChromas(audio_compared_to)
    # append chroma shape
    for i in res: i.append(chroma_shapes[i[1][0]][1])
    #print(res)
    #print(audio_compared_to)

    res = estimate_memory(res)
    res = groupPairsBySize(res)
    '''
    #print(res[0])
    return res


def getTunings(fs):
    print('getting tuning frequencies')
    pool = mp.Pool(nThreads(fs, THREADS_TUNING))
    p = list(tqdm(pool.imap(getTuning, fs), total=len(fs)))
    pool.close()
    pool.join()
    p.sort()
    tunings = {}
    for c in p:
        tunings[c[0]] = (c[1], c[2])   # index, tuning
    return tunings


def getTuning(f):
    shmname = '{0}_{1}_audio'.format(etreeNumber(f[0]), f[1])
    shm = shared_memory.SharedMemory(name=shmname)
    a = np.ndarray(f[2], dtype=np.float32, buffer=shm.buf)
    freq = vamp.collect(a, SR, 'nnls-chroma:tuning', output="tuning")
    freq = freq['list'][0]['values'][0]
    return (f[0], f[1], freq)  # return filename, index, tuning


def groupPairsBySize(res):
    # group processes by how many parallel processes can be run
    shm_mem = sum(s.size for s in gl.shms) * B_TO_GB
    #print(shm_mem)
    avail_mem = FREE_MEM - shm_mem

    # sort audiopairs by memory use
    res = sorted(res, key=lambda x: x[4], reverse=True)
    total = len(res)
    resdict = {}
    for r in res:
        #cpus = int(avail_mem / r[4])
        #if cpus == 0: cpus = 1
        cpus = avail_mem / r[4]
        if cpus < 1: 
            cpus = 1
        else:
            cpus = normal_round(cpus + (0.2 * cpus))
            if cpus > CPUS: cpus = CPUS
        if cpus not in resdict: 
            resdict[cpus] = []
        resdict[cpus].append(r)

    reslist = sorted([(k, v) for k, v in resdict.items()])  # [0] = cpus, [1] = audiopairs
    #json.dump(resdict, open('cpus.json', 'w', encoding='utf-8'), sort_keys=True)
    for i, r in enumerate(reslist[:-1]):
        while len(r[1]) < r[0]:
            moved = False
            for s in reslist[i+1:]:
                for a in s[1]:
                    moved = True
                    s[1].remove(a)
                    r[1].append(a) 
            if not moved: break

    reslist = list(filter(lambda x: len(x[1]) > 0, reslist))
    [shuffle(i[1]) for i in reslist[1:]]

    #json.dump(reslist, open('cpus3.json', 'w', encoding='utf-8'), sort_keys=True)
    return reslist


def normal_round(n):
    if n - floor(n) < 0.5:
        return floor(n)
    return ceil(n)


def runScript(f):
    #file1 = f[0][0]
    #file2 = f[1][0]
    #print(file1, file2)
    # f[2] = chroma shape of f[1]
    #resfile = dtwstart(f[0], f[1], f[2], DATE, f[3])
    resfile = matchStart(f[0], f[1], f[2], DATE, f[3])
    # file1, file2, tuning, date, tuning diff
   
    return resfile


def start():
    filenames = loadRecordings()
    # compare to longest first

    matched_files = []
    for i in range(1,len(filenames)):
        print('Run {0} of {1}'.format(i, len(filenames)-1))
        audiopairs = []
        for n in range(0, len(filenames)-i):
            apairs = list(itertools.product(filenames[n], filenames[-i]))
            audiopairs += apairs
            audiopairs = list(filter(lambda x: x[0][0] not in matched_files, audiopairs))
        #for p in audiopairs: print(p)
        if len(audiopairs) > 0: matched_files += process(audiopairs, filenames[-i], i)
        else: 
            print('finished')
            break
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
    pool = mp.Pool(nThreads(fs, THREADS_CHROMA))
    #p = pool.map(getChroma, fs, chunksize=1)
    p = list(tqdm(pool.imap(getChroma, fs), total=len(fs)))
    pool.close()
    pool.join()
    p.sort()
    chroma_shapes = {}
    for c in p:
        gl.shms.append(shared_memory.SharedMemory(create=True, size=c[2][0].nbytes, name='{0}_{1}_chroma'.format(etreeNumber(c[0]), c[1])))
        s = np.ndarray(c[2][0].shape, dtype=np.float32, buffer=gl.shms[-1].buf)
        s[:] = c[2][0][:]   
        chroma_shapes[c[0]] = (c[1], (c[2][0].shape, c[2][1]))   # c[2][1] = tuning_frac
    return chroma_shapes


def getChroma(f):
    shmname = '{0}_{1}_audio'.format(etreeNumber(f[0]), f[1])
    shm = shared_memory.SharedMemory(name=shmname)
    a = np.ndarray(f[2], dtype=np.float32, buffer=shm.buf)
    tuning_frac = tuningFreq(a)
    c = chroma_cens(y=a, sr=SR, hop_length=DTWFRAMESIZE, win_len_smooth=21, tuning=tuning_frac)
    #print(f[0])
    return (f[0], f[1], (c, tuning_frac))  # return filename, index, (chroma, tuning_frac)


def tuningFreq(b):
    freq = vamp.collect(b, SR, 'nnls-chroma:tuning', output="tuning")
    freq = freq['list'][0]['values'][0]
    frac = 12 * log2(freq / 440)
    return frac 


def tuningDiffStart(fp):
    #fp.append(0)
    #return fp
    file1 = fp[0]
    file2 = fp[1]
    etree_number1 = etreeNumber(file1[0])
    etree_number2 = etreeNumber(file2[0])
    shmname1 = '{0}_{1}_audio'.format(etree_number1, file1[1])
    shm1 = shared_memory.SharedMemory(name=shmname1)
    file1_buf = np.ndarray(file1[2], dtype=np.float32, buffer=shm1.buf)
    shmname2 = '{0}_{1}_audio'.format(etree_number2, file2[1])
    shm2 = shared_memory.SharedMemory(name=shmname2)
    file2_buf = np.ndarray(file2[2], dtype=np.float32, buffer=shm2.buf)
    tuning_diff = tuningDiff(file1_buf, file2_buf)
    fp.append(tuning_diff)
    return fp


def tuningDiff(a, b):
    two_channels = makeTwoChannels(a,b) 
    diff = vamp.collect(two_channels, SR, "tuning-difference:tuning-difference", output="cents", parameters={'maxrange': 4})
    diff = diff['list'][0]['values'][0]
    if diff > 300: diff = 0     # if more than 3 semitones difference there might be something wrong
    #print('diff = ', float(diff))
    return float(diff) 


def makeTwoChannels(a, b):
    if len(a) > len(b):
        pad = np.pad(b, [0,len(a)-len(b)])
        return np.array([a, pad])
    elif len(b) > len(a):
        pad = np.pad(a, [0,len(b)-len(a)])
        return np.array([pad, b])
    else:
        return np.array([a, b])


def nThreads(a, t):
    return min(len(a), t)


def process(apairs, filenames2, run):
    print('measuring pairwise similarity')
    
    if run == -1:
        res = p = pickle.load(open('processed_results1.pickle', 'rb'))
        #pickle.dump(res, open('processed_results1.pickle', 'wb'))
    else:
        pool = mp.Pool(nThreads(apairs, THREADS_SIMILARITY))
        p = list(tqdm(pool.imap(similarity, apairs), total=len(apairs)))
        pool.close()
        pool.join()
        #pickle.dump(p, open('processed_results1.pickle', 'wb'))
    
    res = processResult(p)

  
       
    print('calculating match alignment')
    pool = mp.Pool(nThreads(res, THREADS_MATCH))
    p = list(tqdm(pool.imap(runScript, res), total=len(res)))
    pool.close()
    pool.join()
    
    '''
    total = sum([len(i[1]) for i in res])
    count = 0
    p = []
    for i, e in enumerate(res):
        le = len(e[1])
        print(f'{i+1}/{len(res)} ({e[0]} CPUs) [total: {count}/{total}]')
        threads = e[0]
        if threads > le: threads = le
        pool = mp.Pool(threads)
        q = list(tqdm(pool.imap(runScript, e[1]), total=le))
        pool.close()
        pool.join()
        p += q
        count += le
    '''
    unlinkShm(filenames2, 'audio')
    #unlinkShm(filenames2, 'chroma')
    
    matched_files = list(set(filter(lambda x: x != None, p)))
    #print(matched_files)
    return matched_files
    

def estimate_memory(res):
    mem_estims = []
    print('getting tuning differences')
    pool = mp.Pool(nThreads(res, THREADS_TUNING))
    p = list(tqdm(pool.imap(tuningDiffStart, res), total=len(res)))
    pool.close()
    pool.join()

    for r in p:
        file_len1 = r[0][2][0]
        ratio = 2**(-r[3] / 1200)
        file_len1 *= ratio
    
        chroma_len1 = ceil(ceil(file_len1) / DTWFRAMESIZE)
        chroma_len2 = r[2][0][1]
        msize = 3 * chroma_len1 * chroma_len2
        psize = min([chroma_len1, chroma_len2]) * 2
        res = (msize + psize) * 8 * B_TO_GB
        r.append(res)
    
    return p


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

