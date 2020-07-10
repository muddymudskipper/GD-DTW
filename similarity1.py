# https://essentia.upf.edu/essentia_python_examples.html

import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram
import itertools, os, json
import multiprocessing as mp
from subprocess import Popen, DEVNULL, PIPE
from test_subdtw_NEW_tuning import dtwstart

#THREADS = 12
SCRIPT = '../../test_subdtw-NEW_tuning.command.py'
#SCRIPT = '../../test_subdtw-NEW_tuning_segments.command.py'
JSONFILE = 'test.json'

SR = 22050
#FILE1 = '/Volumes/Beratight2/SDTW/problematic/gd70-06-24t04.flac'
#FILE2 = '/Volumes/Beratight2/SDTW/problematic/gd1970-06-24d1t09.flac'
#REC1 = '/Volumes/Beratight2/SDTW/82-10-10/gd1982-10-10.nak700.anon-poris.LMPP.95682.flac16'
#REC2 = '/Volumes/Beratight2/SDTW/82-10-10/gd1982-10-10.nak700.wagner.miller.109822.flac16'

RECSDIR = '/Volumes/Beratight2/SDTW/82-10-10'

THREADS_SIMILARITY = 24
THREADS_DTW = 5


def folderPairs():
    folders = [os.path.join(RECSDIR, d) for d in os.listdir(RECSDIR) if os.path.isdir(os.path.join(RECSDIR, d))]
    pairs = list(itertools.combinations(folders, 2))
    return pairs


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
    f1 = audiopair[0]
    f2 = audiopair[1]
    file1 = estd.MonoLoader(filename=f1, sampleRate=SR)()
    file2 = estd.MonoLoader(filename=f2, sampleRate=SR)()
    # Now let's compute Harmonic Pitch Class Profile (HPCP) chroma features of these audio signals.
    file1_hpcp = hpcpgram(file1, sampleRate=SR)
    file2_hpcp = hpcpgram(file2, sampleRate=SR)
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
    print(('/').join(file1.split('/')[-2:]), ('/').join(file2.split('/')[-2:]))
    #cmd = 'python {0} "{1}" "{2}"'.format(SCRIPT, file1, file2)
    resfiles = dtwstart(file1, file2)
    print(resfiles)
    #print(cmd)
    #p = Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL).wait()
    return resfiles

def start():
    dpairs = folderPairs()
    for d in dpairs:
        process(d)
        break


def process(recs):

    apairs = audioPairs(recs)    
    manager = mp.Manager()
    count = manager.Value('i', 0)
    pool = mp.Pool(THREADS_SIMILARITY)
    #for i, a in enumerate(apairs):
    #    similarity(a)
    p = pool.map(similarity, apairs, chunksize=1)
    pool.close()
    pool.join()

    res = processResult(p)
    pool = mp.Pool(THREADS_DTW)
    p = pool.map(runScript, res, chunksize=1)
    pool.close()
    pool.join()

    #json.dump(p, open(JSONFILE, 'w', encoding='utf-8'), sort_keys=True)



    

'''
if __name__ == '__main__':
    start()
    os.system('stty sane')
'''

