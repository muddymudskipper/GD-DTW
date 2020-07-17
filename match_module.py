#!/opt/local/bin/python
# test sub dtw post AES

import librosa, os, json, sys, warnings
from samplerate import resample
import numpy as np
from subprocess import Popen, DEVNULL, PIPE
from uuid import uuid4
import matplotlib.pyplot as plt
from scipy import stats
from math import ceil
from multiprocessing import shared_memory
import vamp


np.seterr(divide='ignore', invalid='ignore')   # stats division by zero warning


SR = 22050
DTWFRAMESIZE = 512
SEGMENT_LENGTH = 10  # length for wp segments in linear regression 
LINREGRESS_MIN = 0.94
LINREGRESS_MAX = 1.04

MATCH_INCREMENT = 0.02

#DIR = '/Volumes/Journal/Documents/OneDrive/OneDrive - Queen Mary, University of London/projects/SDTW/'
#TEMP = DIR + 'temp/'
#DST = DIR + '2020/results/'
TEMP = 'temp'
DST = 'results'


#FILE1 = sys.argv[1]
#FILE2 = sys.argv[2]
#FILE1 = '/Volumes/Beratight2/SDTW/82-10-10/gd1982-10-10.nak700.anon-poris.LMPP.95682.flac16/gd82-10-10d3t06.flac'
#FILE2 = '/Volumes/Beratight2/SDTW/82-10-10/gd1982-10-10.nak700.wagner.miller.109822.flac16/gd82-10-10d3t06.flac'

#FILE1 = '/Volumes/Beratight2/SDTW/test/gd1971-08-06d2t04_part.flac'
#FILE2 = '/Volumes/Beratight2/SDTW/test/gd1971-08-06d2t04_part.flac'
#FILE1 = None
#FILE2 = None


    
class gl():
    dstdir = None


def libDtw(X, Y, tuning_diff=0):
    #sigma = np.array([[1, 1], [1, 2], [2, 1]])
    D, wp = librosa.sequence.dtw(X, Y, subseq=True, global_constraints=True)
    wp = processPath(wp, tuning_diff)
    #with open("warping_path.txt", "w") as text_file:
    #    for w in wp: text_file.write(str(w) + '\n')
    return wp


def match(X, Y, tuning=440, tuning_diff=0):
    tuning = float(tuning)
    #print(tuning)
    chns, swap = makeTwoChannels(X, Y)
    p = vamp.collect(chns, SR, 'match-vamp-plugin:match-subsequence', output="path", parameters={'freq1': tuning, 'freq2': tuning, 'zonewidth': 60})
    wp = []
    for i in p['list']:
        if swap:
            wp.append([float(i['values'][0]), float(i['timestamp'])])
        else:
            wp.append([float(i['timestamp']), float(i['values'][0])])
    wp = processPath(np.array(wp), tuning_diff)
    #print(wp)
    return wp

    
    
def makeTwoChannels(a, b):
    if len(a) > len(b):
        pad = np.pad(b, [0,len(a)-len(b)])
        return np.array([a, pad]), False
    elif len(b) > len(a):
        pad = np.pad(a, [0,len(b)-len(a)])
        return np.array([b, pad]), True
    else:
        return np.array([a, b]), False
    

def processPath(wp, tuning_diff):
    #wp = wp / SR * DTWFRAMESIZE            # not needed for match
    if tuning_diff != float(0): wp = scaleDtw(wp, tuning_diff)
    wp = wp[wp[:,0].argsort()]
    wp = removeNonlinear(wp)
    #wp = wp.tolist()
    return wp


def removeNonlinear(wp):
    #slen = SEGMENT_LENGTH * SR / DTWFRAMESIZE
    slen = SEGMENT_LENGTH / MATCH_INCREMENT

    number_of_chunks = len(wp) / slen
    chunks = np.array_split(wp, number_of_chunks)  # split to chunks of roughly same length
    wp_plot = []
    for chunk in chunks:
        slope, intercept, r_value = stats.linregress(chunk)[:3]
        if LINREGRESS_MAX > round(slope, 2) > LINREGRESS_MIN:
            wp_plot.append(chunk)
    if len(wp_plot) == 1: wp_plot = []  # testing: omit if only one chunk is aligned to avoid false positives
    return wp_plot


def _removeNonlinear(wp):   # testing
    frames_per_second = int(SR / DTWFRAMESIZE)
    len_ceil_seconds = ceil(len(wp) / frames_per_second)
    wp_plot = []
    # next chunks with 1s shifting window of 10s:
    for n in range(0, len_ceil_seconds):
         chunk = wp[n*frames_per_second:(n+10)*frames_per_second]
         slope, intercept, r_value = stats.linregress(chunk)[:3]
         if 1.04 > round(slope, 2) > 0.96:
             if len(wp_plot) == 0: wp_plot.append(chunk)
             else: wp_plot.append(chunk[-frames_per_second:])
    if len(wp_plot) == 1: wp_plot = []  # testing: omit if only one chunk is aligned to avoid false positives
    return wp_plot


def scaleDtw(wp, tuning_diff):
    wp[:, 0] *= 2**(tuning_diff / 1200)
    return wp


def resampleAudio(a, tuning_diff=0):
    if tuning_diff != 0: 
        ratio = 2**(-tuning_diff / 1200) # -tuning because resampling of audio 1
        #ar = resample(a, ratio, 'sinc_best')
        #ar = resample(a, ratio, 'sinc_medium')
        ar = resample(a, ratio, 'sinc_fastest')
        return ar
    else:
        return a


def plotFigure(ws, l1, l2, file1, file2, tuning_diff):
    fsplit1 = file1.split('/')
    #fname1 = '/'.join(fsplit1[-2:])
    fsplit2 = file2.split('/')
    #fname2 = '/'.join(fsplit2[-2:])
    pdfname = os.path.join(gl.dstdir, '{0}_{1}.png'.format(fsplit1[-1], fsplit2[-1]))

    jsonname = os.path.join(gl.dstdir, '{0}_{1}.json'.format(fsplit1[-1], fsplit2[-1]))
    dtw = ws[0]
    if len(ws) > 1:
        for d in ws[1:]: 
            dtw = np.concatenate((dtw, d), axis=0)
    dtw = dtw.tolist()
    j = { 'dtw': dtw, 'filenames': [file1, file2], 'lengths': [l1/SR, l2/SR], 'tuning_diff': tuning_diff }
    json.dump(j, open(jsonname, 'w', encoding='utf-8'), sort_keys=True)

    p = plt.figure()
    plt.title('{0}\n{1}'.format(file1, file2))
    for w in ws:
        plt.plot(w[:, 0], w[:, 1], color='y')
    plt.plot(0, 0, color='w')  # include full audio length in plot
    plt.plot(l1/SR, l2/SR, color='w')
    plt.tight_layout()
    p.savefig(pdfname, bbox_inches='tight')
    plt.close(p)


def makeFolders(e1, e2, date):
    datedir = os.path.join(DST, date)
    gl.dstdir = os.path.join(datedir, '{0}_{1}'.format(e1, e2))
    for d in [TEMP, DST, datedir, gl.dstdir]:
        if not os.path.exists(d):
            try: os.makedirs(d)
            except: pass


def etreeNumber(e):
    for j in e.split('/')[-2].split(('.')):
        try: return int(j)
        except: pass


def matchStart(FILE1, FILE2, TUNING, DATE, TUNING_DIFF):
    filename1 = FILE1[0]
    filename2 = FILE2[0]
    
    etree_number1 = etreeNumber(filename1)
    etree_number2 = etreeNumber(filename2)

    resfile = None
    makeFolders(etree_number1, etree_number2, DATE)

    shmname1 = '{0}_{1}_audio'.format(etree_number1, FILE1[1])
    shm1 = shared_memory.SharedMemory(name=shmname1)
    file1_buf = np.ndarray(FILE1[2], dtype=np.float32, buffer=shm1.buf)
    shmname2 = '{0}_{1}_audio'.format(etree_number2, FILE2[1])
    shm2 = shared_memory.SharedMemory(name=shmname2)
    file2_buf = np.ndarray(FILE2[2], dtype=np.float32, buffer=shm2.buf)
    
    #tuning_diff = tuningDiff(file1_buf, file2_buf)

    file1_resampled = resampleAudio(file1_buf, TUNING_DIFF)
    #X = getChroma(file1_resampled, tuning_frac2)

    #shmnameY = '{0}_{1}_chroma'.format(etree_number2, FILE2[1])
    #shmY = shared_memory.SharedMemory(name=shmnameY)
    #Y = np.ndarray(chroma_shape2, dtype=np.float32, buffer=shmY.buf)

    wp_plot = match(file1_resampled, file2_buf, TUNING, TUNING_DIFF)

    if len(wp_plot) > 0:
        plotFigure(wp_plot, FILE1[2][0], FILE2[2][0], filename1, filename2, TUNING_DIFF)
        resfile = filename1
        dtw = wp_plot[0]

    return resfile
