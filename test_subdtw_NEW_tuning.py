#!/opt/local/bin/python
# test sub dtw post AES

import librosa, os, json, sys, csv, samplerate
import numpy as np
from subprocess import Popen, DEVNULL, PIPE
from uuid import uuid4
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import stats
from math import ceil


SR = 22050
DTWFRAMESIZE = 512

DIR = '/Volumes/Journal/Documents/OneDrive/OneDrive - Queen Mary, University of London/projects/SDTW/'
TEMP = DIR + 'temp/'
JSONFILE = DIR + 'test.json'

TEMP = DIR + 'temp/'
JSONFILE = DIR + 'test.json'
PNGFILE = DIR + 'test.pdf'

DST = DIR + '2020/results/'



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


def libDtw(X, Y, tuning=float(0)):
    #sigma = np.array([[1, 1], [1, 2], [2, 1]])
    D, wp = librosa.sequence.dtw(X, Y, subseq=True)
    wp = processPath(wp, tuning)

    #with open("warping_path.txt", "w") as text_file:
    #    for w in wp: text_file.write(str(w) + '\n')
    return wp


def processPath(wp, tuning):
    wp = wp / SR * DTWFRAMESIZE
    if tuning != float(0): wp = scaleDtw(wp, tuning)
    wp = wp[wp[:,0].argsort()]
    wp = removeNonlinear(wp)
    #wp = wp.tolist()
    return wp


def removeNonlinear(wp):
    slen = int(10 * SR / DTWFRAMESIZE)
    l = len(wp)
    #wp_out = np.array([], dtype=np.float64).reshape(0,2)
    wp_plot = []
    slopes = []
    for i in range(0, len(wp), slen):
        if i+slen > l+1: break
        slope, intercept, r_value = stats.linregress(wp[i:i+slen])[:3]
        slopes.append(slope)
        if 1.05 > round(slope, 2) > 0.95:
            #wp_out = np.vstack([wp_out, wp[i:i+slen]])
            wp_plot.append(wp[i:i+slen])
    if len(wp_plot) == 1: wp_plot = []
    return wp_plot



def scaleDtw(wp, t):
    wp[:, 1] *= 2**(-t / 1200)
    return wp


def monoWav(f, tuning=float(0)):
    wav = os.path.join(TEMP, str(uuid4()) + '.wav')
    if f.endswith('.shn'):
        _f = os.path.join(TEMP, str(uuid4()) + '.wav')
        cmd = 'shorten -x "{0}" "{1}"'.format(f, _f)
        p = Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL).wait()
    else: _f = f
    cmd = 'ffmpeg -i "{0}" -ar {1} -ac 1 "{2}"'.format(_f, SR, wav)
    #print(cmd)
    p = Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL).wait()
    if f.endswith('.shn'): os.remove(_f)

    return wav

    #a, sr = sf.read(wav)
    #leng = len(a)
    #if tuning != float(0): 
    #    a = resampleAudio(a, tuning)
    #os.remove(wav)
    #return a, leng

def readAudio(wav, tuning=float(0)):
    a, sr = sf.read(wav)
    os.remove(wav)
    leng = len(a)
    if tuning != float(0): 
        a = resampleAudio(a, tuning)
    return a, leng


def resampleAudio(a, t):
    ratio = 2**(t / 1200)
    #print(t, ratio)
    a = samplerate.resample(a, ratio, 'sinc_best')
    return a


def getChroma(a, tuning=0):
    # tuning Deviation (in fractions of a CQT bin) from A440 tuning
    return librosa.feature.chroma_cens(y=a, sr=SR, hop_length=DTWFRAMESIZE, win_len_smooth=21, tuning=tuning/100)

def tuningFrequency(a, b):
    cmd = 'sonic-annotator -t tuning-difference.n3 -m "{0}" "{1}" -w csv --csv-stdout'.format(a, b)
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=DEVNULL).communicate()[0]
    #p = Popen(cmd, shell=True).communicate()[0]
    diff = int(str(p).split(',')[-1][:-3])
    #if abs(diff) > 150: diff = None
    #with open("tuning_diff.txt", "w") as text_file:
    #    text_file.write(str(diff))
    return float(diff)

#def plotFigure(w, l1, l2):
#    w = np.array(w)
#    p = plt.figure()
#    plt.plot(w[:, 0], w[:, 1], color='y')
#    plt.plot(0, 0, color='w')  # include full audio length in plot
#    plt.plot(l1/SR, l2/SR, color='w')
#    plt.tight_layout()
#    p.savefig(PNGFILE, bbox_inches='tight')

def plotFigure2(ws, l1, l2, file1, file2):
    fsplit1 = file1.split('/')
    fname1 = '/'.join(fsplit1[-2:])
    fsplit2 = file2.split('/')
    fname2 = '/'.join(fsplit2[-2:])
    pdfname = os.path.join(gl.dstdir, '{0}_{1}.pdf'.format(fsplit1[-1], fsplit2[-1]))
    #pdfname = os.path.join(gl.dstdir, 'test.pdf')
    #print(pdfname)
    #fname1 = '/'.join(file1.split('/')[-2:])
    #fname2 = '/'.join(file2.split('/')[-2:])
    p = plt.figure()
    plt.title('{0}\n{1}'.format(fname1, fname2))
    for w in ws:
        plt.plot(w[:, 0], w[:, 1], color='y')
    plt.plot(0, 0, color='w')  # include full audio length in plot
    plt.plot(l1/SR, l2/SR, color='w')
    plt.tight_layout()
    #print(pdfname)
    p.savefig(pdfname, bbox_inches='tight')
    plt.close(p)

def makeFolders(f1, f2):
    fid = [None, None]
    for i, f in enumerate([f1, f2]):
        for j in f.split('/')[-2].split(('.')):
            try:
                fid[i] = int(j)
                break
            except: pass
    gl.dstdir = os.path.join(DST, '{0}_{1}'.format(fid[0], fid[1]))
    for d in [TEMP, DST, gl.dstdir]:
        if not os.path.exists(d):
            try: os.makedirs(d)
            except: pass


def dtwstart(FILE1, FILE2):
    resfiles = None
    makeFolders(FILE1, FILE2)
    #return

    
    file1 = monoWav(FILE1)
    file2 = monoWav(FILE2)
    tuning = tuningFrequency(file1, file2)
    file1_np, len1 = readAudio(file1)
    file2_np, len2 = readAudio(file2, tuning)


    #print(Y.shape)
    X = getChroma(file1_np)
    Y = getChroma(file2_np)
    wp_plot = libDtw(X, Y, tuning)
    #print(wp_plot)

    if len(wp_plot) > 0:
        plotFigure2(wp_plot, len1, len2, FILE1, FILE2)
        resfiles = (FILE1, FILE2)
        dtw = wp_plot[0]
        if len(wp_plot) > 1:
            for d in wp_plot[1:]: 
                dtw = np.concatenate((dtw, d), axis=0)
        #dtw = dtw.tolist()

    
    #j = { 'dtw': dtw }
    #json.dump(j, open(JSONFILE, 'w', encoding='utf-8'), sort_keys=True)
    
    #spamwriter = csv.writer(sys.stdout)
    #for i in dtw:
    #    spamwriter.writerow(i)
    return resfiles
#main()

