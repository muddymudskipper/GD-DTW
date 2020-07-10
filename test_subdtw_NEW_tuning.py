#!/opt/local/bin/python
# test sub dtw post AES

import librosa, os, json, sys, csv, samplerate
import numpy as np
from subprocess import Popen, DEVNULL, PIPE
from uuid import uuid4
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import stats
#from math import ceil
from multiprocessing import shared_memory
import vamp

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
        if i+slen > l+1: break   # TODO: if i+0.5*slen > l  # last piece at least half of slen segment (end in x[start:end] can be > len(x))
        slope, intercept, r_value = stats.linregress(wp[i:i+slen])[:3]
        slopes.append(slope)
        if 1.04 > round(slope, 2) > 0.96:
            #wp_out = np.vstack([wp_out, wp[i:i+slen]])
            wp_plot.append(wp[i:i+slen])
    if len(wp_plot) == 1: wp_plot = []
    return wp_plot



def scaleDtw(wp, t):
    #wp[:, 1] *= 2**(-t / 1200)
    wp[:, 0] *= 2**(t / 1200)
    return wp


#def monoWav(f, tuning=float(0)):
#    wav = os.path.join(TEMP, str(uuid4()) + '.wav')
#    if f.endswith('.shn'):
#        _f = os.path.join(TEMP, str(uuid4()) + '.wav')
#        cmd = 'shorten -x "{0}" "{1}"'.format(f, _f)
#        p = Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL).wait()
#    else: _f = f
#    cmd = 'ffmpeg -i "{0}" -ar {1} -ac 1 "{2}"'.format(_f, SR, wav)
#    #print(cmd)
#    p = Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL).wait()
#    if f.endswith('.shn'): os.remove(_f)
#
#    return wav

    #a, sr = sf.read(wav)
    #leng = len(a)
    #if tuning != float(0): 
    #    a = resampleAudio(a, tuning)
    #os.remove(wav)
    #return a, leng

#def readAudio(wav, tuning=float(0)):
#    a, sr = sf.read(wav)
#    os.remove(wav)
#    leng = len(a)
#    if tuning != float(0): 
#        a = resampleAudio(a, tuning)
#    return a, leng

def resampleAudio2(a, tuning=float(0)):
    if tuning != float(0): 
        ratio = 2**(-tuning / 1200) # -tuning because resampling of audio 1
        ar = samplerate.resample(a, ratio, 'sinc_best') 
        return ar
    else:
        return a

#def resampleAudio(a, t):
#    ratio = 2**(t / 1200)
#    #print(t, ratio)
#    a = samplerate.resample(a, ratio, 'sinc_best')
#    return a


def getChroma(a):
    # tuning Deviation (in fractions of a CQT bin) from A440 tuning
    return librosa.feature.chroma_cens(y=a, sr=SR, hop_length=DTWFRAMESIZE, win_len_smooth=21)

def tuningFrequency(a, b):
    cmd = 'sonic-annotator -t tuning-difference.n3 -m "{0}" "{1}" -w csv --csv-stdout'.format(a, b)
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=DEVNULL).communicate()[0]
    #p = Popen(cmd, shell=True).communicate()[0]
    diff = int(str(p).split(',')[-1][:-3])
    #if abs(diff) > 150: diff = None
    #with open("tuning_diff.txt", "w") as text_file:
    #    text_file.write(str(diff))
    return float(diff)


def tuningFrequency2(a, b):
    two_channels = makeTwoChannels(a,b) 
    diff = vamp.collect(two_channels, SR, "tuning-difference:tuning-difference", output="cents", parameters={'maxduration': 300})
    diff = diff['list'][0]['values'][0]
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





def plotFigure2(ws, l1, l2, file1, file2):
    fsplit1 = file1.split('/')
    fname1 = '/'.join(fsplit1[-2:])
    fsplit2 = file2.split('/')
    fname2 = '/'.join(fsplit2[-2:])
    pdfname = os.path.join(gl.dstdir, '{0}_{1}.pdf'.format(fsplit1[-1], fsplit2[-1]))

    jsonname = os.path.join(gl.dstdir, '{0}_{1}.json'.format(fsplit1[-1], fsplit2[-1]))
    dtw = ws[0]
    if len(ws) > 1:
        for d in ws[1:]: 
            dtw = np.concatenate((dtw, d), axis=0)
    dtw = dtw.tolist()
    j = { 'dtw': dtw, 'filenames': [fname1, fname2], 'lengths': [l1/SR, l2/SR] }
    json.dump(j, open(jsonname, 'w', encoding='utf-8'), sort_keys=True)

    
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

def makeFolders(e1, e2):
    #fid = [None, None]
    #for i, f in enumerate([f1, f2]):
        #for j in f.split('/')[-2].split(('.')):
        #    try:
        #        fid[i] = int(j)
        #        break
        #    except: pass
    #    fid[i] = etreeNumber(f)

    gl.dstdir = os.path.join(DST, '{0}_{1}'.format(e1, e2))
    for d in [TEMP, DST, gl.dstdir]:
        if not os.path.exists(d):
            try: os.makedirs(d)
            except: pass


def etreeNumber(e):
    for j in e.split('/')[-2].split(('.')):
        try: return int(j)
        except: pass


def dtwstart(FILE1, FILE2, CHROMASHAPE2, RECSDIR):
    etree_number1 = etreeNumber(FILE1[0])
    etree_number2 = etreeNumber(FILE2[0])

    resfile = None
    makeFolders(etree_number1, etree_number2)
    #return

    shmname1 = '{0}_{1}_audio'.format(etree_number1, FILE1[1])
    shm1 = shared_memory.SharedMemory(name=shmname1)
    file1_buf = np.ndarray(FILE1[2], dtype=np.float32, buffer=shm1.buf)
    shmname2 = '{0}_{1}_audio'.format(etree_number2, FILE2[1])
    shm2 = shared_memory.SharedMemory(name=shmname2)
    file2_buf = np.ndarray(FILE2[2], dtype=np.float32, buffer=shm2.buf)
    

    tuning = tuningFrequency2(file1_buf, file2_buf)

    file1_resampled = resampleAudio2(file1_buf, tuning)
    X = getChroma(file1_resampled)
    #Y = getChroma(file2_buf)

    shmnameY = '{0}_{1}_chroma'.format(etree_number2, FILE2[1])
    shmY = shared_memory.SharedMemory(name=shmnameY)
    Y = np.ndarray(CHROMASHAPE2, dtype=np.float32, buffer=shmY.buf)

    filename1 = os.path.join(RECSDIR, FILE1[0])
    filename2 = os.path.join(RECSDIR, FILE2[0])
    '''
    file1 = monoWav(filename1)
    file2 = monoWav(filename2)
    tuning = tuningFrequency(file1, file2)
    file1_np, len1 = readAudio(file1)
    file2_np, len2 = readAudio(file2, tuning)
    #print(Y.shape)
    X = getChroma(file1_np)
    Y = getChroma(file2_np)
    '''

    wp_plot = libDtw(X, Y, tuning)
    #print(wp_plot)

    if len(wp_plot) > 0:
        #plotFigure2(wp_plot, len1, len2, filename1, filename2)
        plotFigure2(wp_plot, FILE1[2][0], FILE2[2][0], filename1, filename2)
        resfile = '/'.join(filename1.split('/')[-2:])
        dtw = wp_plot[0]
        #if len(wp_plot) > 1:
        #    for d in wp_plot[1:]: 
        #        dtw = np.concatenate((dtw, d), axis=0)
        #dtw = dtw.tolist()

    
    #j = { 'dtw': dtw }
    #json.dump(j, open(JSONFILE, 'w', encoding='utf-8'), sort_keys=True)
    
    #spamwriter = csv.writer(sys.stdout)
    #for i in dtw:
    #    spamwriter.writerow(i)
    return resfile
#main()

