
import pickle, os, sys
from multiprocessing import shared_memory

DATE = sys.argv[1]

folders = pickle.load(open('date_folder_dict.pickle', 'rb'))[DATE]

def etreeNumber(e):
    for j in e.split('/')[-2].split(('.')):
        try: return int(j)
        except: pass


for d in folders:
    files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('flac', 'mp3', 'shn'))]
    for i, f in enumerate(files):
        etree_number = etreeNumber(f)
        shmname = '{0}_{1}'.format(etree_number, i)
        #print(shmname)
        for t in ['audio', 'hpcp', 'chroma']:
            try:
                shm = shared_memory.SharedMemory(name=shmname + '_' + t)
                shm.close()
                shm.unlock()
            except: pass



