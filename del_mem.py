
import pickle, os, sys
from multiprocessing import shared_memory

DATE = sys.argv[1]

folders = pickle.load(open('date_folder_dict.pickle', 'rb'))[DATE]

def etreeNumber(e):
    for j in e.split('/')[-2].split(('.')):
        try: return int(j)
        except: pass


def deleteShms():
    shmnames = []
    for d in folders:
        files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('flac', 'mp3', 'shn'))]
        for i, f in enumerate(files):
            etree_number = etreeNumber(f)
            shmname = '{0}_{1}'.format(etree_number, i)
            shmnames.append(shmname)
            #print(shmname)
            for t in ['audio', 'hpcp', 'chroma']:
                try:
                    shm = shared_memory.SharedMemory(name=shmname + '_' + t)
                    shm.close()
                    shm.unlock()
                except: pass
    pickle.dump(shmnames, open('shmnames.pickle', 'wb'))

def deleteShms2():
    shmnames = pickle.load(open('shmnames.pickle', 'rb'))
    for shmname in shmnames:
        for t in ['audio', 'hpcp', 'chroma']:
            try:
                shm = shared_memory.SharedMemory(name=shmname + '_' + t)
                shm.close()
                shm.unlock()
            except: pass


#deleteShms()
deleteShms2()



