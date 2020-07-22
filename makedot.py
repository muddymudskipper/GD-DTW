import os, sys
from graphviz import Digraph
from make_folder_dict import dateDict

#DATE = sys.argv[1]
#SRC = os.path.join('results', DATE)


def etreeNumber(e):
    for j in e.split('.'):
        try: return int(j)
        except: pass


def dotPairs(date):
    src = os.path.join('results', date)
    folders = [d for d in os.listdir(src) if not d.startswith('.')]
    pairs = []
    for d in folders:
        ids = d.split('_')
        json_files = [f for f in os.listdir(os.path.join(src, d)) if f.endswith('_full.json')]
        
        for j in json_files:
            p = j.split('__') 
            p[1] = p[1].replace('_full.json', '')
        
            pair = (ids[0]+'_'+p[0], ids[1]+'_'+p[1])
            pairs.append(pair)

    return pairs


def makeDot(pairs, unmatched, date):
    dot = Digraph(comment=date)
    added = []
    for p in pairs:
        for i in p:
            if i not in added:
                added.append(i)
                dot.node(i, i)
        dot.edge(p[0], p[1], constraint='false')
    
    for u in unmatched:
        if u not in added:
            added.append(u)
            dot.node(u, u)

    return dot


def makeDotStart(date):
    folders = dateDict()[date]
    
    all_files = []
    for d in folders:
        all_files += [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('flac', 'mp3', 'shn'))]
    
    all_files = list(set(all_files))

    for i, f in enumerate(all_files):
        s = f.split('/')[-2:]
        all_files[i] = (str(etreeNumber(s[0]))+'_'+s[1])

    pairs = dotPairs(date)

    for p in pairs:
        for i in p:
            if i in all_files:
                #print(i)
                all_files.remove(i)

    print(f'{date}.dot')
    dot = makeDot(pairs, all_files, date)
    dot.render(os.path.join('results', f'{date}.dot'), view=False)  

#makeDotStart(DATE)
