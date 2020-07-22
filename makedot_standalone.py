import os, sys
from graphviz import Digraph
from make_folder_dict import dateDict

DATE = sys.argv[1]
SRC = os.path.join('results', DATE)

COLOURS = ['black', 'red', 'blue', 'yellow', 'darkorange', 'green', 'grey71', 'brown', 'pink1', 'cyan1', 'mediumpurple', 'wheat', 'darkolivegreen', 'burlywood', 'orangered', 'lightsteelblue', 'lightgoldenrod', 'cyan4', 'darkslategray', 'violetred3', 'yellow3']






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
        #json_files = [f for f in os.listdir(os.path.join(src, d)) if f.endswith('.json')]
        
        for j in json_files:
            p = j.split('__') 
            p[1] = p[1].replace('_full.json', '')
            #p[1] = p[1].replace('.json', '')
        
            pair = (ids[0]+'_'+p[0], ids[1]+'_'+p[1])
            pairs.append(pair)

    return pairs


def makeDot(pairs, unmatched, date, col_dict):
    dot = Digraph(comment=date)
    added = []
    for p in pairs:
        for i in p:
            if i not in added:
                added.append(i)
                col = col_dict[i.split('_')[0]]
                dot.node(i, i, color=col)
        dot.edge(p[0], p[1], constraint='false')
    
    for u in unmatched:
        if u not in added:
            added.append(u)
            col = col_dict[u.split('_')[0]]
            dot.node(u, u, color=col)

    return dot


def makeDotStart(date):
    col_dict = {}

    folders = dateDict()[date]

    for i, d in enumerate(folders):
        j = i
        if i > 19: j = i - 19
        col_dict[str(etreeNumber(d.split('/')[-1]))] = COLOURS[j]
        
    #print(col_dict)

    
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
    dot = makeDot(pairs, all_files, date, col_dict)
    dot.save(os.path.join('results', date, f'{date}.dot'))      

makeDotStart(DATE)
