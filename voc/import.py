#!/usr/bin/env python3
import os
from VOC import *
from tqdm import tqdm
import picpac

def import_db (Cls, Set):
    X, Y = load_list(Cls, Set, difficult=False)
    db = picpac.Writer('db/%s.%s' % (Cls, Set), picpac.OVERWRITE)
    C = 0
    print("%s %s loading %d images..." % (Cls, Set, len(Y)))
    for path, label in tqdm(list(zip(X, Y))):
        with open(path, 'rb') as f:
            buf = f.read()
            db.append(float(label), buf)
            pass
        pass
    pass

try:
    os.mkdir('db')
except:
    pass

for Cls in CLASSES:
    for Set in ['train', 'val']:
        import_db(Cls, Set)
        pass
    pass

