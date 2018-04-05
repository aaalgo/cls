#!/usr/bin/env python3
import picpac

def import_db (Set):
    db = picpac.Writer('%s.db' % Set, picpac.OVERWRITE)
    with open('%s.list' % Set, 'r') as f:
        for l in f:
            path, label = l.strip().split('\t')
            with open(path, 'rb') as f2:
                buf = f2.read()
            db.append(float(label), buf)
            pass
        pass
    del db
    pass

import_db('train')
import_db('val')
