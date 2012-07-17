#!/usr/bin/env python
"""
pymongo.Collection:
  - drop() - the whole collection
  - count()
  - create_index(key_or_list), ensure_index(), drop_index(), drop_indexes(), reindex()
  - index_information()
  - rename(new_name)

  - insert(doc) or insert(docs)
  - save(doc):
    + update() - if doc._id  _OR_
    + insert() - if not
  - update(spec, doc, upsert=False, multi=False):
    + spec: dict specifying necessary elements
    + can use update modifiers
  - remove(spec_or_obj_id)

  - find(spec).  other kwargs:
    + fields: [str].  names of fields to return.
    + skip, limit
    + sort: [(key,direction)]
  -find_one(spec_or_id)
  
"""
from pymongo import Connection

# connection
con = Connection()  # ('localhost', 27017)
# database
db = con.test_database

# (new?) collection
tfvs = db.term_freq_vecs
# tfvs.drop()

# all collections
print db.collection_names()
assert sorted(db.collection_names()) == [u'system.indexes', u'term_freq_vecs']
# assert db.collection_names() == [u'system.indexes']

#--------------------------

tfvs = db.term_freq_vecs
print 'count', tfvs.count()
for tfv in tfvs.find():
    print tfv

# document
doc = {'guid': 123,
       'features': {'dog/N': 3,
                    'chase/V': 1,
                    'cat/N': 2}
       }
tfvs.insert(doc)

# d = tfvs.find_one({'editor':'mark'})
d = tfvs.find_one()
print 'found doc:', d

# d['editor'] = 'mark'
tfvs.update({'_id': d['_id']}, {'editor':'mark'})
d = tfvs.find_one()
print 'found doc:', d
