#!/usr/bin/env python
"""
Input:  Two Forbes document IDs.
Output:
  - list of common features
  - each doc, with common features highlighted
  (does NOT show most important DISTINCT features)

Assumes:
  - annotated doc models dir: '../Docs/doc-forbes.annotated'
  - each doc file is named: '<doc ID>.model.json'
"""
from util import mk_ascii
from highlight import highlight
from overlap import common_features

import json, sys

def get_filename(doc_id):
    return '../Docs/doc-forbes.annotated/{0}.model.json'.format(doc_id)

def get_doc(doc_id):
    fn = get_filename(doc_id)
    return json.load(file(fn, 'r'))

#--------------------

# Look for matching:
#   - feature string w/ lemma label
#   - lemma begin w/ pos begin
#   - pos label[0] w/ feature pos ch
def hi_sentence(sent, features):
    locs = []
    ans = sent['annotation']
    lemmas = ans['lemma']
    posses = ans['pos']
    for ft in features:
        for lm in lemmas:
            ft_str = ft[:-2]
            ft_pos = ft[-1:]
            if lm['label'] == ft_str:
                def is_match(pos):
                    return (
                        (pos['begin'] == lm['begin']) and
                        (pos['label'][0] == ft_pos)
                        )
                # at most one, of course...
                for p in filter(is_match, posses):
                    loc = int(p['begin']), int(p['length'])
                    locs.append(loc)
    return highlight(sent['display'], locs, 'green')

def highlighted_doc_str(doc, features):
    strs = [hi_sentence(s, features) for s in doc['sentences']]
    return mk_ascii( '  '.join(strs) )

#--------------------

ids = map(int, sys.argv[1:3])
fs = list(common_features(ids))
print 'common features:', fs
print '-' * 70

a_id, b_id = ids
doc = get_doc(a_id)
print highlighted_doc_str(doc, fs)

print '-' * 70

doc = get_doc(b_id)
print highlighted_doc_str(doc, fs)
