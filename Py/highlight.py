#!/usr/bin/env python
from termcolor import colored

def highlight(sentence, locations, color='red'):
    """
    'locations' are (begin, length) tuples.
    """
    next_idx = 0
    strs = []
    for begin, length in sorted(locations):
        if next_idx < begin:
            s = sentence[next_idx:begin]
            strs.append(s)
        s = sentence[begin : (begin+length)]
        strs.append( colored(s, color, attrs=['bold', 'underline']) )
        next_idx = begin + length
    if next_idx < len(sentence):
        s = sentence[next_idx : len(sentence)]
        strs.append(s)
    return ''.join(strs)
    
def test_highlight():
    s = 'Ting, Zoe, Trevor and I all went ice skating.'
    ting  = (0, 4)
    zoe   = (6, 3)
    trev  = (11, 6)
    skate = (37, 7)
    print highlight(s, [trev, zoe, skate])

# test_highlight()
