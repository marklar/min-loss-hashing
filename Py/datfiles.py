"""
DEPRECATED

Utility fns for 'dat' files.
Each line contains a space-separated tripe:
  * score (float)
  * id1   (int)
  * id2   (int)
"""

def read_dat(filename, score_type=float):
    'Each line: <int|float> <int> <int>'
    for ln in file(filename, 'r'):
        score, a_id, b_id = str.split(ln)
        yield score_type(score), int(a_id), int(b_id)

def write_dat(filename, triples):
    with open(filename, 'w') as file:
        for score, id1, id2 in triples:
            file.write('{0} {1} {2}\n'.format(score, id1, id2))
    
