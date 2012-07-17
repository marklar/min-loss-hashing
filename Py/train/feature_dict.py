import pickle


class FeatureDico:
    'Each feature must have a unique int ID.'

    def __init__(self, dico):
        self._dico = dico

    def get(self, key):
        return self._dico.setdefault(key, len(self._dico) + 1)

    # -- serialize --
    @classmethod
    def from_file(cls, fn):
        with open(fn, 'r') as f:
            d = eval(f.read())
        return cls(d)

    def to_file(self, path):
        with open(path, 'w') as f:
            f.write(repr(self._dico))

    # -- pickle --
    @classmethod
    def from_pickle(cls, fn):
        with open(fn, 'r') as f:
            d = pickle.load(f)
        return cls(d)

    def to_pickle(self, path):
        with open(path, 'w') as f:
            pickle.dump(self._dico, f)

def main():
    # fd = FeatureDico.from_file('foo.txt')
    fd = FeatureDico.from_pickle('foo.dat')
    print fd.get('foo')
    print fd.get('foo')
    print fd.get('bar')
    print fd.get('baz')
    print fd.get('blurfl')
    # fd.to_file('foo.txt')
    fd.to_pickle('foo.dat')
    print FeatureDico.__doc__

main()
