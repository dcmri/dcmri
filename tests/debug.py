from dcmri.ui import ParsView



class A:
    def __init__(self):
        self._pars = {'a': 1}

    def print(self):
        print(self._pars['a'])

class B:
    def __init__(self):
        self._a = A()
        self._pars = {'b': 2}
    def print(self):
        print(self._pars['b'])


def test_pars_view():

    b = B()
    b._a.print()

    d = ParsView(b._pars, b._a._pars)
    d['a'] = 3
    b._a.print()

    d = ParsView(b._a._pars, b._pars)
    d['a'] = -3
    b._a.print()

    d['c'] = 0



if __name__ == "__main__":

    test_example()
    
    print('Done!!')