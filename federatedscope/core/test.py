

class A(dict):
    pass
    # __getattr__ = dict.__getitem__
    # __setattr__ = dict.__setitem__
    #
    # def __getitem__(self, item):
    #     return self[item]




if __name__ == '__main__':
    a= A()
    pass
    print("fd")