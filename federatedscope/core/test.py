

def mywr(func):
    def wrapper(*args, **kw):
        pass
    return wrapper

@mywr
def clear(a, b=None, **kwargs):
    pass


if __name__ == '__main__':
    clear(1, b=2)