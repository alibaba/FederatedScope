

class Father:
    def __init__(self, a: float, b: str):
        self.a = a
        self.b = b
        print('Father', a, b)


class Son(float):
    def __new__(cls, floatDigit, b):
        return super(Son, cls).__new__(cls, floatDigit)

    def __init__(self, a, b):
        # super(Son, self).__init__(self)
        # super(Son, self).__init__(a, b)
        print(a,b)
        pass

if __name__ == '__main__':
    son = Son(1, "b")
    print(son)