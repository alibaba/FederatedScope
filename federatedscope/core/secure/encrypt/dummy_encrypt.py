DEFAULT_KEY_SIZE = 3072


class DummyEncryptKeypair(object):
    def __init__(self, n_length=DEFAULT_KEY_SIZE):
        self.n_length = n_length

    @staticmethod
    def generate_keypair():
        n = p = q = None
        public_key = DummyEncryptPublicKey(n)
        private_key = DummyEncryptPrivateKey(public_key, p, q)
        return public_key, private_key


class DummyEncryptPublicKey(object):
    def __init__(self, n):
        self.n = n

    def __eq__(self, other):
        return self.n == other.n

    def encrypt(self, value):
        ciphertext = value
        encrypted_num = DummyEncryptNumber(self, ciphertext)
        return encrypted_num


class DummyEncryptPrivateKey(object):
    def __init__(self, public_key, p, q):
        self.public_key = public_key
        self.p = p
        self.q = q

    def __eq__(self, other):
        return self.p == other.p and self.q == other.q

    def decrypt(self, encrypted_num):
        assert self.public_key == encrypted_num.public_key, \
            'public_key not matched!'
        decrypt_val = encrypted_num.ciphertext
        return decrypt_val


class DummyEncryptNumber(object):
    def __init__(self, public_key, ciphertext):
        self.public_key = public_key
        self.ciphertext = ciphertext

    def __add__(self, other):
        if isinstance(other, DummyEncryptNumber):
            return DummyEncryptNumber(self.public_key,
                                      self.ciphertext + other.ciphertext)
        return DummyEncryptNumber(self.public_key, self.ciphertext + other)

    def __mul__(self, other):
        if isinstance(other, DummyEncryptNumber):
            return DummyEncryptNumber(self.public_key,
                                      self.ciphertext * self.ciphertext)
        return DummyEncryptNumber(self.public_key, self.ciphertext * other)
