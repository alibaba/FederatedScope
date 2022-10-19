# You can refer to pyphe for the detail implementation. (
# https://github.com/data61/python-paillier/blob/master/phe/paillier.py)
# Or implement an effective version of Paillier (<Public-key cryptosystems
# based on composite degree residuosity classes>)

DEFAULT_KEYSIZE = 3072


def generate_paillier_keypair(n_length=DEFAULT_KEYSIZE):
    """Generate public key and private key used Paillier`.

    Args:
      n_length: key size in bits.

    Returns:
      tuple: The generated :class:`PaillierPublicKey` and
      :class:`PaillierPrivateKey`
    """
    n = p = q = None
    public_key = PaillierPublicKey(n)
    private_key = PaillierPrivateKey(public_key, p, q)

    return public_key, private_key


class PaillierPublicKey(object):
    """Contains a public key and associated encryption methods.
    """
    def __init__(self, n):
        pass

    def encrypt(self, value):
        # We only provide an abstract implementation here

        return value


class PaillierPrivateKey(object):
    """Contains a private key and associated decryption method.
    """
    def __init__(self, public_key, p, q):
        pass

    def decrypt(self, encrypted_number):
        # We only provide an abstract implementation here

        return encrypted_number
