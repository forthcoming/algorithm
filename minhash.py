import itertools


def get_minhash(shingles, n_hashes):
    signature = []
    for i in range(n_hashes):
        salt = str(i)
        minhash = float('inf')
        for shingle in shingles:
            candidate = hash(shingle + salt)
            if candidate < minhash:
                minhash = candidate
        signature.append(minhash)
    return signature


def get_band_hashes(signature, band_size):
    band_hashes = []
    band_hash = 0
    for i, each in enumerate(signature):
        if i and i % band_size == 0:
            band_hashes.append(band_hash)
            band_hash = 0
        band_hash += hash(each)
    return band_hashes


def get_similar_docs(documents, n_hashes=400, band_size=7, shingle_size=3, collect_indexes=True):
    hash_bands = {}
    doc_num = 0
    for doc in documents:
        shingles = {doc[i:i + shingle_size] for i in range(len(doc) - shingle_size + 1)}
        signature = get_minhash(shingles, n_hashes)
        band_hashes = get_band_hashes(signature, band_size)
        doc_member = doc_num if collect_indexes else doc
        for i in range(len(band_hashes)):
            if i not in hash_bands:
                hash_bands[i] = {}
            if band_hashes[i] not in hash_bands[i]:
                hash_bands[i][band_hashes[i]] = [doc_member]
            else:
                hash_bands[i][band_hashes[i]].append(doc_member)
        doc_num += 1

    similar_documents = set()
    for i in hash_bands:
        for hash_num in hash_bands[i]:
            if len(hash_bands[i][hash_num]) > 1:
                for pair in itertools.combinations(hash_bands[i][hash_num], r=2):
                    similar_documents.add(pair)
    return similar_documents


if __name__ == '__main__':
    docs = [
        'g aaiaaaeeriaaeeraeabgboo ee  ir ',
        ' buearreirbe rgrbiaobarbe ebteeuai r',
        'argatgoruiia obabiraiti tagebie eettau',
        'igabggigirrtittibe abatgtebaaio',
        'retegiegi btritogriuuettutgteu ia ',
        'eateeaaurttetigaa e togei a  e oeagriora',
        'uere tguge aouii brae aogtbobgt u ',
        'ibiratgruiubugua toutuuaair rrr',
        'tgoaoebuboe ioeur tbibaiauagu t',
        'argatgoiruiia obabiraiti tagebie eettau',
    ]
    similar_docs = get_similar_docs(docs, 100)
    for index in similar_docs:
        print(index)

'''
import random, struct
from hashlib import sha1
import numpy as np

class MinHash:
    _mersenne_prime = (1 << 61) - 1  # http://en.wikipedia.org/wiki/Mersenne_prime
    _max_hash = (1 << 32) - 1

    def __init__(self, num_perm=128):  # num_perm: Number of random permutation functions.
        self.hashvalues = np.ones(num_perm, dtype=np.uint64)*self._max_hash  # [(1 << 32) - 1]*128
        generator = np.random.RandomState(1)
        result=[(generator.randint(1, self._mersenne_prime, dtype=np.uint64),generator.randint(0, self._mersenne_prime, dtype=np.uint64)) for _ in range(num_perm)]
        self.permutations = np.array(result, dtype=np.uint64).T

    def update(self, b):
        hv = struct.unpack('<I', sha1(b).digest()[:4])[0]
        a, b = self.permutations
        phv = np.bitwise_and((a * hv + b) % self._mersenne_prime, np.uint64(self._max_hash))
        self.hashvalues = np.minimum(phv, self.hashvalues)

    def jaccard(self, other):
        return np.count_nonzero(self.hashvalues==other.hashvalues) / len(self.hashvalues)

if __name__=='__main__':
    minhash=MinHash()
    minhash.update(b'minhash')
    minhash.update(b'aa')
    minhash.update(b'minhash1')
    minhash.update(b'1')
    minhash.update(b'2')
'''
