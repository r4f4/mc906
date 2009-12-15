import zlib
import cPickle
from math import sqrt
from Stemmer import Stemmer
from numpy import dot, array
from cStringIO import StringIO

# Keep just one instance of this
stemmer = Stemmer('english')

class memoize:

    def __init__(self, function):
        self.function = function
        self.memoized = {}

    def __call__(self, *args):
        try:
            return self.memoized[args]
        except KeyError:
            self.memoized[args] = self.function(*args)
            return self.memoized[args]

class memoize2:

    def __init__(self, function):
        self.function = function
        self.memoized = {}

    def __call__(self, *args):
        try:
            return self.memoized[args]
        except KeyError:
            self.memoized[args] = self.function(*args)
            self.memoized[args[::-1]] = self.memoized[args]
            return self.memoized[args]


@memoize
def stemWord(word):
    return stemmer.stemWord(word)

def norm(vector):
    return sqrt(dot(vector, vector.conj()))

def encode_document(doc, factor=5):
    """
    Serialize and compress a document so we can use less memory
    """

    s = StringIO()
    cPickle.dump(doc, s)
    compressed_str = zlib.compress(s.getvalue(), factor)
    return compressed_str

def decode_document(compressed_str):
    """
    Decompress and deserialize a document
    """

    uncompressed_str = zlib.decompress(compressed_str)
    s = StringIO(uncompressed_str)
    doc = cPickle.load(s)
    return doc


@memoize2
def distance(str1, str2):
    """ Calculate the distance between doc1 and doc2. Assume the documents are
    encoded (by encode_document).

    The distance is calculated as a cosine measure:

    dist = cos(d1, d2) = (d1 . d2) / ||d1|| ||d2||
    """

    doc1 = decode_document(str1)
    doc2 = decode_document(str2)

    assert isinstance(doc1, type(doc2)), \
           "objects type mismatch: %s and %s." % (type(doc1), type(doc2))

    return 1 - dot(doc1.char_vector, doc2.char_vector) / \
               (norm(doc1.char_vector) * norm(doc2.char_vector))

def calc_centroid(cluster):
    """
    Calculate the centroid from the given cluster.

    Note: This function assumes that the methods __add__ and __div__ are
    correctly set on the cluster's elements.
    """

    assert len(cluster) > 0

    c0 = cluster.pop()
    res = sum((decode_document(d) for d in cluster), decode_document(c0)) / \
            (len(cluster) + 1)
    cluster.add(c0)
    return encode_document(res)

def normalize(doc, words, idf=True):
    """
    Normalize the characteristic vector of Document doc.

    :idf Whether use or not the inverse document frequence

    Note:
        be aware that parsing additional documents after calling this
        method will result in different characterist vectors from before.
        So parse all documents needed first.
    """

    freq = doc.freq
    # Each term is weightened by the inverse document frequency in the
    # document collection
    if idf is True:
        darray = array(map(lambda w: float(freq.get(w, 0)) / words[w],
            words.keys()))
    else:
        darray = array(map(lambda w: float(freq.get(w, 0)), words.keys()))
    # Normalize the vector
    darray /= norm(darray)
    doc.char_vector = darray


def get_clusters(centroids, data):
    """
    Get list containing sest with the data belonging to them.

    :centroids A ordered set of centroids.
    :data A set (not necesseraly ordered) of data.
    """

    clusters = [set() for _ in xrange(len(centroids))]
    for i in data:
        _, j = min([(distance(c, i), j) for j, c in enumerate(centroids)])
        clusters[j].add(i)
    return clusters
