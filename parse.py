import os
import re
from numpy import dot, array
from math import sqrt

class Document(object):
    """
    A document class
    """
    def __init__(self, pathname, ignore_words=None):
        """
        :pathname The pathname of the document.
        :ignore_words A set of words to ignore when parsing the document.
        """

        self._filename = pathname
        self._char_vector = None
        self._freq = dict()
        if ignore_words is None:
            ignore_words = set()

        with open(pathname, 'r') as f:
            for l in f.readlines():
                for w in re.split(r'[^a-z]+', l.strip().lower()):
                    word = unicode(w)
                    if not word is u'' and word not in ignore_words:
                        self.freq[word] = self.freq.get(word, 0) + 1

    def __len__(self):
        """ The number of words in the current document """

        return len(self.freq)

    def words(self):
        """ Get a list of words parsed from the current document """

        return self.freq.keys()

    def words_frequence(self):
        """ Get an iterator of items in the format (word, frequence) from all
        words parsed from the current document. """

        return self.freq.iteritems()

    def distance(self, other):
        """ Calculate the distance from the current object with other.

        :other A Document instance with which we want to calculate the
        distance.

        The distance is calculated as a cosine measure:

            dist = cos(d1, d2) = (d1 . d2) / ||d1|| ||d2||
        """

        norm = lambda vector: sqrt(dot(vector, vector.conj()))
        if not isinstance(other, type(self)):
            raise TypeError("expected a Parser but got a %s instead." %
                    type(other))

        return dot(self._char_vector, other._char_vector) / \
                (norm(self._char_vector) * norm(other._char_vector))

    @property
    def freq(self):
        """ Get a dictionary with frequencies indexed by the words """

        return self._freq

    @property
    def char_vector(self):
        """ Get the characteristic vector for current document """

        return self._char_vector

    @char_vector.setter
    def char_vector(self, value):
        """
        Set the characterist vector for current document.

        The characteristic vector *must* be an instance of numpy.array or it
        will raise TypeError in that case.
        """

        if isinstance(value, type(array(None))):
            self._char_vector = value
        else:
            raise TypeError("expected a numpy.array but got %s instead." %
                    type(value))

    @property
    def filename(self):
        """ Get current document's pathname """

        return self._filename


if __name__ == "__main__":
    with open('english', 'r') as f:
        ignore_words = set([unicode(word.strip()) for word in f.readlines()])
    path = './cluster-txt/messages/alt-atheism-51119.txt'
    d = Document(path, ignore_words)
    print(d.most_frequent_words_get(1))

    print '\n\nDone\n\n'
