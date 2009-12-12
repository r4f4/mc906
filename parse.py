import os
import re
from numpy import dot, array
from math import sqrt
from Stemmer import Stemmer

class Document(object):
    """
    A document class
    """
    def __init__(self, pathname, ignore_words=None, stem=True):
        """
        :pathname The pathname of the document.
        :ignore_words A set of words to ignore when parsing the document.
        :stem If True the words will be stemmed.
        """

        self._filename = pathname
        self._char_vector = None
        self._freq = dict()
        if ignore_words is None:
            ignore_words = set()
        stemmer = Stemmer('english')

        with open(pathname, 'r') as f:
            for l in f.readlines():
                for w in re.split(r'[^a-z]+', l.strip().lower()):
                    word = unicode(w)
                    if not word is u'' and word not in ignore_words:
                        if stem is True:
                            word = stemmer.stemWord(word)
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

        :other A Document instance from which we want to calculate the
        distance.

        The distance is calculated as a cosine measure:

            dist = cos(d1, d2) = (d1 . d2) / ||d1|| ||d2||
        """

        if not isinstance(other, type(self)):
            raise TypeError("expected a Parser but got a %s instead." %
                    type(other))

        # Store this function to avoid duplication of code
        norm = lambda vector: sqrt(dot(vector, vector.conj()))

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

class Parser(object):
    """
    A text file parser class.
    """

    def __init__(self, wignore_file):
        """
        :wignore_file Pathname of a file containing, on each line, a word to be
        ignored when parsing.
        """

        self._docset = set()
        self._ignored = None
        self._words = dict()

        with open(wignore_file, 'r') as f:
            self._ignored = set([unicode(w.strip()) for w in f.readlines()])

    def __len__(self):
        """ Return number of documents already parsed """

        return len(self._docset)

    def _parse_single(self, docname):
        doc = Document(docname, self._ignored)
        # Add more words to the set of significant and distinct words
        # and keep track of their counting
        freq = doc.freq
        map(lambda w: self._words.__setitem__(w,
            self._words.get(w, 0) + freq[w]), doc.words())
        self._docset.add(doc)

    def parse(self, doclist):
        """ Parse a single document or a list of them """

        if isinstance(doclist, list):
            map(lambda doc: self._parse_single(doc), doclist)
        else:
            self._parse_single(doclist)

    @property
    def docset(self):
        """
        Get a set of Documents with their characteristic vector set accordinly.

        Note:
            be aware that parsing additional documents after calling this
            method will result in different characterist vectors from before.
            So parse all documents needed first.
        """

        word_list = self._words.keys()
        for doc in self._docset:
            freq = doc.freq
            # Each term is weightened by the inverse document frequency in the
            # document collection
            darray = array(map(lambda w: float(freq.get(w, 0)) / self._words[w],
                word_list))
            norm = sqrt(dot(darray, darray.conj()))
            # Normalize the vector
            map(lambda (i, v): darray.put(i, v / norm), enumerate(darray))
            doc.char_vector = darray
        del word_list

        return self._docset


if __name__ == "__main__":
    with open('english', 'r') as f:
        ignore_words = set([unicode(word.strip()) for word in f.readlines()])
    path = './cluster-txt/messages/alt-atheism-51119.txt'
    d = Document(path, ignore_words)
    print(d.most_frequent_words_get(1))

    print '\n\nDone\n\n'
