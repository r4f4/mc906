from re import split
from numpy import dot, array
from util import norm, stemWord

class Document(object):
    """
    A document class
    """
    def __init__(self, pathname, ignore_words=None, stem=True):
        """
        :pathname The pathname of the document.
        :ignore_words A set of words to ignore when parsing the document.
        :stem The words will be stemmed if it is True.
        """

        self._filename = pathname
        self._char_vector = None
        self._freq = dict()
        self.stem = stem
        self._ignore_words = ignore_words or set()

    def __len__(self):
        """ The number of words in the current document """

        return len(self.freq)

    def __add__(self, val):

        docres = Document(None)
        # Add words frequencies in both documents
        if isinstance(val, type(self)):
            docres.char_vector = self.char_vector + val.char_vector
        # Increase each frequence by other in case other is a scalar
        elif isinstance(val, int) or isinstance(val, float):
            docres.char_vector = self.char_vector + val
        return docres

    def __sub__(self, val):

        docres = Document(None)
        if isinstance(val, type(self)):
            docres.char_vector = self.char_vector + val.char_vector
        else:
            docres.char_vector = self.char_vector - val
        return docres

    def __div__(self, val):
        """ The division by a scalar or other document """

        docres = Document(None)
        docres.char_vector = self.char_vector / val
        return docres

    def read(self):
        with open(self.filename, 'r') as f:
            for l in f.readlines():
                for w in split(r'[^a-z]+', l.strip().lower()):
                    word = unicode(w)
                    if not word is u'' and word not in self._ignore_words:
                        if self.stem is True:
                            word = stemWord(word)
                        self.freq[word] = self.freq.get(word, 0) + 1
                    del word

    def words(self):
        """ Get a list of words parsed from the current document """

        return self.freq.keys()

    def words_frequence(self):
        """ Get an iterator of items in the format (word, frequence) from all
        words parsed from the current document. """

        return self.freq.iteritems()

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

    def parse(self, doclist, stem=True, verbose=True):
        """ Parse a list of documents """

        for docname in doclist:
            if verbose is True:
                print 'Parsing %s' % docname
            doc = Document(docname, self._ignored, stem)
            doc.read()
            # Add more words to the set of significant and distinct words
            # and keep track of their counting
            freq = doc.freq
            map(lambda w: self._words.__setitem__(w,
                self._words.get(w, 0) + freq[w]), doc.words())
            self._docset.add(doc)

    @property
    def docset(self):
        """
        Get a set of Documents.
        """

        return self._docset

    @property
    def words(self):
        """
        Get overall word frequencies
        """

        return self._words
