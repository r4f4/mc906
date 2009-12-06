import os
import re


class Document(object):
    """
    Document object
    """
    def __init__(self, name, ignore_words=set()):
        # FIXME: Check for errors
        self.freq = dict()
        with open(name, 'r') as f:
            for l in f.readlines():
                for w in re.split(r'[^a-z]+', l.lower().strip()):
                    word = unicode(w)
                    if not word is u'' and word not in ignore_words:
                        self.freq[word] = self.freq.get(word, 0) + 1

    def most_frequent_words_get(self, percent):
        return map(lambda l: l[0], sorted(self.wfreq.iteritems(),
            key=lambda (k, v): (v, k), reverse=True))[:int(percent*len(self.wfreq))]


if __name__ == "__main__":
    with open('english', 'r') as f:
        ignore_words = set([unicode(word.strip()) for word in f.readlines()])
    path = './cluster-txt/messages/alt-atheism-51119.txt'
    d = Document(path, ignore_words)
    print(d.most_frequent_words_get(1))

    print '\n\nDone\n\n'
