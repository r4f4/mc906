#!/usr/bin/env python

import os
import numpy
from kmeans import Kmeans, choose_initial, choose_initial_pp
from parse import Parser
from re import split
from util import distance, calc_centroid, normalize, get_clusters

def print_result(freqs, errors):
    for i, (f, e) in enumerate(zip(freqs, errors)):
        print 'Cluster #%d' % i
        print ', '.join(f.keys())
        print 'Error: %d\n' % e

def get_docs_frequencies(clusters):
    docs = []
    for c in clusters:
        freq = {}
        for doc in c:
            name = os.path.basename(split(r'-[0-9]+.txt$', doc.filename)[0])
            freq[name] = freq.get(name, 0) + 1
        docs.append(freq)
    return docs

def calc_error(freqs):
    errors = []
    for freq in freqs:
        f, name = max((f, name) for name, f in freq.iteritems())
        err = sum(freq[f] for f in filter(lambda n: n != name, freq.keys()))
        errors.append(err)
    return errors

if __name__ == "__main__":
    parser = Parser('english')
    # Path to the directory containing the messages
    path="./cluster-txt/messages/"

    filelist = [(path + f) for f in os.listdir(path)]

    for i in True, False:
        for j in True, False:
            parser.parse(filelist, i)
            map(lambda doc: normalize(doc, parser.words, j), parser.docset)
            for chooser in choose_initial_pp, choose_initial:
                for k in 10, 20, 30, 40:
                    errors = []
                    print '\nStemming words: %s' % i
                    print 'Using IDF: %s' % j
                    print 'Running with %d centroids' % k
                    if chooser is choose_initial:
                        print 'Chooser: normal'
                    else:
                        print 'Chooser: plusplus'
                    for _ in xrange(13):
                        kmeans = Kmeans(parser.docset, k, distance,
                                calc_centroid, chooser)
                        clusters = get_clusters(kmeans.result(), parser.docset)
                        freqs = get_docs_frequencies(clusters)
                        errors.append(sum(calc_error(freqs)))
                    print 'Error mean: %d and median: %d' % \
                        (numpy.mean(errors), numpy.median(errors))
