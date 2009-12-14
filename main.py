#!/usr/bin/env python

import os
from kmeans import Kmeans, choose_initial_pp
from parse import Parser
from re import split
from util import distance, calc_centroid, normalize, get_clusters

def print_docs(centroids, data):
    clusters = get_clusters(centroids, data)
    for i, c in enumerate(clusters):
        print 'Clusters #%d' % i
        names = set([os.path.basename(split(r'-[0-9]+.txt$', doc.filename)[0])
            for doc in c])
        print names

if __name__ == "__main__":
    parser = Parser('english')
    # Path to the directory containing the messages
    path="./cluster-txt/messages/"
    k = 20

    filelist = [(path + f) for f in os.listdir(path)]
    parser.parse(filelist[:60], False)
    map(lambda doc: normalize(doc, parser.words), parser.docset)

    kmeans = Kmeans(parser.docset, k, distance, calc_centroid)
    print_docs(kmeans.result(), parser.docset)
