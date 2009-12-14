import random

def choose_initial(data, k, distfunc=None):
    """
    Choose randomly k different centroids.

    :data The elements being clustered
    :k The number of clusters
    :distfunc Ignored
    """
    return random.sample(data, k)

def choose_initial_pp(data, k, distfunc):
    """
    Choose randomly k different centroids using the kmeans++ heuristic
    by David Arthur and Sergei Vassilvitskii (see the article "k-means++:
    The Advantages of Careful Seeding".

    :data The elements being clustered
    :k The number of clusters
    :distfunc Function to calculate the distance between two elements.
    """
    from bisect import bisect

    # Calculate squared distance
    distance2 = lambda c, x: distfunc(c, x)

    # The first centroid is a random one
    centroids = [random.choice(data)]

    tries = 0
    while len(centroids) < k and tries < (k * 5):
        mindists = [min((distance2(c, x), x) for c in centroids)
                    for x in data]
        # Divide because we add it twice: first for (c, x) and then for (x, c)
        totaldist = float(sum(d for d, _ in mindists))
        probs = [(d / totaldist) for d, _ in mindists]
        addedProb = []
        last = 0
        for p in probs:
            last += p
            addedProb.append(last)
        pos = bisect(addedProb, random.random())
        centroids.append(data[pos])
        tries += 1

    if len(centroids) < k:
        centroids.extend(random.sample(data, k - len(centroids)))

    return centroids

class Kmeans(object):
    """
    k-means algorithm
    """

    def __init__(self, data, k, distfunc=None, centroidfunc=None,
            chooser=choose_initial, tol=0.0005):
        """
        The K-means algorithm

        :data A data set
        :k The number of clusters
        :distfunc A function to calculate the distance between two elements
        from data.
        :centroidfunc A function to calculate the centroid of the cluster. It
        will receive a cluster.
        :chooser A function to choose the initial k centroids. It must receive
        the data set, the number of clusters k and the distfunc.
        :tol Error tolerance
        """

        assert distfunc is not None, "distfunc can't be None"
        assert centroidfunc is not None, "centroidfunc can't be None"

        self.centroids = chooser(data, k, distfunc)

        err = -1
        niter = 1
        while True:
            bins = [set() for _ in xrange(k)]
            #print 'Iteration #%d' % niter
            niter += 1

            for i in data:
                ml = [(distfunc(c, i), ic) for ic, c in enumerate(self.centroids)]
                ml_min, _ =  min(ml)
                c = random.choice([j for m, j in ml if m == ml_min])
                bins[c].add(i)

            for bi, b in enumerate(bins):
                self.centroids[bi] = centroidfunc(b)

            olderr = err
            err = sum([sum([distfunc(d, self.centroids[c]) for d in b])
                      for c, b in enumerate(bins)])

            if err - olderr < tol:
                break

    def result(self):
        return self.centroids
