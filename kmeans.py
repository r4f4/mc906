import random

class Kmeans(object):
    """
    k-means algorithm
    """
    
    def __init__(self, data, clusters, metric, avg):
        
        self.centroids = random.sample(data, clusters)

        err = -1
        while True:
            bins = [set() for k in xrange(clusters)]
            
            for i in data:
                ml = [(metric(c,i), ic) for ic,c in enumerate(self.centroids)]
                ml_min,_ =  min(ml)
                c = random.choice( [k for m,k in ml if m == ml_min] )
                bins[c].add(i)

            for bi,b in enumerate(bins):
                self.centroids[bi] = avg(b)
                
            olderr = err
            err = sum( [sum( [metric(d,self.centroids[c]) for d in b] )
                        for c,b in enumerate(bins)] )
            
            if err - olderr < 0.0005:
                break

    def result(self):
        return self.centroids
