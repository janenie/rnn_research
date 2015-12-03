from numpy import *
import itertools
import time
import sys

# Import NN utils
from nn.base import NNBase
from nn.math import softmax, sigmoid, make_onehot
from nn.math import MultinomialSampler, multinomial_sample
from misc import random_weight_matrix


class RNNLM(NNBase):
    """
    Implements an RNN language model of the form:
    h(t) = sigmoid(H * h(t-1) + L[x(t)])
    y(t) = softmax(U * h(t))
    where y(t) predicts the next word in the sequence

    U = |V| * dim(h) as output vectors
    L = |V| * dim(h) as input vectors

    You should initialize each U[i,j] and L[i,j]
    as Gaussian noise with mean 0 and variance 0.1

    Arguments:
        L0 : initial input word vectors
        U0 : initial output word vectors
        alpha : default learning rate
        bptt : number of backprop timesteps
    """

    def __init__(self, L0, U0=None,
                 alpha=0.005, rseed=10, bptt=1):
        random.seed(rseed)
        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
        param_dims = dict(H = (self.hdim, self.hdim),
                          U = L0.shape)
        # note that only L gets sparse updates
        param_dims_sparse = dict(L = L0.shape)
        NNBase.__init__(self, param_dims, param_dims_sparse)

        #### YOUR CODE HERE ####
        self.sparams.L = L0.copy()
        self.params.H = random_weight_matrix(self.hdim, self.hdim)
        self.alpha = alpha
        self.bptt = bptt


        # Initialize word vectors
        # either copy the passed L0 and U0 (and initialize in your notebook)
        # or initialize with gaussian noise here
        if U0 is not None:
            self.params.U = U0.copy()
        else:
            sigma = 0.1
            mu = 0
            #self.params.U = random.normal(mu, sigma, (self.vdim, self.hdim))
            self.params.U = sigma*random.randn(self.vdim, self.hdim) + mu

        # Initialize H matrix, as with W and U in part 1

        #### END YOUR CODE ####


    def _acc_grads(self, xs, ys):
        """
        Accumulate gradients, given a pair of training sequences:
        xs = [<indices>] # input words
        ys = [<indices>] # output words (to predict)

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.H += (your gradient dJ/dH)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # update row

        Per the handout, you should:
            - make predictions by running forward in time
                through the entire input sequence
            - for *each* output word in ys, compute the
                gradients with respect to the cross-entropy
                loss for that output word
            - run backpropagation-through-time for self.bptt
                timesteps, storing grads in self.grads (for H, U)
                and self.sgrads (for L)

        You'll want to store your predictions \hat{y}(t)
        and the hidden layer values h(t) as you run forward,
        so that you can access them during backpropagation.

        At time 0, you should initialize the hidden layer to
        be a vector of zeros.
        """

        # Expect xs as list of indices
        ns = len(xs)

        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
        hs = zeros((ns+1, self.hdim))
        # predicted probas
        ps = zeros((ns, self.vdim))

        #### YOUR CODE HERE ####
        L = self.sparams.L
        U = self.params.U
        H = self.params.H
        

        ##
        # Forward propagation
        for i in xrange(ns):
            hs[i+1] = sigmoid(H.dot(hs[i]) + L[xs[i]])
            ps[i] = softmax(U.dot(hs[i+1]))
        
        
        ##
        # Backward propagation through time
        for j in xrange(ns):
            target = make_onehot(ys[j], self.vdim)
            dscore = ps[j] - target
            dU = outer(dscore, hs[j+1])
            self.grads.U += dU
            dhidden = U.T.dot(dscore) * hs[j+1] * (1 - hs[j+1])
            
            for i in xrange(j, j - self.bptt, -1):
                if i >= 0:
                    self.sgrads.L[xs[i]] = dhidden
                    self.grads.H += outer(dhidden, hs[i])
                    dhidden = H.T.dot(dhidden)* hs[i] * (1 - hs[i])


        #### END YOUR CODE ####



    def grad_check(self, x, y, outfd=sys.stderr, **kwargs):
        """
        Wrapper for gradient check on RNNs;
        ensures that backprop-through-time is run to completion,
        computing the full gradient for the loss as summed over
        the input sequence and predictions.

        Do not modify this function!
        """
        bptt_old = self.bptt
        self.bptt = len(y)
        print >> outfd, "NOTE: temporarily setting self.bptt = len(y) = %d to compute true gradient." % self.bptt
        NNBase.grad_check(self, x, y, outfd=outfd, **kwargs)
        self.bptt = bptt_old
        print >> outfd, "Reset self.bptt = %d" % self.bptt


    def compute_seq_loss(self, xs, ys):
        """
        Compute the total cross-entropy loss
        for an input sequence xs and output
        sequence (labels) ys.

        You should run the RNN forward,
        compute cross-entropy loss at each timestep,
        and return the sum of the point losses.
        """

        J = 0
        #### YOUR CODE HERE ####
        ns = len(xs)
        hs = zeros((ns+1, self.hdim))
        # predicted probas
        ps = zeros((ns, self.vdim))

        #### YOUR CODE HERE ####
        L = self.sparams.L
        U = self.params.U
        H = self.params.H
        
        ##
        # Forward propagation
        for i in xrange(ns):
            hs[i+1] = sigmoid(H.dot(hs[i]) + L[xs[i]])
            ps[i] = softmax(U.dot(hs[i+1]))
            J -= log(ps[i][ys[i]])
        
        

        #### END YOUR CODE ####
        return J


    def compute_loss(self, X, Y):
        """
        Compute total loss over a dataset.
        (wrapper for compute_seq_loss)

        Do not modify this function!
        """
        if not isinstance(X[0], ndarray): # single example
            return self.compute_seq_loss(X, Y)
        else: # multiple examples
            return sum([self.compute_seq_loss(xs,ys)
                       for xs,ys in itertools.izip(X, Y)])

    def compute_mean_loss(self, X, Y):
        """
        Normalize loss by total number of points.

        Do not modify this function!
        """
        J = self.compute_loss(X, Y)
        ntot = sum(map(len,Y))
        return J / float(ntot)


    def generate_sequence(self, init, end, maxlen=100):
        """
        Generate a sequence from the language model,
        by running the RNN forward and selecting,
        at each timestep, a random word from the
        a word from the emitted probability distribution.

        The MultinomialSampler class (in nn.math) may be helpful
        here for sampling a word. Use as:

            y = multinomial_sample(p)

        to sample an index y from the vector of probabilities p.


        Arguments:
            init = index of start word (word_to_num['<s>'])
            end = index of end word (word_to_num['</s>'])
            maxlen = maximum length to generate

        Returns:
            ys = sequence of indices
            J = total cross-entropy loss of generated sequence
        """

        J = 0 # total loss
        ys = [init] # emitted sequence

        #### YOUR CODE HERE ####
        ps = zeros((maxlen, self.vdim))
        hs = zeros((maxlen, self.hdim))
        H = self.params.H
        L = self.sparams.L
        U = self.params.U
        
        start = init
        for i in xrange(maxlen):
            hs[i+1] = sigmoid(H.dot(hs[i]) + L[start])
            ps[i] = softmax(U.dot(hs[i+1]))
            start = multinomial_sample(ps[i])
            J -= log(ps[i][start])
            ys.append(start)
            
            if start == end:
                break

        #### YOUR CODE HERE ####
        return ys, J



class ExtraCreditRNNLM(RNNLM):
    """
    Implements an improved RNN language model,
    for better speed and/or performance.

    We're not going to place any constraints on you
    for this part, but we do recommend that you still
    use the starter code (NNBase) framework that
    you've been using for the NER and RNNLM models.
    """

    def __init__(self, L0, **kwargs):
        #### YOUR CODE HERE ####
        isCompression = False
        isME = False
        compression_size = 0
        alpha = 0.1
        bptt = 1
        class_size = 2
        U0 = zeros((10, 10))
        Lcluster = zeros(10)
        cwords = zeros((10, 10))
        cfreq = zeros(10)
        ngram_feat = 3
        hash_size = 10000
        gradient_cutoff = 5
        rseed = 0
        
        for key, value in kwargs.items():
            if key == "U0":
                U0 = value.copy()
            if key == "isCompression":
                isCompression = value
            if key == "compression_size":
                compression_size = value
            if key == "isME":
                isME = value
            if key == "bptt":
                bptt = value
            if key == "alpha":
                alpha = value
            if key == "Lcluster":
                Lcluster = value
            if key == "cwords":
                cwords = value
            if key == "cfreq":
                cfreq = value
            if key == "ngram":
                ngram_feat = value
            if key == "hash_size":
                hash_size = value
            if key == "cutoff":
                gradient_cutoff = value
            if key == "rseed":
                rseed = value
            if key == "class_size":
                class_size = value
        
        random.seed(rseed)
        #print L0
        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
        #print self.hdim
        self.cdim = compression_size # compression layer size
        self.isCompression = isCompression # True for self.cdim > 0
        #print self.isCompression
        
        self.class_size = class_size # making word clusters
        self.Udim = self.vdim + self.class_size
        self.cutoff = gradient_cutoff
        
        self.isME =  isME #max entropy optimization
        self.ngram = ngram_feat
        self.hsize = hash_size
        param_dims = {}
        if self.isCompression is True:
            param_dims = dict(H = (self.hdim, self.hdim),
                          C = (self.cdim, self.hdim),    
                          U = (self.Udim, self.cdim))
        else:
            param_dims = dict(H = (self.hdim, self.hdim),    
                          U = (self.Udim, self.hdim))
        # note that only L gets sparse updates
        param_dims_sparse = dict(L = L0.shape)    
        NNBase.__init__(self, param_dims, param_dims_sparse)

        #### YOUR CODE HERE ####
        self.sparams.L = L0.copy()
        #word cluster informations
        self.Lcluster = Lcluster.copy() #cluster for every word
        self.cfreq = cfreq.copy() #every cluster containing word numbers
        self.cwords = cwords.copy() #every cluster containing word indexs
        self.htable = zeros(self.hsize)
        #print "CWORD SIZE ",cwords.shape
        
        self.params.H = random_weight_matrix(self.hdim, self.hdim)
        
        self.alpha = alpha
        self.bptt = bptt


        # Initialize word vectors
        # either copy the passed L0 and U0 (and initialize in your notebook)
        # or initialize with gaussian noise here
        if isCompression is True:
            self.params.C = random_weight_matrix(self.cdim, self.hdim)
        if U0 is not None:
            self.params.U = U0.copy()
        else:
            sigma = 0.1
            mu = 0
            if self.isCompression:
                self.params.U = sigma*random*randn(self.Udim, self.cdim) + mu
            else:
                self.params.U = sigma*random*randn(self.Udim, self.hdim) + mu
            #self.params.U = random.normal(mu, sigma, (self.vdim, self.hdim))
        #### END YOUR CODE ####

    def _acc_grads(self, xs, ys):
        #### YOUR CODE HERE ####
        # Expect xs as list of indices
        ns = len(xs)

        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
        hs = zeros((ns+1, self.hdim))
        cs = zeros((ns, self.cdim)) #compression units
        # predicted probas
        ps = zeros((ns, self.Udim)) #for both words and cluster

        #### YOUR CODE HERE ####
        L = self.sparams.L
        
        Lc = self.Lcluster
        cfreq = self.cfreq
        cwords = self.cwords
        
        U = self.params.U
        H = self.params.H
        C = zeros((10, 10))
        htable = zeros(10)
        
        if self.isCompression is True:
            C = self.params.C
        if self.isME is True:
            htable = self.htable
        ##
        # Forward propagation
        for i in xrange(ns):
            hs[i+1] = sigmoid(H.dot(hs[i]) + L[xs[i]])
            #without maximum entropy optimization
            word_cluster = Lc[ys[i]]
            st_word = cwords[word_cluster, 0]
            ed_word = st_word + cfreq[word_cluster]
            #print st_word, ed_word, cfreq[word_cluster]
            if self.isCompression:
                cs[i] = sigmoid(C.dot(hs[i+1]))
                #if self.isME is False:
                ps[i, self.vdim:] = softmax(U[self.vdim:,:].dot(cs[i]))
                ps[i, st_word:ed_word] = softmax(U[st_word:ed_word].dot(cs[i]))
                #if self.isME is True:
                #    #find the hash value (ME value) for different 1-n gram
                #    htable[:self.ngram] = 0
                    
            else:
                #if self.isME is False:
                ps[i, self.vdim:] = softmax(U[self.vdim:,:].dot(hs[i+1]))
                ps[i, st_word:ed_word] = softmax(U[st_word:ed_word, :].dot(hs[i+1]))
                    
        
        ##
        # Backward propagation through time
        for j in xrange(ns):
            #word cluster and word parts
            word_cluster = Lc[ys[j]]
            st_word = cwords[word_cluster, 0]
            ed_word = st_word + cfreq[word_cluster]
            
            dscore_cluster = ps[j, self.vdim:] #clusters parts
            dscore_word = ps[j, 0:self.vdim] #word parts
            
            #cross entropy err for word and cluster
            dscore_cluster[word_cluster] -= 1
            dscore_word[ys[j]] -= 1
            dscore_word = dscore_word[st_word:ed_word]
            
            dhidden = zeros((self.hdim,))
            
            if self.isCompression is True:
                #dU = zeros((self.Udim, self.cdim))
                dU_word = outer(dscore_word, cs[j])
                dU_cluster = outer(dscore_cluster, cs[j])
                
                self.grads.U[st_word:ed_word] += dU_word
                self.grads.U[self.vdim:] += dU_cluster
                
                dclayer = (U[self.vdim:].T.dot(dscore_cluster) + U[st_word:ed_word].T.dot(dscore_word))*cs[j]*(1-cs[j])
                
                dC = outer(dclayer, hs[j+1])
                self.grads.C += dC
                
                dhidden = C.T.dot(dclayer) * hs[j+1] * (1- hs[j+1])
            else:
                #dU = zeros((self.Udim, self.hdim))
                dU_word = outer(dscore_word, hs[j+1])
                dU_cluster = outer(dscore_cluster, hs[j+1])
                
                self.grads.U[st_word:ed_word] += dU_word
                self.grads.U[self.vdim:] += dU_cluster
  
                dhidden = (U[st_word:ed_word].T.dot(dscore_word) + U[self.vdim:].T.dot(dscore_cluster)) * hs[j+1] * (1 - hs[j+1])
            
            for i in xrange(j, j - self.bptt, -1):
                if i >= 0:
                    dhidden[dhidden > self.cutoff] = self.cutoff
                    dhidden[dhidden < -self.cutoff] = -self.cutoff
                    self.sgrads.L[xs[i]] = dhidden
                    self.grads.H += outer(dhidden, hs[i])
                    dhidden = H.T.dot(dhidden)* hs[i] * (1 - hs[i])

        
        #### END YOUR CODE ####

    def compute_seq_loss(self, xs, ys):
        #### YOUR CODE HERE ####
        J = 0
        ns = len(xs)
        hs = zeros((ns+1, self.hdim))
        cs = zeros((ns, self.cdim))
        # predicted probas
        ps = zeros((ns, self.Udim))

        #### YOUR CODE HERE ####
        L = self.sparams.L
        Lc = self.Lcluster
        cfreq = self.cfreq
        cwords = self.cwords
        U = self.params.U
        H = self.params.H
        C = zeros((self.cdim, self.hdim))
        if self.isCompression is True:
            C = self.params.C
        ##
        # Forward propagation
        for i in xrange(ns):
            hs[i+1] = sigmoid(H.dot(hs[i]) + L[xs[i]])
            #without maximum entropy optimization
            word_cluster = Lc[ys[i]]
            st_word = cwords[word_cluster, 0]
            ed_word = st_word + cfreq[word_cluster]
            if self.isCompression:
                cs[i] = sigmoid(C.dot(hs[i+1]))
                ps[i, self.vdim:] = softmax(U[self.vdim:,:].dot(cs[i]))
                ps[i, st_word:ed_word] = softmax(U[st_word:ed_word,:].dot(cs[i]))
                #print ps[i, st_word:ed_word]
            else:
                ps[i, self.vdim:] = softmax(U[self.vdim:,:].dot(hs[i+1]))
                ps[i, st_word:ed_word] = softmax(U[st_word:ed_word,:].dot(hs[i+1]))
            
            #print ps[i, ys[i]], ps[i, self.vdim+word_cluster]
            J -= log(ps[i, ys[i]] * ps[i, self.vdim+word_cluster])
        return J
        #### END YOUR CODE ####

    def compute_loss(self, X, Y):
        """
        Compute total loss over a dataset.
        (wrapper for compute_seq_loss)

        Do not modify this function!
        """
        if not isinstance(X[0], ndarray): # single example
            return self.compute_seq_loss(X, Y)
        else: # multiple examples
            return sum([self.compute_seq_loss(xs,ys)
                       for xs,ys in itertools.izip(X, Y)])

    def compute_mean_loss(self, X, Y):
        """
        Normalize loss by total number of points.

        Do not modify this function!
        """
        J = self.compute_loss(X, Y)
        ntot = sum(map(len,Y))
        return J / float(ntot)
    
    def generate_sequence(self, init, end, maxlen=100):
        #### YOUR CODE HERE ####
        raise NotImplementedError("generate_sequence() not yet implemented.")
        #### END YOUR CODE ####