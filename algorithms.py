# encoding: utf-8

import collections
import re
import sys
import time
import utils_sys as utils
import heapq
from operator import itemgetter

# find LCS (contiguous); for the non-contiguous, use lcs()
from difflib import SequenceMatcher


class Graph(object):
    def __init__(adjlist=None, vertices=None, V=None): 
        self.adj = {}
        self.V = V
        self.path = {}  # keep track of all reachable vertics starting from a given vertex v

        if adjlist is not None: 
            self.V = len(adjlist)
            for h, vx in adjlist.items(): 
                self.adj[h] = vx

        elif vertices is not None: 
            assert hasattr(vertices, '__iter__')
            self.V = len(vertices)
            for v in vertices: 
                self.adj[v] = []
        else: 
            assert isinstance(V, int)
            self.V = V
            for i in range(V): 
                self.adj[i] = []
    def DFS(x): 
        pass 
    def DFStrace(): 
        pass

def lcs_contiguous(s1, s2): 
    match = SequenceMatcher(None, s1, s2).find_longest_match(0, len(s1), 0, len(s2))
    s_sub = s1[match.a: match.a+match.size]
    assert s_sub == s2[match.b: match.b + match.size]
    return s_sub

def least_common(array, to_find=None):
    # import heapq 
    # from operator import itemgetter
    counter = collections.Counter(array)
    if to_find is None:
        return sorted(counter.items(), key=itemgetter(1), reverse=False)
    return heapq.nsmallest(to_find, counter.items(), key=itemgetter(1))

def tokenize(string):
    """Convert string to lowercase and split into words (ignoring
    punctuation), returning list of words.
    """
    # '\w+' does not work well for codes with special chars such as '.' as part of the 'word'
    return re.findall(r'([-0-9a-zA-Z_:.]+)', string.lower())  


def find_ngrams(input_list, n=3):
    """
    Example
    -------
    input_list = ['all', 'this', 'happened', 'more', 'or', 'less']

    """
    return zip(*[input_list[i:] for i in range(n)])

def count_given_ngrams(seqx, ngrams, partial_order=True):
    """
    Count numbers of occurrences of ngrams in input sequence (seqx, a list of a list of ngrams)

    Related
    -------
    count_given_ngrams2()

    Output
    ------
    A dictionary: n-gram -> count 
    """    

    # usu. the input ngrams have the same length 
    ngram_tb = {1: [], }
    for ngram in ngrams: # ngram is in tuple form 
        if isinstance(ngram, tuple): 
            length = len(ngram)
            if not ngram_tb.has_key(length): ngram_tb[length] = []
            ngram_tb[length].append(ngram)
        else: # assume to be unigrams 
            ngram_tb[1].append(ngram)
            
    ng_min, ng_max = min(ngram_tb.keys()), max(ngram_tb.keys())
    if partial_order:

        # evaluate all possible n-grams 
        counts = count_ngrams2(seqx, min_length=ng_min, max_length=ng_max, partial_order=True)

        counts_prime = {}
        for n, ngx in ngram_tb.items(): # n in n-gram
            if counts.has_key(n): 
                for ngram in ngx: # query each desired ngram 
                    # if n == 1: print '> unigram: %s' % ngram
                    # sorted('x') == sorted(('x', )) == ['x'] =>  f ngram is a unigram, can do ('e', ) or 'e'
                    counts_prime[ngram] = counts[n][tuple(sorted(ngram))] 
            else: 
                for ngram in ngx: 
                    counts_prime[ngram] = 0 
    else: 
        counts = count_ngrams2(seqx, min_length=ng_min, max_length=ng_max, partial_order=False)
        
        counts_prime = {}
        for n, ngx in ngram_tb.items(): # n in n-gram
            if counts.has_key(n): 
                for ngram in ngx: # query each desired ngram 
                    counts_prime[ngram] = counts[n][tuple(ngram)]
            else: 
                for ngram in ngx: 
                    counts_prime[ngram] = 0 

    return counts_prime  # n-gram -> count

def count_given_ngrams2(seqx, ngrams, partial_order=True):
    """
    Count numbers of occurrences of ngrams in input sequence (seqx, a list of a list of ngrams)

    Output
    ------
    A dictionary: n (as in ngram) -> ngram -> count 
    """
    # from batchpheno import utils

    # the input ngrams may or may not have the same length 
    ngram_tb = {1: [], }
    for ngram in ngrams: # ngram is in tuple form 
        if isinstance(ngram, tuple): 
            length = len(ngram)
            if not ngram_tb.has_key(length): ngram_tb[length] = []
            ngram_tb[length].append(ngram)
        else: # assume to be unigrams 
            assert isinstance(ngram, str)
            ngram_tb[1].append(ngram)
            
    # print('verify> ngram_tb:\n%s\n' % ngram_tb) # utils.sample_hashtable(ngram_tb, n_sample=10))

    ng_min, ng_max = min(ngram_tb.keys()), max(ngram_tb.keys())
    if partial_order:

        # evaluate all possible n-grams 
        counts = count_ngrams2(seqx, min_length=ng_min, max_length=ng_max, partial_order=True)

        counts_prime = {}
        for n, ngx in ngram_tb.items(): # n in n-gram
            if not counts_prime.has_key(n): counts_prime[n] = {} 
            if counts.has_key(n):
                for ngram in ngx: # query each desired ngram 
                    # if n == 1: print '> unigram: %s' % ngram
                    # sorted('x') == sorted(('x', )) == ['x'] =>  f ngram is a unigram, can do ('e', ) or 'e'
                    counts_prime[n][ngram] = counts[n][tuple(sorted(ngram))] 
            else: 
                for ngram in ngx: 
                    counts_prime[n][ngram] = 0 
    else: 
        counts = count_ngrams2(seqx, min_length=ng_min, max_length=ng_max, partial_order=False)
        
        counts_prime = {}
        for n, ngx in ngram_tb.items(): # n in n-gram
            if not counts_prime.has_key(n): counts_prime[n] = {} 
            if counts.has_key(n): 
                for ngram in ngx: # query each desired ngram 
                    # assert isinstance(ngram, tuple), "Ngram is not a tuple: %s" % str(ngram)
                    counts_prime[n][ngram] = counts[n][tuple(ngram)]
            else: 
                for ngram in ngx: 
                    counts_prime[n][ngram] = 0 

    return counts_prime  # n (as n-gram) -> counts (ngram -> count)

def count_ngrams2(lines, min_length=2, max_length=4, **kargs): 
    def eval_sequence_dtype(): 
        if not lines: 
            return False # no-op
        if isinstance(lines[0], str): # ['a b c d', 'e f', ]
            return False
        elif hasattr(lines[0], '__iter__'): # [['a', 'b'], ['c', 'd', 'e'], ]
            return True
        return False

    is_partial_order = kargs.get('partial_order', True)
    lengths = range(min_length, max_length + 1)    
    
    # is_tokenized = eval_sequence_dtype()
    seqx = []
    for line in lines: 
        if isinstance(line, str): # not tokenized  
            seqx.append([word for word in tokenize(line)])
        else: 
            seqx.append(line)
    
    # print('count_ngrams2> debug | seqx: %s' % seqx[:5]) # list of (list of codes)
    if not is_partial_order:  # i.e. total order 
        # ordering is important

        # this includes ngrams that CROSS line boundaries 
        # return count_ngrams(seqx, min_length=min_length, max_length=max_length) # n -> counter (of n-grams)

        # this counts ngrams in each line independently 
        counts = count_ngrams_per_seq(seqx, min_length=min_length, max_length=max_length) # n -> counter (of n-grams)
        return {length: counts[length] for length in lengths}

    # print('> seqx:\n%s\n' % seqx)
    # print('status> ordering NOT important ...')
    
    counts = {}
    for length in lengths: 
        counts[length] = collections.Counter()
        # ngrams = find_ngrams(seqx, n=length)  # list of n-grams in tuples
        if length == 1: 
            for seq in seqx: 
                counts[length].update([(ugram, ) for ugram in seq])
        else: 
            for seq in seqx:  # use sorted n-gram to standardize its entry since ordering is not important here
                counts[length].update( tuple(sorted(ngram)) for ngram in find_ngrams(seq, n=length) ) 

    return counts

def count_ngrams_per_line(**kargs):
    return count_ngrams_per_seq(**kargs)
def count_ngrams_per_seq(lines, min_length=1, max_length=4): # non boundary crossing  
    def update(ngrams):
        # print('> line = %s' % single_doc)
        for n, counts in ngrams.items(): 
            # print('  ++ ngrams_total: %s' % ngrams_total)
            # print('      +++ ngrams new: %s' % counts)
            ngrams_total[n].update(counts)
            # print('      +++ ngrams_total new: %s' % ngrams_total)

    lengths = range(min_length, max_length + 1)
    ngrams_total = {length: collections.Counter() for length in lengths}

    doc_boundary_crossing = False
    if not doc_boundary_crossing: # don't count n-grams that straddles two documents
        for line in lines: 
            nT = len(line)
            # print(' + line=%s, nT=%d' % (line, nT))
            single_doc = [line]

            # if the line length, nT, is smaller than max_length, will miscount
            ngrams = count_ngrams(single_doc, min_length=1, max_length=min(max_length, nT))
            update(ngrams) # update total counts
    else: 
        raise NotImplementedError

    return ngrams_total

def count_ngrams(lines, min_length=1, max_length=4): 
    """
    Iterate through given lines iterator (file object or list of
    lines) and return n-gram frequencies. The return value is a dict
    mapping the length of the n-gram to a collections.Counter
    object of n-gram tuple and number of times that n-gram occurred.
    Returned dict includes n-grams of length min_length to max_length.

    Use this only when (strict) ordering is important; otherwise, use count_ngrams2()

    Input
    -----
    lines: [['x', 'y', 'z'], ['y', 'x', 'z', 'u'], ... ]
    """
    def add_queue():
        # Helper function to add n-grams at start of current queue to dict
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length:  # count n-grams up to length those in queue
                ngrams[length][current[:length]] += 1  # ngrams[length] => counter
    def eval_sequence_dtype(): 
        if not lines: 
            return False # no-op
        if isinstance(lines[0], str): 
            return False
        elif hasattr(lines[0], '__iter__'): 
            return True
        return False

    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}
    queue = collections.deque(maxlen=max_length)

    # tokenized or not? 
    is_tokenized = eval_sequence_dtype()
    # print('> tokenized? %s' % is_tokenized)

    # Loop through all lines and words and add n-grams to dict
    if is_tokenized: 
        # print('input> lines: %s' % lines)
        for line in lines:
            for word in line:
                queue.append(word)
                if len(queue) >= max_length:
                    add_queue()  # this does the counting
            # print('+ line: %s\n+ngrams: %s' % (line, ngrams))
    else: 
        for line in lines:
            for word in tokenize(line):
                queue.append(word)
                if len(queue) >= max_length:
                    add_queue()

    # Make sure we get the n-grams at the tail end of the queue
    while len(queue) > min_length:
        queue.popleft()
        add_queue()
        # print('+ line: %s\n+ngrams: %s' % (line, ngrams))

    return ngrams

def check_boundary(lines, ngram_counts):
    # def isInDoc(ngstr): 
    #     for line in lines: 
    #         linestr = sep.join(str(e) for e in line)
    #         if linestr.find(ngstr) >= 0: 
    #             return True 
    #     return False

    # sep = ' ' 
    # for n, counts in ngram_counts: 
    #     counts_prime = []  # only keep those that do not cross line boundaries
    #     crossed = set()
    #     for ngr, cnt in counts: 

    #         # convert to string 
    #         ngstr = sep.join([str(e) for e in ngr])
    #         if isInDoc(ngstr): 
    #             counts_prime[]
    raise NotImplementedError
    # return ngram_counts  # new ngram counts


def print_most_frequent(ngrams, num=10):
    """Print num most common n-grams of each length in n-grams dict."""
    for n in sorted(ngrams):
        print('----- {} most common {}-grams -----'.format(num, n))
        for gram, count in ngrams[n].most_common(num):
            print('{0}: {1}'.format(' '.join(gram), count))
        print('')

def calc_cache_pos(strings, indexes):
    factor = 1
    pos = 0
    for s, i in zip(strings, indexes):  # iterate over each string
        pos += i * factor
        factor *= len(s)
    return pos

def lcs_back(strings, indexes, cache):
    if -1 in indexes:
        return ""
    match = all(strings[0][indexes[0]] == s[i]
                for s, i in zip(strings, indexes))
    if match:
        new_indexes = [i - 1 for i in indexes]
        result = lcs_back(strings, new_indexes, cache) + strings[0][indexes[0]]
    else:
        substrings = [""] * len(strings)
        for n in range(len(strings)):
            if indexes[n] > 0:
                new_indexes = indexes[:]
                new_indexes[n] -= 1
                cache_pos = calc_cache_pos(strings, new_indexes)
                if cache[cache_pos] is None:
                    substrings[n] = lcs_back(strings, new_indexes, cache)
                else:
                    substrings[n] = cache[cache_pos]
        result = max(substrings, key=len)
    cache[calc_cache_pos(strings, indexes)] = result
    return result

def lcs_back2(strings, indexes, cache):
    if -1 in indexes:
        return []
    match = all(strings[0][indexes[0]] == s[i]
                for s, i in zip(strings, indexes))
    if match:
        new_indexes = [i - 1 for i in indexes]
        result = lcs_back2(strings, new_indexes, cache)
        result.append(strings[0][indexes[0]])
    else:
        substrings = [[] for i in range(len(strings))] 
        for n in range(len(strings)):
            if indexes[n] > 0:
                new_indexes = indexes[:]
                new_indexes[n] -= 1
                cache_pos = calc_cache_pos(strings, new_indexes)
                if cache[cache_pos] is None:
                    substrings[n] = lcs_back2(strings, new_indexes, cache)
                else:
                    substrings[n] = cache[cache_pos]
        result = max(substrings, key=len)
    cache[calc_cache_pos(strings, indexes)] = result
    return result

def lcs(strings):
    """
    >>> lcs(['666222054263314443712', '5432127413542377777', '6664664565464057425'])
    '54442'
    >>> lcs(['abacbdab', 'bdcaba', 'cbacaa'])
    'baa'
    """
    import random
    isListOfTokens = False
    N = len(strings)
    if N >= 1: 
        sample_str = random.sample(strings, 1)[0]
        if isinstance(sample_str, list):
            isListOfTokens = True 
        else: 
            assert isinstance(sample_str, str)

    if len(strings) == 0:
        return [] if isListOfTokens else ""
    elif len(strings) == 1:
        return strings[0]
    else:
        cache_size = 1
        # result_seq = ""
        for s in strings:  # for each string 
            cache_size *= len(s)  # size(string) ~ size(list of tokens)
        cache = [None] * cache_size
        indexes = [len(s) - 1 for s in strings]

        if isListOfTokens: 
            return lcs_back2(strings, indexes, cache)
        else: 
            return lcs_back(strings, indexes, cache)
    return [] if isListOfTokens else ""

def demo_priority_queue(): 
    import platform
    import heapq

    try:
        import Queue as Q  # ver. < 3.0
    except ImportError:
        print("> import queue | python version %d" % platform.python_version())
        import queue as Q

    q = Q.PriorityQueue()
    q.put((10,'ten'))
    q.put((1,'one'))
    q.put((5,'five'))
    while not q.empty():
        print(q.get(),)

    print('info> try heapq module ...')
    
    heap = []
    heapq.heappush(heap, (-1.5, 'negative one'))
    heapq.heappush(heap, (1, 'one'))
    heapq.heappush(heap, (10, 'ten'))
    heapq.heappush(heap, (5.7,'five'))
    heapq.heappush(heap, (100.6, 'hundred'))

    for x in heap:
        print(x,)
    print

    heapq.heappop(heap)

    for x in heap:
        print(x,)
    print()

    # the smallest
    print('info> smallest: %s' % str(heap[0]))

    smallestx = heapq.nsmallest(2, heap)  # a list
    print('info> n smallest: %s, type: %s' % (str(smallestx), type(smallestx)))

    return

def demo_preprocessing(): 
    line = '496 492.8 496 492.8 496 492.8 496 CDC:123459000 MED:7015 MULTUM:127 unknown poison-drug WRONGCODE:172.5'
    tokens = tokenize(line)
    print('string: %s' % line)
    print('tokens: %s' % tokens)
 
    return

def demo_count_ngrams(): 
    from itertools import chain 
    lines = [['a', 'x', 'y', 'z'], ['x', 'y', 'z'], ['z', 'y', 'x', 'u'], ['x', 'y'], ['z', 'y', 'u', 'x'], ['x', 'a', 'x', 'y', 'b']]
    ngrams = count_ngrams(lines, min_length=1, max_length=5)
    print('> ngrams frequency:\n%s\n' % ngrams)

    ngrams = count_ngrams2(lines, min_length=1, max_length=5, partial_order=True)
    print('> ngrams frequency (unordered):\n%s\n' % ngrams)


    tokens = list(chain.from_iterable(lines))
    print('> tokens:\n%s\n' % tokens)

    seq = ['A', 'C', 'G', 'T', 'A', 'C', 'A', 'T', 'C', 'G', 'C', 'T']
    n = 3
    print('> sequence of %d-gram:\n%s\n' % (n, find_ngrams(seq, n=n)))

    ngrams = [('x', 'y'), ('u', 'x'), ('u', 'v'), ('x', 'c'), ('a', 'x'), 'x', ('a', 'x', 'x', 'y'), ('u', 'x', 'y', 'z'), ('x', 'y', 'z', 'u'), ('z', 'y', 'x'), ]
    counts = count_given_ngrams2(lines, ngrams, partial_order=True)
    print(counts)

    counts = count_given_ngrams2(lines, ngrams, partial_order=False)    
    print(counts)

    # test the summing of frequencies
    seqx1 = [['a', 'x', 'y', 'z'], ['x', 'y', 'z'], ['z', 'y', 'x', 'u']]
    seqx2 = [['x', 'y'], ['z', 'y', 'u', 'x'], ['y', 'y'], ['x', 'a', 'x', 'y', 'b']]
    ngrams = [('x', 'y'), ('u', 'v'), 'x', ('a', 'x', 'x', 'y'), ('u', 'x', 'y', 'z'), ('z', 'y', 'x'), ]
    counts1 = count_given_ngrams2(seqx1, ngrams, partial_order=True)
    counts2 = count_given_ngrams2(seqx2, ngrams, partial_order=True)

    print('> counts1: %s' % counts1)
    print('> counts2: %s' % counts2)

    print("\n")

    seqx1 = [ ['a', 'x', 'y', 'z', ], ['x', 'y', 'z'], ['x', 'y', 'z'], ['x', 'y', 'z'], ['z', 'y', 'x', 'u'], ['z', 'x'], ['y', 'y'], ] 
    counts1 = count_ngrams2(seqx1, min_length=1, max_length=10, partial_order=False)
    print(counts1)
    # [log]
    # partial ordering or ordering not important 
    # bigrams: {('x', 'y'): 3, ('y', 'z'): 3, ('u', 'x'): 1, ('a', 'x'): 1, ('y', 'y'): 1, ('x', 'z'): 1}
    # 4-grams: {('u', 'x', 'y', 'z'): 1, ('a', 'x', 'y', 'z'): 1}}
    # crossing boundary? no

    # strict ordering 
    
    # use count_ngram()
    # 4-grams {('y', 'z', 'z', 'y'): 1, ('x', 'y', 'z', 'z'): 1, ('y', 'z', 'x', 'y'): 1, ('z', 'x', 'y', 'y'): 1, ('a', 'x', 'y', 'z'): 1, ('x', 'u', 'z', 'x'): 1, ('z', 'x', 'y', 'z'): 1, ('z', 'z', 'y', 'x'): 1, ('u', 'z', 'x', 'y'): 1, ('y', 'x', 'u', 'z'): 1, ('x', 'y', 'z', 'x'): 1, ('z', 'y', 'x', 'u'): 1})
    # crossing boundary? yes

    # use count_ngram_per_seq()  
    # bigrams: {('x', 'y'): 2, ('y', 'z'): 2, ('a', 'x'): 1, ('z', 'x'): 1, ('y', 'x'): 1, ('z', 'y'): 1, ('y', 'y'): 1, ('x', 'u'): 1}
    # 4-grams: {('a', 'x', 'y', 'z'): 1, ('z', 'y', 'x', 'u'): 1})

    # {('u', 'x', 'y', 'z'): 1, ('a', 'x', 'y', 'z'): 1}
    return


def demo_count_ngrams2(): 
    seqx1 = [ ['a', 'x', 'y', 'z'],  ] #  ['x', 'y', 'z'], ['z', 'y', 'x', 'u'], ['z', 'x'], ['y', 'y']
    counts1 = count_ngrams2(seqx1, min_length=1, max_length=3, partial_order=False)
    print(counts1)

    # [log]
    # +ngrams: {1: Counter({('a',): 1, ('x',): 1}), 2: Counter({('a', 'x'): 1, ('x', 'y'): 1}), 3: Counter({('x', 'y', 'z'): 1, ('a', 'x', 'y'): 1})}
    # +ngrams: {1: Counter({('y',): 1, ('a',): 1, ('x',): 1}), 2: Counter({('a', 'x'): 1, ('x', 'y'): 1, ('y', 'z'): 1}), 3: Counter({('x', 'y', 'z'): 1, ('a', 'x', 'y'): 1})}
    # +ngrams: {1: Counter({('y',): 1, ('a',): 1, ('z',): 1, ('x',): 1}), 2: Counter({('a', 'x'): 1, ('x', 'y'): 1, ('y', 'z'): 1}), 3: Counter({('x', 'y', 'z'): 1, ('a', 'x', 'y'): 1})}
    # ~> {1: Counter({('y',): 1, ('z',): 1, ('x',): 1, ('a',): 1}), 2: Counter({('a', 'x'): 1, ('x', 'y'): 1, ('y', 'z'): 1}), 3: Counter({('x', 'y', 'z'): 1, ('a', 'x', 'y'): 1})}

    return

def demo_lcs(): 
    def to_list_repr(strings):
        slx = []
        for string in strings: 
            slx.append([e for e in string]) 
        return slx
    def to_str(lists): 
        strl = []
        for tokens in lists: 
            strl.append(' '.join(tokens))
        return strl
    from utils_sys import highlight

    # ['123.5', '374.7', 'J23'] is a subseq of ['123.5', 'X27.1', '374.7', '334.7', '111', 'J23', '223.4']? True
    q1 = ['123.5', '374.7', 'J23']
    r1 = ['123.5', 'X27.1', '374.7', '334.7', '111', 'J23', '223.4']
    r2 = ['123.5', 'y', 'z', 'X27.1', 'y', '374.7', 'z', 'z', '334.7', '111', 'y', 'J23', 'y', 'z', 'y', 'x', '223.4']
    r3 = ['y', 'z', '374.7', 'x', 'x', '374.7', 'x', '334.7', 'J23']  # missing 123.5

    ### Inputs are strings
    q = ['666222054263314443712', '5432127413542377777', '6664664565464057425']
    q = [q1, r1, r2, r3]
    # q2 = to_list_repr(q) # doesn't work 
    # q = to_str(q)
    s = lcs(q)

    print('  + %s ~>\n  %s\n' % (q, s))

    highlight("(demo) Now let's consider contiguous LCS ...")

    q = "HEPATITIS C GENO TYPE 1 NS3 RESIST"
    r = "HEPATITIS C VIRAL RNA GENOTYPE 1 NS"
    s = lcs_contiguous(q, r)
    print("... query: {}\n... target: {}\n=>\n... {}".format(q, r, s))

    return

def test():

    ### preprocessing documents, texts
    # demo_preprocessing()

    ### enumerate all possible n-grams 
    # demo_count_ngrams()

    ### subsequences
    demo_lcs()

    return

if __name__ == "__main__": 
    test()