import inspect, sys, os, subprocess # commands
# In Python3 commands -> subprocess

import random
import numpy as np

class autovivify_list(dict):
    """Pickleable class to replicate the functionality of collections.defaultdict"""
    def __missing__(self, key):
        value = self[key] = []
        return value

    def __add__(self, x):
        '''Override addition for numeric types when self is empty'''
        if not self and isinstance(x, Number):
            return x
        raise ValueError

    def __sub__(self, x):
        '''Also provide subtraction method'''
        if not self and isinstance(x, Number):
            return -1 * x
        raise ValueError

def search(file_patterns=None, basedir=None): 
    """

    Memo
    ----
    1. graphic patterns 
       ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']

    """
    import fnmatch

    if file_patterns is None: 
        file_patterns = ['*.dat', '*.csv', ]
    if basedir is None: 
        basedir = os.getcwd()

    matches = []

    for root, dirnames, filenames in os.walk(basedir):
        for extensions in file_patterns:
            for filename in fnmatch.filter(filenames, extensions):
                matches.append(os.path.join(root, filename))
    return matches

### Formatting tools 

def indent(message, nfill=6, char=' ', mode='r'): 
    if mode.startswith('r'): # left padding 
        return message.rjust(len(message)+nfill, char)
    return message.ljust(len(message)+nfill, char)

def div(message=None, symbol='=', prefix=None, n=80, adaptive=False, border=0, offset=5, stdout=True): 
    output = ''
    if border is not None: output = '\n' * border
    # output += symbol * n + '\n'
    if isinstance(message, str) and len(message) > 0:
        if prefix: 
            line = '%s: %s\n' % (prefix, message)
        else: 
            line = '%s\n' % message
        if adaptive: n = len(line)+offset 

        output += symbol * n + '\n' + line + symbol * n
    elif message is not None: 
        # message is an unknown object of some class
        if prefix: 
            line = '%s: %s\n' % (prefix, str(message))
        else: 
            line = '%s\n' % str(message)
        if adaptive: n = len(line)+offset 
        output += symbol * n + '\n' + line + symbol * n
    else: 
        output += symbol * n
        
    if border is not None: 
        output += '\n' * border
    if stdout: print(output)
    return output

def dictToList(adict):
    lists = []
    for k, v in nested_dict_iter(adict): 
        alist = []
        if not hasattr(k, '__iter__'): k = [k, ]
        if not hasattr(v, '__iter__'): v = [v, ]
        alist.extend(k)
        alist.extend(v)
        lists.append(alist)
    return lists

def nested_dict_iter(nested):
    import collections

    for key, value in nested.iteritems():
        if isinstance(value, collections.Mapping):
            for inner_key, inner_value in nested_dict_iter(value):
                yield inner_key, inner_value
        else:
            yield key, value

def dictSize(adict): # yeah, size matters  
    return len(list(nested_dict_iter(adict)))
def size_dict(adict): 
    """

    Note
    ----
    1. size_hashtable()
    """
    return len(list(nested_dict_iter(adict)))


def partition(lst, n):
    """
    Partition a list into almost equal intervals as much as possible. 
    """
    q, r = divmod(len(lst), n)
    indices = [q*i + min(i, r) for i in xrange(n+1)]
    return [lst[indices[i]:indices[i+1]] for i in xrange(n)]

def divide_interval(total, n_parts):
    pl = [0] * n_parts
    for i in range(n_parts): 
        pl[i] = total // n_parts    # integer division

    # divide up the remainder
    r = total % n_parts
    for j in range(r): 
        pl[j] += 1

    return pl 