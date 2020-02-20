# encoding: utf-8

import os, sys, random, collections
import numpy as np
import logging

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.

import matplotlib.pyplot as plt

# select plotting style 
plt.style.use('seaborn')  # values: {'seaborn', 'ggplot', }

from sklearn.metrics import r2_score
from utils_plot import saveFig

# ElasticNet
from sklearn.linear_model import ElasticNet

# model selection
from sklearn.model_selection import cross_val_score
from itertools import product
# import operator

import pandas as pd 
from pandas import DataFrame, Series

import common
from utils_sys import div

import config

class Metrics(object): 

    tracked = ['auc', 'fmax', 'fmax_negative', 'sensitivity', 'specificity', 'brier', ]

    def __init__(self, records={}, op=np.mean): 

        # self.records is nomrally a map: metric -> measurements ...
        # ... but in the case of PerformanceMetrics (derived class), this can be used to hold description (when merging multiple instances of PerformanceMetrics)
        self.records = {}
        if len(records) > 0: 
            self.add(records)
        self.op = op   # combiner on bags

    def add(self, item): 
        if isinstance(item, dict): 
            for k, v in item.items(): 
                if not k in self.records: self.records[k] = []
                if hasattr(v, '__iter__'): 
                    self.records[k].extend(list(v))  # this feature is usually used by PerformaceMetrics.add_doc()
                else: 
                    self.records[k].append(v)
        elif hasattr(item, '__iter__'):
            if len(item) == 0: 
                return # do nothing 
            elif len(item) == 1: 
                self.add_value(name=item)
            else: 
                self.add_value(name=item[0], value=item[1])
        else: 
            print('Warning: dubious input: %s' % str(item))
            self.add_value(name=item)         
        return
    def size(self):
        return len(self.records) 
    def size_bags(self):
        return sum(len(v) for v in self.records.values())  

    def clone(self): 
        import copy
        m = Metrics()
        m.records = copy.deepcopy(self.records)
        m.op = self.op 

        return m

    def add_value(self, name, value=None):
        if not name in self.records: 
            self.records[name] = []
        if value is not None: 
            self.records[name].append(value)
        else: 
            pass # do nothing
        return # no return value 

    def do(self, op=None, min_freq=0):  # perform an operation on the bags but does not change the internal represnetation
        if op is not None: self.op = op
        
        if hasattr(self.op, '__call__'): 
            mx = {}
            for k, v in self.records.items():
                # prune the keys with insufficient data points 
                if len(v) >= min_freq:  
                    mx[k] = self.op(v) 
        else: 
            assert isinstance(self.op, str), "Invalid operator: %s" % self.op
            if op.startswith('freq'):
                self.op = len
                mx = {}
                for k, v in self.records.items(): 
                    if len(v) >= min_freq: 
                        mx[k] = self.op(v)
            else: 
                raise NotImplementedError
        return mx
    def apply(self, op=None, min_freq=0):
        return self.do(op=op, min_freq=min_freq) 
    def aggregate(self, op=None, min_freq=0, by=''):
        if len(by) > 0: 
            if by.startswith('freq'): 
                return self.do(op=len, min_freq=min_freq)
            elif by == 'mean': 
                return self.do(op=np.mean, min_freq=min_freq)
        return self.do(op=op, min_freq=min_freq) # precedence op, self.op

    def display(self, by='freq', reverse=True, op=None, formatstr=''): 
        if by.startswith(('freq', 'agg')): 
            records = self.sort(by, reverse=reverse, op=op)
        else: 
            records = list(self.records.items())
        
        if formatstr: 
            for k, bag in records:
                try:  
                    print(formatstr.format(k, bag))
                except: 
                    print(formatstr.format(key=k, value=bag))
        else: 
            # default
            for k, bag in records: 
                print('[%s] %s' % (k, bag))
        return

    def sort_by_freq(self, reverse=True):
        # set op to len
        v = next(iter(self.records.values())) 
        # reduced? 
        if hasattr(v, '__iter__'): 
            return sorted( [(key, len(bag)) for key, bag in self.records.items()], key=lambda x: x[1], reverse=reverse) 
        return list(self.records.keys())  
    def sort_by_aggregate(self, reverse=True, op=None, min_freq=0):
        import operator 
        sorted_bags = self.aggregate(op=op, min_freq=min_freq)
        return sorted(sorted_bags.items(), key=operator.itemgetter(1, 0), reverse=reverse)  # sort by values first and then sort by keys 
    def sort(self, by='aggregate', reverse=True, op=None, min_freq=0):
        if by.startswith('agg'):
            return self.sort_by_aggregate(op=op, reverse=reverse, min_freq=min_freq) 
        elif by.startswith('freq'):  
            # choosing this will ignore what the default aggregate function is
            return self.sort_by_freq(reverse=reverse) 
        elif by == 'mean': 
            return self.sort_by_aggregate(op=np.mean, reverse=reverse, min_freq=min_freq)
        else: 
            raise ValueError("Metrics.sort(), invalid sort mode: {by}".format(by=by))
        # return self.sort_by_aggregate(op=op, reverse=reverse, min_freq=min_freq)

    def is_uniform(self): 
        # check if the meta data has a uniform length
        if not self.records: return True

        values = next(iter(self.records.values())) # python 2: self.records.itervalues().next() # python3: next(iter(self.records.values()))
        n_values = len(values)

        tval = True
        for k, vals in self.records.items(): 
            if n_values != len(vals): 
                tval = False
                break
        return tval

    # to be overridden by derived class
    def report(self, op=np.mean, message='', order='desc', tracked_only=True): 
        if op is not None: self.op = op
        assert hasattr(self.op, '__call__')

        title_msg = 'Performance metrics (aggregate method: %s)' % self.op.__name__
        div(message=title_msg, symbol='#', border=1)

        rank = 0; tracked_metrics = []
        if not message: 
            for i, metric in enumerate(Metrics.tracked): 
                val = self.op(self.records[metric]) # if hasattr(adict[metric]) > 1 else adict[metric]
                tracked_metrics.append((metric, val))
                # if tracked_only and not metric in Metrics.tracked: continue  
                rank += 1 
                print('... [%d] %s: %s' % (rank, metric, val))
        else: 
            message = '(*) %s\n' % message
            for i, metric in enumerate(Metrics.tracked): 
                val = self.op(self.records[metric]) # if hasattr(adict[metric]) > 1 else adict[metric]
                tracked_metrics.append((metric, val))
                # if tracked_only and not metric in Metrics.tracked: continue 
                rank += 1 
                message += '... [%d] %s: %s\n' % (rank, metric, val)

            # which performance metric(s) has the most advantange? 
            # metrics_sorted = sorted([(k, v) for k, v in self.do().items()], key=lambda x:x[1], 
            #                             reverse=True if order.startswith('desc') else False)  # best first
            metrics_sorted = sorted(tracked_metrics, key=lambda x: x[1], reverse=True if order.startswith('desc') else False)
            message += '... metrics order: %s\n'  % ' > '.join([m for m, _ in metrics_sorted])
            div(message=message, symbol='*', border=2)


        return

    def save(self, columns=[]): # save performance records 
        pass

    def my_shortname(self, context='suite', size=-1, domain='test', meta=''): 
        # domain: the name of the dataset or project 
        # context: the context in which the performance metrics was derived
        # meta: other parameters
        if size == -1: size = self.n_methods()
        
        name = 'performance_metrics-{context}-N{size}-D{domain}'.format(context=context, size=size, domain=domain)
        if meta: 
            name = '{prefix}-M{meta}'.format(prefix=name, meta=meta)
        return name

    @staticmethod
    def my_shortname(context, size=-1, domain='test', meta=''): 
        # domain: the name of the dataset or project 
        # context: the context in which the performance metrics was derived
        # meta: other parameters
        if size != -1:
            name = 'performance_metrics-{context}-N{size}-D{domain}'.format(context=context, size=size, domain=domain)
        else: 
            name = 'performance_metrics-{context}-D{domain}'.format(context=context, domain=domain)
        if meta: 
            name = '{prefix}-M{meta}'.format(prefix=name, meta=meta)
        return name 

    @staticmethod
    def plot_path(name='test', basedir=None, ext='tif', create_dir=True):
        # create the desired path to the plot by its name
        if basedir is None: basedir = PerformanceMetrics.plot_dir
        if not os.path.exists(basedir) and create_dir:
            print('(plot) Creating plot directory:\n%s\n' % basedir)
            os.mkdir(basedir) 
        return os.path.join(basedir, '%s.%s' % (name, ext))

### end class Metrics

# refactor -> evaluate
class PerformanceMetrics(Metrics): # or PerformanceComparison
    """

    Memo
    ----
    create a dataframe/table, where cols: methods, rows: metrics   
        ret = {method: [] for method in methods}
    """

    # tracked = ['auc', 'fmax', 'fmax_negative', 'sensitivity', 'specificity', ]  # Metrics.tracked
    prefix = config.project_path # os.getcwd()
    data_dir = os.path.join(prefix, 'data')  # output directory for the PerformanceMetrics object
    log_dir = os.path.join(prefix, 'log')      
    plot_dir = os.path.join(prefix, 'plot')
    analysis_dir = os.path.join(prefix, 'analysis') 

    def __init__(self, scores=None, method=None, **kargs):
        # other params: description={}, overwrite_=True, op=np.mean, load=False, file_name=''

        ### @ base level
        # self.records  # ... re-purposed to hold descriptions
        # self.op     # ... a combiner 
        # Metrics.__init__(self, op=op)
        super(PerformanceMetrics, self).__init__(op=kargs.get('op', np.mean))
        self.table = DataFrame() # rows: metrics, cols: methods (e.g. a BP method such as SVM, ensemble learning method such as stacking, etc.)
         
        ## add meta data
        # super(PerformanceMetrics, self).add(description)
        self.add_doc(kargs.get('meta', {}))   # self.records, self.op

        # cross validation 
        self.cv = []  # add (y_true, y_score) across CV fold

        # I/O 
        self.data_dir = kargs.get('data_dir', PerformanceMetrics.data_dir) 
        # assert os.path.exists(self.data_dir)

        self.delimit = '|'

        if scores is not None:  
            # add data (method by method)
            if isinstance(scores, dict): 
                assert isinstance(method, str)
                self.table[method] = [scores[metric] for metric in PerformanceMetrics.tracked]
                self.table.index = PerformanceMetrics.tracked
            elif isinstance(scores, list): 
                assert isinstance(method, str)
                assert len(scores) == len(PerformanceMetrics.tracked)
                self.table.index = PerformanceMetrics.tracked
                self.table[method] = scores 
            elif isinstance(scores, PerformanceMetrics): # scores can also be a PerformanceMetrics object

                # check index
                # assert scores.table.shape[0] == len(PerformanceMetrics.tracked)
                # assert all(scores.table.index == PerformanceMetrics.tracked), "Inconsistent metric order:\n%s vs %s\n" % (scores.table.index.values, PerformanceMetrics.tracked)
           
                self.copy(scores, overwrite_=overwrite_)  # copy constructor
            else: 
                assert isinstance(scores, DataFrame)
                # [todo]
        else: 
            # self.table[method] = [0.] * len(PerformanceMetrics.tracked)

            # load from file
            if kargs.get('load', False):
                self.load(file_name=self.my_shortname(), throw_=kargs.get('throw_', False)) # throw exception if_file_not_found
            else: 
                # cannot set index at this point
                if hasattr(method, '__iter__'):   # this is only used when we need to add metric one by one (e.g. taking average performance score across CV folds)
                    self.table = DataFrame(columns=method)

    @classmethod
    def set_path(cls, prefix, basedirs=['log', 'data', 'plot', 'analysis', ], create_dir=True):
        for basedir in basedirs:
            attr = '{base}_dir'.format(base=basedir)

            path = os.path.join(prefix, basedir)
            setattr(cls, attr, path)
            print('(PerformanceMetrics.set_path) attr %s -> %s | check: %s' % (attr, os.path.join(prefix, basedir), getattr(PerformanceMetrics, attr)))

            if not os.path.exists(path) and create_dir: 
                print('(PerformanceMetrics.set_path) Creating %s directory:\n%s\n' % (basedir, path))
                os.mkdir(path) 

        return 
            
    def copy(self, p_object, overwrite_=True):
        # import copy

        # self.table = DataFrame()
        # self.table = p_object.table.copy(deep=True)
        for method in p_object.table.columns: 
            if overwrite_ or (not method in self.table.columns): # add only if not existed
                self.table[method] = p_object[method] # if not incr_update, then update no matter what  
        self.table.index = p_object.table.index

        self.records.update(p_object.records)
        self.op = p_object.op

        # [todo] 
        #  1. check index and only add those metrics in PerformanceMetrics.tracked
        #  2. base level copy constructor

        return  
    def clone(self, except_table=True):
        perf = PerformanceMetrics() 

        if not except_table: 
            perf.table = self.table.copy()
        
        # base
        perf.records.update(self.records)
        perf.op = self.op

        return perf

    def isEmpty(self):
        return self.table.empty 

    def add_doc(self, records):
        # records is a "bag" not just a dictionary
        # use this method only when new key and values are to be incrementally addded to the existing records rather than overwriting them 

        super(PerformanceMetrics, self).add(records) 
        # for k, v in records.items(): 
        #     if not k in self.records: self.records[k] = []
        #     if isinstance(v, list): 
        #         self.records[k].extend(v)
        #     else: 
        #         self.records[k].append(v)
        return

    def query(self, key='domain', single_value=True):
        val = self.records.get(key, ['null', ])
        
        if single_value:  # self.records is a multibag
            val = val[0] 
        return val 

    def _write(self, scores, method):
        tAddIndex = self.table.empty
        if isinstance(scores, dict): 
            
            self.table[method] = [scores.get(metric, 0) for metric in PerformanceMetrics.tracked]
            if tAddIndex: 
                self.table.index = PerformanceMetrics.tracked
        elif isinstance(scores, Series):
            assert all(Series.index == PerformanceMetrics.tracked)
            self.table[method] = scores   # should add the index from Series if empty; should align with the index if not empty
        elif isinstance(scores, list): 
            assert len(scores) == len(PerformanceMetrics.tracked)
            self.table[method] = scores 
            if tAddIndex: 
                self.table.index = PerformanceMetrics.tracked

    # add column
    def add(self, scores, method, overwrite_=True): 
        """
        Add the performance scores of a method to the entry of the hashtable (self.table), where key: method, value is the scores 
            scores can be: 
               i) dictionary 
               ii) list 

        """
        if overwrite_: 
            self._write(scores, method)
        else: 
            if not method in self.table.columns: 
                self._write(scores, method)
            else: 
                raise ValueError("Method %s already exist!" % method)
    def add_and_eval(self, labels, predictions, method):
        metrics = scores(labels, predictions, metrics=PerformanceMetrics.tracked)
        self.add(metrics, method=method, overwrite_=True) 
        return

    # add row? 
    def add_metric(self, metric, values, overwrite_=True):
        # add values associated with a particular perf. metric for all algorithms/columns
        # this is possible only when columns are already known 

        assert len(self.table.columns) == len(values), "Got %d values but have %d columns: %s" % (len(values), len(self.table.columns), self.methods_as_str())
        
        # values can be i) a dictionary (keys are to be matched with columns) ii) Series (indices are to be matched with columns), iii) a list
        if overwrite_ or (not metric in self.table.index): 
            self.table.loc[metric] = values

        return

    def get(self, metric=None, method=None): 
        if metric is not None and method is not None: 
            return self.table.loc[metric][method]  # a number/scalar
        elif metric is not None: 
            return self.table.loc[metric] # Series
        elif method is not None: 
            return self.table[method]  # Series
        raise ValueError

    def sort(self, metric, reverse=True, verbose=True, sorted_pairs=True):  
        # print('(sort) table index:\n%s\n' % self.table.index.values)

        # sort methods according to the given metric 
        mdict = dict(self.table.loc[metric])  # cols: keys, rows: scores
        # print('(sort) mdict:\n%s\n' % mdict)
        methods = sorted(mdict, key=mdict.__getitem__, reverse=reverse)
        
        if verbose: 
            msg = "\nMetric: %s\n------" % metric
            msg += "%s (%s=%f)" % (methods[0], metric, mdict[methods[0]])
            for i, method in enumerate(methods[1:]):  # tip: methods[1:] will not throw an exception even when methods is empty
                # if i == 0: continue 
                msg += " >= %s (%f)" % (methods[i], mdict[methods[i]])
            msg += '\n'
            print( msg ) 
            # print( methods )
        if sorted_pairs: 
            return [(method, mdict[method]) for method in methods]
        return methods

    def unbag(self, bag_count=None, sep='.', exception_=False):

        # use utility function defined below
        table = self.table
        self.table = unbag(table, bag_count=bag_count, sep=sep, exception_=exception_) 
        return

    def plot(self): 
        """
        Performance comparison of methods held in the table based on the metrics observed. 

        Design
        ------
            1. (grouped) bar plots, horizontal bars preferred 

            2. 

        Related
        -------
            plot_roc

        """
        pass

    def plot_roc(self, file_name='roc'):
        assert len(self.cv) > 0, "No CV data recorded!"
        # but this is expensive for objects to keep
     
        # call friend function 
        return plot_roc(self.cv, file_name=file_name) 

    def save(self, file_name='performance_metrics'):
        if not os.path.exists(self.data_dir): 
            print('(save) Creating data directory:\n%s\n' % self.data_dir)
            os.mkdir(self.data_dir)

        file_stem, file_ext = os.path.splitext(file_name)
        if not file_ext: file_name = '%s.csv' % file_name

        if not self.table.empty:
            fpath = os.path.join(self.data_dir, file_name)
            print('(save) Saving performance metrics to .csv: %s ...' % fpath)
            self.table.to_csv(fpath, sep=self.delimit, index=True, header=True) 
        return
    def load(self, file_name='performance_metrics', throw_=False):
        import pandas as pd

        file_stem, file_ext = os.path.splitext(file_name)
        if not file_ext: file_name = '%s.csv' % file_name
        
        fpath = os.path.join(self.data_dir, file_name)
        st = 0
        if throw_: 
            assert os.path.exists(fpath)
        else: 
            st = 1
        self.table = pd.read_csv(fpath, sep=self.delimit, header=0, index_col=0, error_bad_lines=True)

        # check index 

        # load description 
        # self.records = pickle.load() # or another dataframe  # [todo]

        if self.table.empty: 
            st = 2 

        if st > 0: 
            msg = '(load) Failed to load from:\n%s\n' % fpath 
            if st > 1: 
                msg += '... Could not find prior records of performance metrics ...\n'
            print(msg)
        return

    def divide(self, n_parts=2, metric='fmax', verbose=False): 
        from algorithms import split 
        # sometimes the table is too big for graphic rendering
        # n_parts=2 => divide into two tables

        if self.n_methods() <= n_parts:
            # do nothing 
            return 
        
        tables = []
        method_scores = np.array(self.sort(metric, reverse=True, verbose=verbose))

        # [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]
        for i, part in enumerate(list(split(range(self.n_methods()), n_parts))): 
            lower, upper = part[0], part[-1]
            subset = method_scores[lower:upper]
            tables.append(self.table[[method for method, _ in subset]])
      
        ## if we know the size of each partitions, then use the following block
        # tables = []
        # lower, upper = 0, n_methods
        # while True: 
        #     subset = method_scores[lower:upper]
        #     tables.append(self.table[[method for method, _ in subset]])

        #     upper+= n_methods
        #     lower+= n_methods

        #     if lower >= self.n_methods(): break 

        perfx = []
        for table in tables: 
            p_new = self.clone(except_table=True)
            p_new.table = table
            perfx.append(p_new)

        return perfx

    def n_methods(self): 
        return self.table.shape[1]
    def n_metrics(self):
        return self.table.shape[0] 

    def methods_as_str(self):
        return ' '.join([str(col) for col in self.table.columns]) 
    def get_methods(self):
        return self.table.columns # ' '.join([str(col) for col self.table.columns])
    def get_metrics(self): 
        return self.table.index 
    def set_metrics(self, metrics):  # if we only need a subset of the tracked metric
        # allowed = []
        self.table.index = [metric for metric in metrics if metric in PerformanceMetrics.tracked]
        return
    metrics = property(get_metrics, set_metrics) 

    @staticmethod
    def sort2(p_obj, metric, reverse=True, verbose=True, sorted_pairs=False): 

        # empty PerformanceMetrics object, no op 
        if p_obj is None or p_obj.isEmpty(): 
            # no-op 
            print('(sort2) No-op.')
            return []

        # sort methods according to the given metric 
        if isinstance(p_obj, PerformanceMetrics): 
            table = p_obj.table
        else: 
            assert isinstance(p_obj, DataFrame)
            table = p_obj
        
        # print('(sort2) table index:\n%s\n' % table.index)
        mdict = dict(table.loc[metric])  # cols: keys, rows: scores
        methods = sorted(mdict, key=mdict.__getitem__, reverse=reverse)
        
        if verbose: 
            msg = "\nMetric: %s\n------" % metric
            msg = "%s (%s=%f)" % (methods[0], metric, mdict[methods[0]])
            for i, method in enumerate(methods[1:]):
                # if i == 0: continue 
                msg += " >= %s (%f)" % (methods[i], mdict[methods[i]])
            msg += '\n'
            print( msg ) 

        if sorted_pairs: 
            return [(method, mdict[method]) for method in methods]
        return methods 

    @staticmethod
    def getTopK(perf_metrics, metric, k=10, reverse=True, verbose=True):
        # design: always return a consolidated and sorted PerformanceMatrics object

        if isinstance(perf_metrics, list): 
            if not perf_metrics: return PerformanceMetrics()
            
            perf_all = PerformanceMetrics.merge(perf_metrics)
        else: 
            perf_all = perf_metrics 

        methods = perf_all.sort(metric, reverse=reverse, verbose=verbose, sorted_pairs=False)

        p_new = PerformanceMetrics() # DataFrame()
        p_new.table = perf_all.table[methods[:k]]
        p_new.table.index = perf_all.table.index
        p_new.records.update(perf_all.records)  # [todo]
        
        return p_new

    @staticmethod
    def summarize(perf_metrics, metric, reverse=True, verbose=True, keywords=[]): 
        import utils_sys as us

        attributes = ['max', 'min', 'mean', 'median', ]
        ret = {a: -1 for a in attributes}

        if isinstance(perf_metrics, list): 
            if not perf_metrics: return PerformanceMetrics()
            
            perf_all = PerformanceMetrics.merge(perf_metrics)
        else: 
            perf_all = perf_metrics 

        # select only a subset of methods
        table = perf_all.table  # dict(self.table.loc[metric])

        selected = list(table.columns)
        if keywords: 
            selected = []
            for col in table.columns:
                matched = False 
                for k in keywords:
                    if col.find(k) >= 0: 
                        matched = True
                    else: 
                        matched = False; break 
                if matched: selected.append(col)
            assert len(selected) > 0, "(summarize) No qualified methods found given keywords: %s | methods:\n... %s\n" % (keywords, table.columns.values)
        
        print('(summarize) Considering the following methods:\n{list}\n'.format(list=us.format_list(selected, mode='v', padding=6)))
        # method_score_pairs = perf_all.sort(metric, reverse=reverse, verbose=verbose, sorted_pairs=True)
        if metric is None: 
            ret['methods'] = selected

            ### consider all metrics
            ret['metrics'] = list(table.index)
            ret['scores'] = table[selected].mean(axis=1)
            ret['mean'] = table[selected].mean(axis=1)
            ret['median'] = table[selected].median(axis=1)
            ret['max'] = table[selected].max(axis=1)
            ret['min'] = table[selected].min(axis=1)
        else: 
            ret['methods'] = selected = [m for m in perf_all.sort(metric, reverse=reverse, verbose=verbose, sorted_pairs=False) if m in selected]
            ret['metrics'] = [metric, ]
            ret['scores'] = list(table[selected].loc[metric]) 
            ret['mean'] = table[selected].loc[metric].mean()
            ret['median'] = table[selected].loc[metric].median()
            ret['max'] = table[selected].loc[metric].max()
            ret['min'] = table[selected].loc[metric].min()
        
        return ret  # keys: methods, metrics, mean, median, max, min
        
    @staticmethod
    def merge(alist, overwrite_=True):
        """
        
        Use 
        ---
        Used to merge various PerformanceMetrics objects (e.g. returned by different function calls), each of which 
        should ideally reference different methods (otherwise, may be subject to overwriting)
        """
        def verify(): 
            n_rows = 0 
            row_names = []
            for i, p_obj in enumerate(alist): 
                table = p_obj.table
                if i == 0: 
                    n_rows = table.shape[0] 
                    row_names = table.index 
                else: 
                    assert table.shape[0] == n_rows
                    assert all(row_names == table.index)
            return row_names
        import copy

        # emtpy list
        if len(alist) == 0:
            # do nothing 
            print('(merge) No input data. Exiting ...')
            return PerformanceMetrics()  # return a dummy object 

        # design: assume that each df references different methods
        row_names = verify()

        # todo: do it according to the class protocol

        # create a new table and merge all the algorithm metrics (columns)
        p_new = PerformanceMetrics() # DataFrame()
        records = {}   # 
        for i, p_obj in enumerate(alist): 
            table = p_obj.table
            for col in table.columns: 
                if overwrite_ or (not col in p_new.table.columns): # add only if not existed
                    p_new.table[col] = table[col] # if not incr_update, then update no matter what
            p_new.records.update(p_obj.records) # but this overwrites the key (and its value)
        
        p_new.table.index = row_names

        ## this operation does not merge the following base attributes
        # p_new.op 
        # p_new.records 
    
        return p_new      

    # base do() overriden
    def do(self, op):
        assert hasattr(op, '__call__')
        return self.aggregate(op, new_col=op.__name__) 
    apply = do  # alias

    def aggregate(self, op=None, new_col=''):
        if op is None: 
            if not new_col: new_col = 'mean'
            self.table[new_col] = self.table.mean(axis=1)
        else: 
            assert hasattr(op, '__call__')
            if not new_col: new_col = op.__name__
            self.table[new_col] = self.table.apply(op, axis=1)  # apply op over all the columns (as parameters)
        return

    @staticmethod 
    def consolidate(alist, **kargs): 
        """
        Consolidate performance measurements across CV folds.

        This is not the same as merge(), which combines multiple PerformanceMetrics objects (mainly by combining table columns)

        Params
        ------

        **kargs 

        unbag: set to True to merge all bagged models
        exception_: 

        Use 
        ---
        Used after looping over and collecting all performance metrics 
        across all CV folds. 

        Memo
        ----
        1. df.loc[metric] <- row
        """
        if len(alist) == 0: 
            # do nothing 
            print('(consolidate) No input data. Exiting ...')
            return PerformanceMetrics()  # return a dummy object
        
        # consolidate a set/list of PerformanceMetrics
        table = alist[0].table
        methods = table.columns.values
        metrics = table.index.values
        records = descriptions = alist[0].records # typically a bag 

        n_methods = len(methods)
        n_sets = n_folds = len(alist)  # n sets of performance metrics from n fold CV

        # check data integrity 
        if len(alist) > 1: 
            for i, p_obj in enumerate(alist[1:]):
                table = p_obj.table 
                assert all(table.index == metrics)
                assert table.shape[1] == n_methods, "Inconsistent number of methods ..."

        # df_avg = DataFrame(columns=methods)
        p_new = PerformanceMetrics(method=methods)
        for metric in metrics: 
            # if not metric in PerformanceMetrics.tracked: 
            #     msg = 'Metric %s is not in the tracked set: %s\n' % (metric, ' '.join(PerformanceMetrics.tracked))

            M = np.zeros((n_sets, n_methods))
            
            for i, p_obj in enumerate(alist): 
                table = p_obj.table
                # if metric == 'auc': print("(test) auc:\n%s" % table.loc[metric].values)
                M[i, :] = table.loc[metric].values 
            
            # p_new.table.loc[metric] = np.mean(M, axis=0) # row means
            p_new.add_metric(metric=metric, values=np.mean(M, axis=0)) # overwrite_=True

            # if metric == 'auc': print "(test) mean auc:\n%s" %  df_avg.loc[metric]
        p_new.add_doc(records)

        if kargs.get('unbag', False): 
            table = p_new.table 
            p_new.table = unbag(table, bag_count=kargs.get('bag_count', None), sep=kargs.get('sep', '.'), exception_=kargs.get('exception_', False))

        # if clone_: 
        #     return PerformanceMetrics(scores=df_avg)  
        # print('(consolidate) return type: %s' % type(p_new))
        return p_new

    @staticmethod
    def analyze_performance(L_test, Th, method, **kargs):
        return analyzePerf(L_test, Th, method, **kargs)

    @staticmethod
    def log(message, file_name='', context='test', size=-1, domain='', ext='log'): 
        """
        import logging
        logging.basicConfig(filename='example.log',level=logging.DEBUG)
        logging.debug('This message should go to the log file')
        logging.info('So should this')
        logging.warning('And this, too')

        """
        pass

    @staticmethod
    def report(p_baseline, p_target, metrics=[], rule='max', descriptions={}, verbose=True, greater_is_better=True): 
        """

        Params
        ------
        p_baseline: performance scores associated with the baseline methods are represented by p_baseline
        p_target: 

        rule: {'max', 'max-all', 'all-all', }
            max: best baseline method (e.g. SVM among all base predictors) vs best target methods (e.g. NMF)
            max-all: best base line method vs all target methods (i.e. comparing column by column in p_target)

        }

        Use
        ---
        1. p_target: target CF method 
           p_baseline: baseline methods (BPs)

           compare performance wrt a metric, say, AUC

           how does the target CF method compare to the best (max AUC) BP? 

        Memo
        ----
        1. ranking can be computed faster given two sorted lists (see demo_algorithms)

        """
        def rank_perf(method_set, metric, highlight=[], sorted_=True, greater_is_better=True):  # method_set: a list of 2-tuples: (method, score)
            msg =  "(report) Performance ranking on metric=%s ..." % metric
            div(message=msg, symbol='=', border=1)

            sorted_set = method_set if sorted_ else sorted(method_set, key=lambda x: x[1], reverse=greater_is_better)
            for i, (method, score) in enumerate(sorted_set):  # sorted according to metric (default, the larger the better)
                msg = 'rank [%d] | method: %s, score: %f (%s)' % (i+1, method, score, metric)
                if method in highlight: 
                    div(message='  > %s' % msg, symbol='*', border=1)
                else: 
                    print('... %s' % msg)

        if p_target is None or p_target.isEmpty(): 
            print('(report) Null PerformanceMetrics. No-op #')
            return
            
        # init 
        if not metrics: metrics = PerformanceMetrics.tracked

        if verbose: 
            # best performance? 
            for p_obj in (p_baseline, p_target, ): 
                # it possible that baseline metrics is not passed in
                assert isinstance(p_obj, PerformanceMetrics), "Inavlid input type: %s" % p_obj
                # if p_obj is not None and not p_obj.table.empty: 
                div(message=p_obj.table, symbol='=', border=2)

        # init data structures
        baseline_won, target_won = {metric:[] for metric in metrics}, {metric:[] for metric in metrics}
        base_set, target_set = [], []  # [todo]
        topk = 5
        ranking = []
        if rule in ('max', 'best-best', ):  # best vs best 
            for metric in metrics: 
                
                ## sort performance
                # base_set: a list of 2-tuples: (method, score)
                base_set = p_baseline.sort(metric=metric, reverse=greater_is_better)
                target_set = p_target.sort(metric=metric, reverse=greater_is_better)  # usu. only method is being compared to
                ranking = sorted(base_set+target_set, key=lambda x: x[1], reverse=greater_is_better)

                # [todo] find all methods that won
                best_baseline, best_target = base_set[0], target_set[0]
                if best_baseline[1] > best_target[1]:
                    # baseline_won[metric].append(best_baseline)  # + top K at this point
                    # baseline_won[metric]['winner'].append(best_baseline)
                    # baseline_won[metric]['ranking'] = ranking # [:topk]  # the topk (method, score) at this point? 

                    baseline_won[metric].append((best_baseline, best_target)) # an instance of 'winning' 

                    # where does the target method stand at this point? 
                else: 
                    # target_won[metric].append(best_target)
                    # target_won[metric]['winner'].append(best_target)
                    # target_won[metric]['ranking'] = ranking # [:topk]

                    target_won[metric].append((best_target, best_baseline))

                # [result]
                if metric in ['auc', 'fmax', 'fmax_negative', ]:
                    rank_perf(ranking, metric=metric, highlight=[m for m, s in target_set], sorted_=True) 

        elif rule in ('max-all', 'best-all', ):  # bp vs all target methods
            for metric in metrics: 

                # [todo] ranking can be computed faster given two sorted lists (see demo_algorithms)
                # base_set: a list of 2-tuples: (method, score)
                base_set = p_baseline.sort(metric=metric, reverse=greater_is_better)
                target_set = p_target.sort(metric=metric, reverse=greater_is_better)  # multiple target methods (e.g. different stackers)
                ranking = sorted(base_set+target_set, key=lambda x: x[1], reverse=greater_is_better)

                best_baseline = base_set[0]   # (method, score)
                weakest_target = target_set[-1]  # assuming greater is better
                if best_baseline[1] >= weakest_target[1]: 
                    baseline_won[metric].append( (best_baseline, weakest_target) )
                else: 
                    target_won[metric].append( (weakest_target, best_baseline) )

                ## 
                # for i, entry in enumerate(target_set):  # foreach varition of the target method (e.g. nmf, nmf+similarity)
                #     # [todo] find all methods that won
                #     if best_baseline[1] > entry[1]:
                #         # baseline_won[metric].append(best_baseline)
                #         baseline_won[metric]['winner'].append(best_baseline)
                #     else: 
                #         # target_won[metric].append(entry)
                #         target_won[metric]['winner'].append(entry)
                        
                # [result]
                if metric in ['auc', 'fmax', 'fmax_negative', ]:
                    rank_perf(ranking, metric=metric, highlight=[m for m, s in target_set], sorted_=True) 

        elif rule in ('avg', 'mean', 'average', ):
            baseline_won, target_won = [], []
            for metric in metrics: 
                table_baseline = p_baseline.table.mean(axis=1)
                table_target = p_target.table.mean(axis=1)  # usu. only method is being compared to            

            raise NotImplementedError("to be continued ...")
        
        # [result]
        if verbose:
            logger = PerformanceMetrics.getLogger(context='ranking')  # file name depends on context and domain

            target_method = method = descriptions.get('method', 'target_method')
            print('(report) Q1. Does %s has an advantage in any of the metrics? %s' % (method.upper(), True if len(target_won) > 0 else False)) 
            indent_level = 6  # pad extra spaces  # '              '
            for metric, entry in target_won.items():  # target_won: metric -> {'winner', 'topk'}, metric['winner']: {(method, score), ...}, metric[topk]: sortted (method, score)
                for i, paired_records in enumerate(entry): 
                    target_pair, bp_pair = paired_records
                    # print('(test) records: {0}, target_pair: {1}, bp_pair: {2}'.format(paired_records, target_pair, bp_pair))
                    print('... metric=%s | target (%s: %f) >= baseline (%s: %f)' % (metric, target_pair[0], target_pair[1], bp_pair[0], bp_pair[1])) # show the winning method
                    
                # > show the top K methods at this point (with the winner above displayed first) 
                s = format_ranked_list(ranking, metric=metric, topk=topk, verbose=False)  # [note] format_ranked_list is used often enough to be a 'friend' function
                print( s.rjust(len(s)+indent_level, ' ') )
         
            print( '(report) Baseline methods still have advantage over %s in the following metrics ...\n' % target_method )
            for metric, entry in baseline_won.items():
                msg = '' 
                for i, paired_records in enumerate(entry): 
                    bp_pair, target_pair = paired_records
                    msg += '... metric=%s | baseline (%s: %f) >= target (%s: %f)\n' % (metric, bp_pair[0], bp_pair[1], target_pair[0], target_pair[1])

                s = format_ranked_list(ranking, metric=metric, topk=topk, highlight=target_set, verbose=False)
                msg += '{0: >{fill}}\n'.format(s, fill=len(s)+indent_level)  # rank top method but also 'highlight' where the target method stands
                div(msg, symbol="#", border=2)

            msg = '\n(report) Full Ranking:\n'
            s = format_ranked_list(ranking, metric=metric, topk=None)
            msg += '{0: >{fill}}\n'.format(s, fill=len(s)+indent_level)  # s.rjust(len(s)+indent_level, ' ')
            
            msg_log = div(msg, symbol="#", border=2); logger.info(msg_log)  # save log to file
        

        return

    @staticmethod
    def getLogger(file_name='', logger_name='', context='report', domain='', ext='log', prefix=None): 
        # if PerformanceMetrics.logging_initialized: 
        #     # do nothing
        #     return
        # import logging
        # basedir = PerformanceMetrics.data_dir
        log_dir = PerformanceMetrics.log_dir   # keep log under <prefix>/log, in which prefix can be customized
        if prefix is not None: 
            PerformanceMetrics.log_dir = log_dir = os.path.join(prefix, 'log')
        if not os.path.exists(log_dir): 
            print('(logging) Creating log directory:\n%s\n' % log_dir)
            os.mkdir(log_dir)

        if not domain: domain = config.domain  
        if not file_name: 
            # performance_metrics-{context}-N{size}-D{domain}'.format(context=context, size=size, domain=domain)
            name = '{context}-D{domain}'.format(context=context, domain=domain)
            file_name = '%s.%s' % (name, ext)
        log_path = os.path.join(log_dir, file_name)

        # basicConfig can only be called once
        # logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO) # format='%(name)s - %(levelname)s - %(message)s'
        
        # create logger
        if not logger_name: logger_name = PerformanceMetrics.__name__
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        # Create handlers
        c_handler = logging.StreamHandler(stream=sys.stdout)
        f_handler = logging.FileHandler(log_path, mode='w')
        c_handler.setLevel(logging.ERROR)
        f_handler.setLevel(logging.INFO)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('[%(name)s] %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        return logger

    ### todo: add methods to compute all metrics


### end PerformanceMetrics

def unbag(df, bag_count=None, sep='.', exception_=False):
    # import re
    import pandas as pd
    from collections import Counter

    # first ensure that columns are sorted according to their names and bag indices 
    df = df.reindex(sorted(df.columns), axis=1) # df[sorted(df.columns.values)]

    # is this a bagged dataframe? 
    # p_col = re.compile(r'^(?P<bp>\w+)\.(?P<bag>\d+)$')

    # n_bagged_cls = sum([1 for col in df.columns.values if p_col.match(col)]) 
    # print('(test) columns:\n%s\n' % df.columns.values)
    n_bagged_cls = sum([1 for col in df.columns.values if len(str(col).split(sep)) >= 2] )  # raise exception when col is not a strong (e.g. numbers)
    tBagged = True if n_bagged_cls > 0 else False
    if not tBagged: 
        msg = "(evaluate.unbag) Input dataframe does not contain bagged models:\n%s\n" % df.columns.values
        if exception_: 
            raise ValueError(msg)
        else: 
            print(msg)

        # nothing else to do 
        return df

    # infer bag count if None
    if tBagged and bag_count is None: 
        counts = Counter([  sep.join(col.split(sep)[:-1]) for col in df.columns.values])  # e.g. single datatype: LogitBoost.0 | multiple datatypes: coexpression.LogitBoost.0 
        bag_count = counts[counts.keys()[0]]

        # assuming that we do not mixed unbagged with bagged
        for name, count in counts.items(): 
            if count != bag_count: 
                msg = "Inconsistent bag counts: %s" % counts
                if exception_: 
                    raise ValueError(msg)
                else: 
                    print(msg)
        print('(unbag) inferred bag_count=%d' % bag_count)

    cols = []
    bag_start_indices = range(0, df.shape[1], bag_count)
    # names = [_.split(sep)[0] for _ in df.columns.values[bag_start_indices]]
    names = [sep.join(col.split(sep)[:-1]) for col in df.columns.values[bag_start_indices]]
    for i in bag_start_indices:
        cols.append(df.iloc[:, i:i+bag_count].mean(axis = 1))
    df = pd.concat(cols, axis = 1)
    df.columns = names
    return df  # tip: this 'df' references the new unbagged dataframe, not the input 'df' => unbag() is not an in-place operation

def format_ranked_list(sorted_pairs, metric='?', topk=None, verbose=False, highlight=[], inverse_highlight=[]): 
    """
    Params
        highlight: the set specified in 'highlight' is a set of strings that represent keywords e.g. 'wmf', 'nmf' 
                   which are to be matched against the columns in the PerformanceMetrics's table ... 
        inverse_highlight: the set specified in 'inverse_highlight' is a set of strings that represent keywords e.g. 'wmf', 'nmf'
                   the columns that do NOT contain these keywords will be highlighted
    """
    def preprocess(sorted_pairs): 
        method_lookup = set()
        
        # instructions = [ (highlight, False), (inverse_highlight, True) ]  # inversed or not
        for target in highlight: 
            target_method = target[0] if isinstance(target, tuple) else target # in (method, score) format 
            assert isinstance(target_method, str)
            for method, _ in sorted_pairs: 
                # print('(test) target: %s, method in sorted_pairs: %s' % (target, method))
                if (target_method in method) or (method in target_method): 
                    method_lookup.add(method)
                # elif inverse and (not target_method in method): # see if they are substring to each other (undirectional) 
                #     method_lookup.add(method) # use the name in the sorted_pairs for lookup later on
        matched = set()
        for target in inverse_highlight:
            target_method = target[0] if isinstance(target, tuple) else target # in (method, score) format 
            assert isinstance(target_method, str)
            for method, _ in sorted_pairs: 
                if (target_method in method) or (method in target_method): 
                    matched.add(method)
        for method, _ in sorted_pairs: 
            if not method in matched: 
                method_lookup.add(method)

        # print('(test) method_lookup:\n%s\n' % method_lookup)
        return method_lookup

    # msg = "Metric: %s\n------" % metric
    # print('(test) full ranking (type: %s):\n%s\n' % (type(sorted_pairs), sorted_pairs))
    method, score = sorted_pairs[0]
    msg = "%s (%s=%f)" % (method, metric, score)

    # preprocess target set (those that we wish to highlight)
    highlighting_methods = preprocess(sorted_pairs)

    targets = [] # [ (method, score, rank), ]
    lookup = dict(sorted_pairs) # dict(target_set)
    for i, (method, score) in enumerate(sorted_pairs[1:]): 
        rank = i+2
        if topk: 
            if rank <= topk: 
                msg += " >= %s (%f)" % (method, score)
        else: 
            msg += " >= %s (%f)" % (method, score)
                
        # highlight these methods
        if method in highlighting_methods: # also keep track of target method's ranking 
            targets.append((method, lookup[method], rank))  # method, score, rank
    
    # this block can be delegated to the display of full ranking 
    indent_level = 8
    if len(targets) > 0:  # descriptions must provide info about the target method of interest 
        msg += '\n'
        for entry in targets: 
            method, score, rank = entry
            s = '~> target:{0} (score={1}, rank={2})\n'.format(method, score, int(rank))
            msg += '{message: >{fill}}\n'.format(message=s, fill=len(s)+indent_level)
    else: 
        msg += '\n'
    if verbose: print( msg )
    return msg

def format_ranked_list2(sorted_pairs, metric='?', topk=None, verbose=False, highlight=[], inverse_highlight=[]): 
    def preprocess(sorted_pairs): 
        method_lookup = set()
        
        # instructions = [ (highlight, False), (inverse_highlight, True) ]  # inversed or not
        for target in highlight: 
            target_method = target[0] if isinstance(target, tuple) else target # in (method, score) format 
            assert isinstance(target_method, str)
            for method, _ in sorted_pairs: 
                # print('(test) target: %s, method in sorted_pairs: %s' % (target, method))
                if (target_method in method) or (method in target_method): 
                    method_lookup.add(method)
                # elif inverse and (not target_method in method): # see if they are substring to each other (undirectional) 
                #     method_lookup.add(method) # use the name in the sorted_pairs for lookup later on
        matched = set()
        for target in inverse_highlight:
            target_method = target[0] if isinstance(target, tuple) else target # in (method, score) format 
            assert isinstance(target_method, str)
            for method, _ in sorted_pairs: 
                if (target_method in method) or (method in target_method): 
                    matched.add(method)
        for method, _ in sorted_pairs: 
            if not method in matched: 
                method_lookup.add(method)

        # print('(test) method_lookup:\n%s\n' % method_lookup)
        return method_lookup

    indent_level = 2
    # msg = "Metric: %s\n------" % metric
    # print('(test) full ranking (type: %s):\n%s\n' % (type(sorted_pairs), sorted_pairs))
    method, score = sorted_pairs[0]

    rank = 1
    s = "[rank #%d] %s (%s=%f)" % (rank, method, metric, score)
    msg = '{message: >{fill}}\n'.format(message=s, fill=len(s)+indent_level)

    # preprocess target set (those that we wish to highlight)
    highlighting_methods = preprocess(sorted_pairs)

    targets = [] # [ (method, score, rank), ]
    lookup = dict(sorted_pairs) # dict(target_set)
    for i, (method, score) in enumerate(sorted_pairs[1:]): 
        rank = i+2
        if topk: 
            if rank <= topk: 
                s = "[rank #%d] %s (%f)" % (rank, method, score)
                msg += '{message: >{fill}}\n'.format(message=s, fill=len(s)+indent_level)
        else: 
            s = "[rank #%d] %s (%f)" % (rank, method, score)
            msg += '{message: >{fill}}\n'.format(message=s, fill=len(s)+indent_level)
                
        # highlight these methods
        if method in highlighting_methods: # also keep track of target method's ranking 
            targets.append((method, lookup[method], rank))  # method, score, rank
    
    # this block can be delegated to the display of full ranking 
    indent_level = 6
    if len(targets) > 0:  # descriptions must provide info about the target method of interest 
        msg += '========\n'
        for entry in targets: 
            method, score, rank = entry
            s = '+ target:{0} (score={1}, rank={2})\n'.format(method, score, int(rank))
            msg += '{message: >{fill}}\n'.format(message=s, fill=len(s)+indent_level)
    else: 
        msg += '\n'
    if verbose: print( msg )
    return msg

def findOptimalCutoff(L, R, **kargs):
    """ 
    Find the optimal probability cutoff point for a classification model related to event rate
    
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    from sklearn.metrics import roc_curve, auc

    metric = kargs.get('metric', 'fmax')
    thresholds = []
    if metric == 'auc': 
        for i in range(R.shape[0]):  # one threshould per user/classifier
            thresholds.append( findOptimalCutoffAuc(L, R[i]))

    elif metric == 'fmax': 
        beta = kargs.get('beta', 1.0)
        pos_label = kargs.get('pos_label', 1)
        for i in range(R.shape[0]):  # one threshould per user/classifier
            thresholds.append( findOptimalCutoffFmax(L, R[i], beta=beta, pos_label=pos_label) )
    else: 
        raise NotImplementedError
    
    assert len(thresholds) == R.shape[0]
    return thresholds

def findOptimalCutoffAuc(y_true, y_score, target_class=None, pos_label=1, neg_label=0): 
    """

    Memo
    ----
    1. Find optimal threshold via Youden's J statistic: sensitivity+specificity-1

        https://www.kaggle.com/willstone98/youden-s-j-statistic-for-threshold-determination
    """

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, threshold = roc_curve(y_true, y_score)
    # roc_auc = auc(fpr, tpr) 

    classes = [neg_label, pos_label]
    num_classes= len(classes)

    J_stats = [None]*num_classes
    opt_thresholds = [None]*num_classes

    # Compute Youden's J Statistic for each class
    for i in range(num_classes):
        J_stats[i] = tpr[i] - fpr[i]
        opt_thresholds[i] = thresholds[i][np.argmax(J_stats[i])]
        print('Optimum threshold for '+classes[i]+': '+str(opt_thresholds[i]))

    # i = np.arange(len(tpr)) 
    # roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})

    # # The optimal cut off would be where tpr is high and fpr is low => tpr - (1-fpr) is zero or near to zero is the optimal cut off point
    # roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]  

    # p_th = list(roc_t['threshold'])[0]
    if not target_class: target_class = pos_label
    p_th = opt_thresholds[target_class]

    return p_th

def findOptimalCutoffFmax(y_true, y_score, beta = 1.0, pos_label=1):
    import common
    import sklearn.metrics

    precision, recall, threshold = sklearn.metrics.precision_recall_curve(y_true, y_score, pos_label) 

    fmax, p_th = common.fmax_score_threshold(y_true, y_score, beta = beta, pos_label = pos_label)
    
    # fmax score
    # f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    # then take argmax/nanmax (considering nan)

    # i = np.arange(len(precision)) 
    # tradeoff = pd.DataFrame({'tf' : pd.Series(precision-recall, index=i), 'threshold' : pd.Series(threshold, index=i)})

    # # The optimal cut off would be where tpr is high and fpr is low => tpr - (1-fpr) is zero or near to zero is the optimal cut off point
    # tradeoff_t = tradeoff.iloc[(tradeoff.tf-0).abs().argsort()[:1]]  

    return p_th

def combiner(Th, weights=None, aggregate_func=np.mean, axis=0, **kargs): 
    """

    Use 
    ---
    1. Th contains reconstructed probabilities 

    2. Th contains preference scores 

       combiner(Th, aggregate_func='pref', T=T)
    """
    # two cases: Th holds the predictive labels OR Th is a 'rating matrix' (users/classifiers vs items/data)
    nrow = 1 
    try: 
        nrow, ncol = Th.shape[0], Th.shape[1]  # Th is 2D
    except: 
        nrow = 1 
        ncol = len(Th)
        Th = Th[np.newaxis, :]

    # W = weights
    if isinstance(aggregate_func, str): 
        if aggregate_func.startswith('pref'): 
            assert 'T' in kargs and kargs['T'] is not None, "Missing T"
            T = kargs['T']
            return combiner_pref(Th, T)  # return predictive scores
        elif aggregate_func in ('mean', 'av'): # mean or average 
            if weights is not None: 
                print('(combiner) aggregate_func: mean | using predict_by_importance_weights() | n(zeros):{}'.format(np.sum(weights==0)))
                return predict_by_importance_weights(Th, weights, aggregate_func='mean', fallback_on_low_weight=True, min_weight=0.1)
            return np.mean(Th, axis=axis)  # e.g. mean prediction of users/classifiers
        elif aggregate_func == 'median':
            if weights is not None: 
                return predict_by_importance_weights(Th, weights, aggregate_func='median', fallback_on_low_weight=True, min_weight=0.1)
            return np.median(Th, axis=axis) 
        else: 
            raise NotImplementedError
    else: 
        assert hasattr(aggregate_func, '__call__')
        predictions = aggregate_func(Th, axis=axis)  # e.g. mean prediction of users/classifiers

    return predictions

def eval_performance(T, labels, **kargs):
    """
    Evaluate performance scores by policy. 
    """

    # aggregate_func = kargs.get('aggregate_func', np.mean)
    mode = kargs.get('mode', 'mean') # options: average (average individual classifier's performance scores)
    aggregate_func = kargs.get('aggregate_func', 'mean')

    ##############################
    topk = kargs.get('topk', -1)
    target_metric = kargs.get('target_metric', 'fmax')
    weights = kargs.get('weights', None)
    ##############################

    if mode == 'mean':
        # [note] aggregate_func: either a string or a function
        predictions = combiner(T, weights=weights, aggregate_func=aggregate_func)
        # print('... predictions > type:{0}: {1}'.format(type(predictions), predictions[:10]))
        metrics = getPerformanceScores(labels, predictions, **kargs)   # opt: metrics, p_threshold
    elif mode.startswith('stack'):
        # import stacking
        # stacker_name = kargs.get('classifier', '')
        # if not stacker_name: stacker_name = aggregate_func
        # assert 'train_data' in kargs and len(kargs['train_data']) >= 2, "Training data must be an N-tuple (N>=2), e.g. (X_train, y_train)"
        # X_train, y_train, *rest = kargs['train_data']

        # print('(eval_performance) Fitting model {name} with n(X_train): {n}'.format(name=stacker_name, n=X_train.shape[1]))
        # stacker = stacking.choose_classifier(stacker_name)  # e.g. log, enet, knn
        # model = stacker.fit(X_train.T, y_train) # remember to take transpose => X_train.T
        # predictions = model.predict_proba(T.T)[:, 1]  # remember to take transpose
        # metrics = getPerformanceScores(labels, predictions, **kargs)   # opt: metrics, p_threshold
        raise NotImplementedError("Coming soon!")

    elif mode.startswith('user'):  # average over individual classifier/user performances
        M = Metrics(op=aggregate_func)  # M.add( (name, score) )
        for i in range(T.shape[0]): 
            metrics = getPerformanceScores(labels, T[i, :], **kargs) 
            for k, v in metrics.items(): 
                M.add( (k, v) )
        metrics = M.aggregate() # take the average

    return metrics, predictions

# friend function for PerformanceMetrics
def analyzePerf(L_test, Th, method, **kargs): 
 
    ### Q1: does CF construct "better" probabilities? This is answered if T is given
    ### Q2: does CF leads to a better combined result? This is the ensemble learning part. 

    tracked = kargs.get('metrics_tracked',  PerformanceMetrics.tracked) # ['auc', 'fmax', 'fmax_negative', 'sensitivity', 'specificity', ] 
    target_metric = kargs.get('target_metric', 'fmax') # focus on this metric when comparing Th and T
    verify = kargs.get('verify', False)
    #######################################################
    # ensemble model
    aggregate_func = kargs.get('aggregate_func', 'mean')
    classifier = kargs.get('classifier', '')  # e.g. log, knn, qda (see stacking.choose_classifier())

    # method as to how multiple ensemble learner's performacne scores are combined 
    #    'mean': take the mean predictions and compare them with the true labels 
    #    'user': use user provided function and assign it to aggregate_func
    #    'stacking'
    mode = kargs.get('mode', 'mean')

    # training data
    tTrainingData = False
    if mode.startswith('stack'):
        assert len(classifier) > 0 or isinstance(aggregate_func, str), \
            "Specifiy the stacker name preferably via 'classifier'; if not, aggregate_func must be a string for the stacker's name (e.g. 'log' for logistic regression)" 
        assert 'train_data' in kargs, "Missing training data (X_train, y_train) to use a stacker!"
        tTrainingData = True

    T = kargs.get('T', None) # provided only when needing to compare T (original) and Th (transformed)
    #######################################################
    # training data

    throw_exception = kargs.get('exception_', True)
    cycle = kargs.get('fold', -1)  # only for debugging
    outer_cycle = kargs.get('outer_fold', -1)
    
    # I/O
    ##############################################
    analysis_path = kargs.get('output_path', PerformanceMetrics.analysis_dir)
    output_file = kargs.get('output_file', 'performance_delta-{meta}.csv'.format(meta=method))
    save = kargs.get('save', False)
    ##############################################
    verbose = kargs.get('verbose', True)

    # perf_scores = {m:[] for m in metrics}

    # if (len(Th.shape) == 2 and Th.shape[0] > 1) or (T is not None and (Th.shape == T.shape)): 

    perf = PerformanceMetrics() 
    if len(Th.shape) > 1 and Th.shape[0] > 1: 
        n_illegal = np.sum(Th > 1.0) + np.sum(Th < 0.0)
        if n_illegal > 0: print('(analyzePerf) %d illegal probabilities found.' % n_illegal)

        metric_score, pv = eval_performance(Th, labels=L_test, 
                            aggregate_func=aggregate_func, 
                                weights=kargs.get('weights', None),  # weighted average? 
                                train_data=kargs.get('train_data', []),  # needed in 'stacking' mode
                                # classifier=classifier, # needed in 'stacking' mode
                                mode=mode, exception_=throw_exception)  # mean is not the ideal combining rule
    else: 
        assert len(Th.shape) == 1, "Th must be already a 1-D prediction vector but got {}".format(Th)
        # Th must be already a 1-D prediction vector
        pv = Th
        metric_score = getPerformanceScores(L_test, pv, exception_=False)   # opt: metrics, p_threshold 
    # ... if predictions have been made
    # metric_score = getPerformanceScores(labels, predictions, **kargs)   # opt: metrics, p_threshold

    # print('... (verify) got scores for metrics (via eval_performance()): {metrics}'.format(metrics=list(metric_score.keys())))
    # ... ['auc', 'fmax', 'fmax_negative', 'sensitivity', 'specificity', 'brier', 
    #             'TPR', 'recall', 'TNR', 'precision', 'PPV', 'NPV', 'FPR', 'FNR', 'FDR', 'accuracy']
    perf.add(scores=metric_score, method=method)  # use a dataframe to keep tracck of performance scores, which then facilia

    if T is not None: 
        # compare T and Th on the test set (with labels given by L_test)
        # compareEstimates(L_test, Ts=[T, Th, ], methods=['bp', method, ], U=None, fold=kargs.get('fold', 0))
        
        # compute performance delta perf(Th) - perf(T)
        metric_score_baseline, pv0 = eval_performance(T, labels=L_test, 
                                        # weights=kargs.get('weights', None),
                                        train_data=kargs.get('train_data', []),  # needed in 'stacking' mode
                                        classifier=classifier, # needed in 'stacking' mode
                                        aggregate_func=aggregate_func, 
                                        mode=mode, 
                                        exception_=throw_exception)

        # header = ['method', ] + [k for k in metric_score.keys()] + ['%s_h' % k for k in metric_score.keys()]
        table = {'method': [method, ], }
        for metric, score in metric_score_baseline.items(): 
            if not metric in table: table[metric] = []
            table[metric].append(score)

            metric_h = '{m}_h'.format(m=metric)
            if not metric_h in table: table[metric_h] = []
            table[metric_h].append( metric_score[metric] )

            if metric == target_metric: 
                delta = table[metric_h][-1] - table[metric][-1]  
                table['delta_%s' % metric] = delta

        # save performance delta 
        if save: 
            df = DataFrame(table)
            fpath = os.path.join(analysis_path, output_file)
            df.to_csv(fpath, index=False, sep=',')
            print('(output) Saved performance_delta to:\n{path}\n ... (verify) #'.format(path=os.path.join(analysis_path, output_file)))
        else: 
            # usually, there's only one row 
            displayed = ['fmax', 'auc', ]
            div("[result] Cycle ({fo}, {fi}) | method: {method}".format(fo=outer_cycle, fi=cycle, method=method), symbol='#')
            for metric, score in metric_score_baseline.items():
                if metric in displayed: 
                    delta = metric_score[metric]-score
                    print('... metric {m} | score: {s} -> score_h: {s2} | delta: {delta} ({symbol})'.format(m=metric, 
                        s=score, s2=metric_score[metric], delta=delta, symbol='+' if delta >=0 else '-'))
    
    assert len(pv) == len(L_test)
    return perf, pv  # holds a dataframe where rows are indexed by performance metrics and columns are indexed by methods
            
def compareEstimates(L_test, Ts, **kargs):
    """
    Compare the probability estimates between T and Th (possibly multple sets), where T comes from base predictors and 
    Th comes from a CF algorithm (specifiied by - method -). 

    Params
    ------
    Ts: a set of matrices representing different probability estimates
        T, Th1, Th2, ... 

    **kargs

    U: users
    method: 
    """
    def eval_entry(y_actual, y_hat):   # [todo]
        TP = FP = TN = FN = 0
        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
                TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
                FP += 1
            if y_actual[i]==y_hat[i]==0:
                TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
                FN += 1

        return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    def to_label(y_hat, p_threshould=0.5): 
        labels = np.zeros(len(y_hat))
        for i, p in enumerate(y_hat): 
            if p >= p_threshould: 
               labels[i] = 1
        return list(labels)
    def report(T_d, Th_d, metrics, greater_is_better=True, method='?', log=False):  
        T_won, Th_won = [], []
        for metric in metrics: 
            # assume greater is better
            if T_d[metric] > Th_d[metric]: 
                T_won.append((metric, T_d[metric]))
            else: 
                Th_won.append((metric, Th_d[metric]))
    
        print('(result) @compareEstimates')
        if len(Th_won) > 0: 
            msg = 'Th (method: {target}) has an advantage in the following metrics (fold={fold}):\n'.format(target=method, fold=kargs.get('fold', '?'))
            for metric, score in Th_won: 
                msg += '... Th ({metric}: {s1}; {target}) > T ({s2})\n'.format(metric=metric, s1=score, target=method, s2=T_d[metric]) 
            result = div(msg, symbol='*', border=2); result = logger.info(result)
             
        if len(T_won) > 0:    
            msg = 'T has an advantage (over {target}) in the following metrics (fold={fold}):\n'.format(target=method, fold=kargs.get('fold', '?'))
            for metric, score in T_won: 
                msg += '... T (%s: %f) > Th (%f; %s)\n' % (metric, score, Th_d[metric], method) 
            div(msg, symbol='*', border=2)
        return

    # todo
    # base_perf = analyzeBasePerf(L_test, T, U, **kargs)
    # analyzePerf = analyzePerf(L_test, Th, method=method)
    logger = PerformanceMetrics.getLogger(logger_name='compareEstimates', context='T_Th_comparison')  # logger_name: determines the logger's name; same logger can only be initiated once
    assert len(Ts) >= 2

    # optional params
    metrics = kargs.get('metrics', PerformanceMetrics.tracked)
    target_method = kargs.get('method', '?')  # if there's only one target method
    methods = kargs.get('methods', ['bp', 'cf', ]) if target_method == '?' else ['bp', target_method]

    # Q1: does CF construct "better" probabilities? 
    T, Th = Ts[0], Ts[1]  # T: probabilities from BPs, Th: probabilities from target methods
    n_users, n_items = T.shape[0], T.shape[1]
    assert n_users == Th.shape[0]
    assert n_items == Th.shape[1]

    ## Sol 1-1: think of each row of T & Th as a result of a classifier predicition; same applies to Th even though it's re-estimated from CF algorithms

    for i in range(1, len(Ts)): 
        T, Th = Ts[0], Ts[i]   # T, from BPs, is always the first one
        estT, estTh = Metrics(), Metrics() # {metric:[] for metric in metrics}
        ut = random.choice(range(n_users))
        for u in range(n_users): 
            t, th = T[u, :], Th[u, :]
            mdict = scores(L_test, t, **kargs)  # mdict: metric -> score
            estT.add(mdict)  # add scores user by user
            mdict = scores(L_test, th, **kargs)
            estTh.add(mdict)
            # if u == ut: print('(compareEstimates) Test | user #%d estT:\n%s\n... estTh:\n%s\n' % (ut, estT.records, estTh.records))
        T_d = estT.apply(op=np.mean); # print('... T_d:\n%s\n' % T_d)
        Th_d = estTh.apply(op=np.mean); # print('... Th_d:\n%s\n' % Th_d)

        report(T_d, Th_d, metrics=['auc', 'fmax', 'fmax_negative', ], method=methods[i])

    ## Sol 1-2: overall predictability? accuracy
    p_th = kargs.get('p_threshold', 0.5)
    estT, estTh = Metrics(), Metrics()
    for i in range(1, len(Ts)): 

        ut = random.choice(range(n_users))
        for u in range(n_users): 
            t, th = T[u, :], Th[u, :]
            # yhat_t = to_label(L_test, t, p_threshould=p_th)
            # yhat_th = to_label(L_test, th, p_threshould=p_th)
            # entries_t = eval_entry(L_test, yhat_t)
            # entries_th = eval_entry(L_test, yhat_th)

            # update
            estT.add(eval_entry(L_test, to_label(t, p_threshould=p_th)))
            estTh.add(eval_entry(L_test, to_label(th, p_threshould=p_th)))
            # if u == ut: print('(compareEstimates) Test | user #%d estT:\n%s\n... estTh:\n%s\n' % (ut, estT.records, estTh.records))

        # print('(test 1.5) estT:\n%s\n' % estT.records)
        T_d = estT.apply(op=sum); # print('(test) summed > T_d:\n%s\n' % T_d) # get total TP, FP, ... 
        Th_d = estTh.apply(op=sum) 
        T_est = perf_measures2(T_d, ep=1e-9); #  print('(test) T_est:\n%s\n' % T_est)
        Th_est = perf_measures2(Th_d, ep=1e-9)

        # report(T_est, Th_est, metrics=['accuracy', 'recall', 'precision'], method=methods[i])

    return

def validationCurve(X=None, y=None): 
    """

    Memo
    ----
    1. references 
        a. yellowbrick
            http://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
        b. scikit-learn
            https://scikit-learn.org/stable/modules/learning_curve.html#validation-curve

    2. reference modules 
            plot_validation_curve
            plot_learning_curve


    """
    from sklearn.model_selection import validation_curve

    if not X or not y: 
        from sklearn.datasets import load_iris
        from sklearn.linear_model import Ridge

        ### generate data
        np.random.seed(0)
        iris = load_iris()
        X, y = iris.data, iris.target
        indices = np.arange(y.shape[0])
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

    train_scores, valid_scores = validation_curve(Ridge(), X, y, "alpha",
                                                  np.logspace(-7, 3, 3),
                                                  cv=5)

    return 

def learningCurve(): 

    return


# utils_plot? 
# hook for PerformanceMetrics.plot()
def plot_barh(perf, metric='fmax', **kargs): 
    """

    Reference
    ---------
    1. bar annotation  
        + http://robertmitchellv.com/blog-bar-chart-annotations-pandas-mpl.html

        + Annotate bars with values on Pandas bar plots
            https://stackoverflow.com/questions/25447700/annotate-bars-with-values-on-pandas-bar-plots

            <idiom> 
            for p in ax.patches:
                ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

        + adding value labels on a matplotlib bar chart
            https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart

    2. colors:  
        + https://stackoverflow.com/questions/11927715/how-to-give-a-pandas-matplotlib-bar-graph-custom-colors

        + color map
          https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib

        + creating custom colormap: 
            https://matplotlib.org/examples/pylab_examples/custom_cmap.html

    3. fonts: 
        http://jonathansoma.com/lede/data-studio/matplotlib/changing-fonts-in-matplotlib/
    """
    from utils_plot import truncate_colormap 
    from itertools import cycle, islice
    import matplotlib.colors as colors
    from matplotlib import cm

    # perf: PerformanceMetrics object

    plt.clf()

    # performance metric vs methods
    methods = kargs.get('methods', [])  # [] to plot all methods
    perf_methods = perf.table.loc[metric] if not methods else perf.table.loc[metric][methods]  # a Series
    if kargs.get('sort', True): 
        perf_methods = perf_methods.sort_values(ascending=kargs.get('ascending', False))

    # default configuration for plots
    matplotlib.rcParams['figure.figsize'] = (10, 20)
    matplotlib.rcParams['font.family'] = "sans-serif"

    # canonicalize the method/algorithm name [todo]

    # coloring 
    # cmap values: {'gnuplot2', 'jet', }  # ... see demo/coloarmaps_refeference.py
    ####################################################################
    # >>> change coloring options here
    option = 2
    if option == 1: 
        cmap = truncate_colormap(plt.get_cmap('jet'), 0.2, 0.8) # params: minval=0.0, maxval=1.0, n=100)

        # fontsize: Font size for xticks and yticks
        ax = perf_methods.plot(kind=kargs.get('kind', "barh"), colormap=plt.get_cmap('gnuplot2'), fontsize=12)  # colormap=cmap 
    elif option == 2: 
        # Make a list by cycling through the desired colors to match the length of your data
        
        # methods_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(perf_methods)))
        if 'color_indices' in kargs: 
            method_colors = kargs['color_indices']
        else: 
            n_models = len(perf_methods)
            method_colors = cm.gnuplot2(np.linspace(.2,.8, n_models+1))  # this is a 2D array

        ax = perf_methods.plot(kind=kargs.get('kind', "barh"), color=method_colors, fontsize=12)  # colormap=cmap 
    elif option == 3:  # fixed color 
        # my_colors = ['g', 'b']*5 # <-- this concatenates the list to itself 5 times.
        # my_colors = [(0.5,0.4,0.5), (0.75, 0.75, 0.25)]*5 # <-- make two custom RGBs and repeat/alternate them over all the bar elements.
        method_colors = [(x/10.0, x/20.0, 0.75) for x in range(len(perf_methods))] # <-- Quick gradient example along the Red/Green dimensions.
        ax = perf_methods.plot(kind=kargs.get('kind', "barh"), color=method_colors, fontsize=12)  # colormap=cmap 
    ####################################################################

    # title 
    ax.set_xlabel('%s scores' % metric.upper(), fontname='Arial', fontsize=12) # color='coral'
    ax.set_ylabel("Methods in descending order of performance", fontname="Arial", fontsize=12)
    
    ## annotate the bars
    # Make some labels.
    labels = scores = perf_methods.values
    min_abs, max_abs = kargs.get('min', 0.0), kargs.get('max', 1.0)
    min_val, max_val = np.min(scores), np.max(scores)
    epsilon = 0.1 
    min_x, max_x = max(min_abs, min_val-epsilon), min(max_abs, max_val+epsilon)

    # [todo] set upperbound according to the max score
    ax.set_xlim(min_x, max_x)

    # annotation option
    option = 1
    if option == 1: 
        for i, rect in enumerate(ax.patches):
            x_val = rect.get_x()
            y_val = rect.get_y()

            # if i % 2 == 0: 
            #     print('... x, y = %f, %f  | width: %f, height: %f' % (x_val, y_val, rect.get_width(), rect.get_height()))   # [log] x = 0 all the way? why? rect.get_height = 0.5 all the way
            # get_width pulls left or right; get_y pushes up or down
            # memo: get_width() in units of x ticks
            assert rect.get_width()-scores[i] < 1e-3

            width = rect.get_width()
            # lx_offset = width + width/100. 

            label = "{:.3f}".format(scores[i])
            ax.text(rect.get_width()+0.008, rect.get_y()+0.08, \
                    label, fontsize=8, color='dimgrey')
    elif option == 2: 
        # For each bar: Place a label
        for i, rect in enumerate(ax.patches):
            # Get X and Y placement of label from rect.
            y_value = rect.get_y()  # method  ... rect.get_height()
            x_value = rect.get_x()  # score

            # Number of points between bar and label. Change to your liking.
            space = 5
            
            # # Vertical alignment for positive values
            # va = 'bottom'

            # If value of bar is negative: Place label below bar
            # if x_value < 0:
            #     # Invert space to place label below
            #     space *= -1
            #     # Vertically align label at top
            #     va = 'left'

            # Use Y value as label and format number with one decimal place
            label = "{:.2f}".format(x_value) # "{:.1f}".format(y_value)

            # Create annotation
            plt.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va='center')                      # Vertically align label differently for
                #                             # positive and negative values.

    # invert for largest on top 
    # ax.invert_yaxis()  # 

    plt.yticks(fontsize=8)
    plt.title("Performance Comparison In %s" % metric.upper(), fontname='Arial', fontsize=15) # fontname='Comic Sans MS'

    n_methods = perf.n_methods()
    
    # need to distinguish the data set on which algorithms operate
    domain = perf.query('domain')
    if domain == 'null': 
        assert 'domain' in kargs
        domain = kargs['domain']

    file_name = kargs.get('file_name', '{metric}_comparison-N{size}-D{domain}'.format(metric=metric, size=n_methods, domain=domain))  # no need to specify file type
    if 'index' in kargs: 
        file_name = '{prefix}-{index}'.format(prefix=file_name, index=kargs['index'])
    saveFig(plt, PerformanceMetrics.plot_path(name=file_name), dpi=300)  # basedir=PerformanceMetrics.plot_path

    return
# universal interface 
# plot_performance = plot_barh   # pick a plot functiond

def plot_performance(perf, metric='fmax', **kargs): 
    import math
    from matplotlib import cm
    from algorithms import split

    visual_limit = 30

    n_methods = perf.n_methods()

    # configure the main function here
    plot_func = kargs.get('plot_func', plot_barh)  # <<< CONFIGURE 
    assert hasattr(plot_func, '__call__')

    if n_methods > visual_limit: 
        n_parts = int(math.ceil(n_methods/ (visual_limit+0.0) ))
        
        perfx = perf.divide(n_parts=n_parts, metric=metric)
        print('(plot_performance) n_methods: %d > visual_limit: %d? %s | n_parts: %d =?= %d' % \
            (n_methods, visual_limit, n_methods > visual_limit, n_parts, len(perfx)))

        method_colors = cm.gnuplot2(np.linspace(.2,.8, n_methods+1))  # this is a 2D array
        color_indices = list(split(range(n_methods), len(perfx)))

        for i, perf in enumerate(perfx): 
            kargs['index'] = i+1

            indices = color_indices[i]
            lower, upper = indices[0], indices[-1]
            subset = method_colors[lower: upper]
            plot_func(perf, metric=metric, color_indices=subset, **kargs)  # color_indices=subset
    else: 
        plot_func(perf, metric=metric, **kargs)

    return

def measureGeneralization(model, sample, scoring):
    """

    'sample' consists of 
        train_split: X_train, y_train
        test_split: X_test, y_test
        => 4-tuple: (X_train, y_train, X_test, y_test)

    scoring: a scoring function (e.g.  sklearn.metrics.roc_auc_score)
    
    y_train: a 2-tuple consisting of (y_true, y_score) from the training set
    y_test: a 2-tuple ... from the test set

    Memo
    ----
    1. Compare training error and test error. 
       Compare training performance and test performance. 

       => average difference of errors or 
          average difference of performance 

    2. Reference
       demo/plot_train_error_vs_test_error

    """ 
    assert len(sample) >= 2
    X_train, y_train, X_test, y_test = sample

    
    train_errors = list()
    test_errors = list()
        
    # model.set_params(alpha=alpha)
    model.fit(X_train, y_train)

    # mean accuracy
    # train_errors.append(model.score(X_train, y_train))
    # test_errors.append(model.score(X_test, y_test))

    return

def plot_path(name='test', basedir=None, ext='tif', create_dir=False):
    # create the desired path to the plot by its name
    if basedir is None: basedir = os.path.join(os.getcwd(), 'plot')
    if not os.path.exists(basedir) and create_dir:
        print('(plot) Creating plot directory:\n%s\n' % basedir)
        os.mkdir(basedir) 
    return os.path.join(basedir, '%s.%s' % (name, ext))

def rmse_cv(model, X, y):
    # DeprecationWarning: Scoring method mean_squared_error was renamed to neg_mean_squared_error in version 0.18
    # rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="mean_squared_error", cv = 5))
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
    
    return(rmse)

def select_model_elnet(X, y, alphas=[], l1_ratios=[]):
    # from sklearn.linear_model import ElasticNet

    # [todo] for now, only works on ElasticNet

    alphas = [0.0005, 0.001, 0.01, 0.03, 0.05, 0.1]
    l1_ratios = [1.5, 1.1, 1, 0.9, 0.8, 0.7, 0.5] 

    # grid search
    enet = ElasticNet(alpha =alpha, l1_ratio=l1_ratio)
    cv_elastic = [rmse_cv(enet, X, y).mean() for (alpha, l1_ratio) in product(alphas, l1_ratios)]

    plt.clf()
    matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
    idx = list(product(alphas, l1_ratios))
    p_cv_elastic = pd.Series(cv_elastic, index = idx)
    p_cv_elastic.plot(title = "Validation - Just Do It")
    plt.xlabel("alpha - l1_ratio")
    plt.ylabel("rmse")

    saveFig(plt, plot_path(name='rmse_cv'))

    # Zoom in to the first 10 parameter pairs
    matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
    idx = list(product(alphas, l1_ratios))[:10]
    p_cv_elastic = pd.Series(cv_elastic[:10], index = idx)
    p_cv_elastic.plot(title = "Validation - Just Do It")
    plt.xlabel("alpha - l1_ratio")
    plt.ylabel("rmse")

    saveFig(plt, plot_path(name='rmse_cv_zoomed'))

    return

def select_model_elnet_classifier(alphas=[], l1_ratios=[]): 
    """

    Memo
    ----
    1. examples: 
       SGDClassifier(max_iter=1000, tol=1e-3)
    """

    from sklearn.linear_model import SGDClassifier

    coef = pd.Series(elastic.coef_, index = X_train.columns)


    return

def demo_r2_score(X_train, y_train, X_test, y_test): 
    # from sklearn.metrics import r2_score
    # from sklearn.linear_model import ElasticNet

    alpha = 0.1
    enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

    y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
    r2_score_enet = r2_score(y_test, y_pred_enet)
    print(enet)
    print("r^2 on test data : %f" % r2_score_enet)

    plt.plot(enet.coef_, color='lightgreen', linewidth=2,
         label='Elastic net coefficients')
    plt.plot(lasso.coef_, color='gold', linewidth=2,
         label='Lasso coefficients')
    plt.plot(coef, '--', color='navy', label='original coefficients')
    plt.legend(loc='best')
    plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))

    # save plot
    # basedir = os.path.join(os.getcwd(), 'plot')
    # ext = 'tif' 
    # fpath = os.path.join(basedir, 'sparse_coefficients.%s' % ext)
    saveFig(plt, plot_path(name='sparse_coefficients'), ext=ext)

    return

def visualize_coeffs(X, y, features=[], **kargs): 
    """

    Memo
    ----
    1. Example parameters: 
       'penalty': 'elasticnet', 'alpha': 0.001, 'loss':
                      'modified_huber', 'fit_intercept': True, 'tol': 1e-3

    2. Changing fonts in matplotlib: 
       http://jonathansoma.com/lede/data-studio/matplotlib/changing-fonts-in-matplotlib/

       Adjust figure sizes 
       https://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib

    """
    from sklearn.linear_model import SGDClassifier

    if len(features) == 0: 
        features = [f'x{i}' for i in range(X.shape[1])]

    output_file = kargs.get('output_file', 'coeffs') # output file minus the extension 

    # print('... n_features: %d | dim(X):%s, dim(y):%s' % (len(features), str(X_train.shape), str(y_train.shape)))
    
    # [todo] customize classifier
    params = {'penalty': 'elasticnet', 'alpha': 0.01, 'loss': 'modified_huber', 'fit_intercept': True, 'tol': 1e-3}
    model = SGDClassifier(**params)

    model.fit(X, y)  # predict(X_test)
    # print('... n_coeffs: %d =>\n%s\n' % (len(model.coef_), model.coef_))

    assert hasattr(model, 'coef_'), "%s has no coef_!" % model
    coef = pd.Series(model.coef_[0], index = features)   # coeff_ is a 2D array

    print("(visualize_coeffs) Elastic Net picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

    # imp_coef = pd.concat([coef.sort_values().head(10),
    #                  coef.sort_values().tail(10)])
    imp_coef = coef.sort_values()

    plt.clf()
    matplotlib.rcParams['figure.figsize'] = (16, 20)
    # matplotlib.rcParams['font.size'] = 9  # update({'font.size': 22})

    # (a1)
    # fig = plt.figure(1) # We prepare the plot  
    # plot = fig.add_subplot(111)  # We define a fake subplot that is in fact only the plot.  

    ax = imp_coef.plot(kind = "barh")
    # ax.set_xlabel("", fontname="Arial", fontsize=12)
    # ax.set_ylabel("Base predictors", fontname="Arial", fontsize=16)
    # ax.set_title("Coefficients in the Elastic Net Model", fontsize=12)

    # change the fontsize of minor ticks label ... (a1)
    # plot.tick_params(axis='both', which='major', labelsize=10)
    # plot.tick_params(axis='both', which='minor', labelsize=8)

    # plt.xticks(fontsize=14, rotation=90)
    plt.yticks(fontsize=6)
    plt.title("Coefficients in the Elastic Net Model", fontsize=12)

    if output_file == '': output_file = 'coeffs'
    saveFig(plt, plot_path(name=output_file), dpi=300)

    return model

def visualizeCoeffs(model, features, file_name='', exception_=False): 
    """
    Given a trained model (fitted with training data), visualize the model coeffs

    """
    assert hasattr(model, 'coef_'), "%s has no coef_!" % model

    try: 
        coef = pd.Series(model.coef_[0], index = features)   # coeff_ is a 2D array
    except: 
        msg = "%s has not coef_!" % model
        if exception_: 
            raise ValueError(msg)
        else: 
            print( msg )

            # [todo] do something else
            return

    print("(visualize_coeffs) Elastic Net picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    imp_coef = coef.sort_values()

    plt.clf()
    matplotlib.rcParams['figure.figsize'] = (16, 20)

    ax = imp_coef.plot(kind = "barh")
    plt.yticks(fontsize=6)
    plt.title("Coefficients in the Elastic Net Model", fontsize=12)

    if file_name == '': file_name = 'coeffs'
    saveFig(plt, PerformanceMetrics.plot_path(name=file_name), dpi=300)

    return

def fmax_score(labels, predictions, beta = 1.0, pos_label = 1):
    """
        Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
        Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In Introduction to Information Retrieval. Cambridge University Press.
    """
    return common.fmax_score(labels, predictions, beta=beta, pos_label=pos_label)

def score(y_true, y_score, metric='auc', **kargs):
    """

    Reference
    ---------
    1. model evaluation
           https://scikit-learn.org/stable/modules/model_evaluation.html

    """
    import sklearn.metrics as skmetrics 
    
    if metric == 'auc': 
        return skmetrics.roc_auc_score(y_true, y_score)
    elif metric == 'fmax': 
        return fmax_score(y_true, y_score, beta = 1.0, pos_label = kargs.get('pos_label', 1))
    elif metric == 'fmax_negative': 
        return fmax_score(y_true, y_score, beta = 1.0, pos_label = 0)
    elif metric.startswith('brier'):
        loss = skmetrics.brier_score_loss(y_true, y_score)
        # convert to confidence for convenience (i.e. the greater the better)
        return 1.-loss
    else: 
        # user provided scoring function
        scoring = kargs.get('scoring', None)
        if hasattr(scoring, '__call__'): 
            return scoring(y_true, y_score)
        else: 
            # sensitivity, specificity, 
            metrics = perf_measures(y_true, y_score, p_threshould=kargs.get('p_threshould', 0.5))
            if metric in metrics: 
                return metrics[metric]

    msg = "Unknown metric: %s" % metric
    raise ValueError(msg)

def getPerformanceScores(y_true, y_score, **kargs):
    # kargs
    #   metrics 
    #   exception_
    #   p_threshold
    def eval_metrics(metrics, y_true, y_score):
        table = {}
        for metric in metrics: 
            try: 
                table[metric] = score(y_true, y_score, metric=metric)  # fmax, auc, etc.
            except Exception as e:
                msg = "(getPerformanceScores) metric={metric} > {error}\n... y_true:{label}\n... y_score:{prediction}\n".format(metric=metric, 
                    error=e, label=y_true[:100], prediction=y_score[:100])
                if kargs.get('exception_', True): 
                    raise RuntimeError(msg)
                else: 
                    y_score = np.nan_to_num(y_score, nan=0.0)
                    table[metric] = score(y_true, y_score, metric=metric)  # fmax, auc, etc.
                    print(msg)
                    # table[metric] = -1
        return table

    metrics = kargs.get('metrics', Metrics.tracked)
    metric_table = eval_metrics(metrics, y_true, y_score)  # not all metric is defined in score()

    # print('... (verify) got scores for metrics (via eval_metrics()): {metrics}'.format(metrics=list(metric_table.keys())))
    # ... example: ['auc', 'fmax', 'fmax_negative', 'sensitivity', 'specificity', 'brier']

    fmax, pth_fmax = common.fmax_score_threshold(y_true, y_score, beta = 1.0, pos_label = 1)

    # precision recall analysis
    metric_table.update( analyze_precision_recall(y_true, y_score, p_threshold=kargs.get('p_threshold', pth_fmax)) )
    
    return metric_table

def scores(y_true, y_score, **kargs): 
    def run_test(metric):
        try: 
            metric_table[metric] = score(y_true, y_score, metric=metric)
        except Exception as e:
            msg = "(scores) metric={metric} > {error}\n... y_true:{label}\n... y_score:{prediction}\n".format(metric=metric, 
                error=e, label=y_true[:100], prediction=y_score[:100])
            if kargs.get('exception_', True): 
                raise RuntimeError(msg)
            else: 
                print(msg)
                metric_table[metric] = -1

    metrics = kargs.get('metrics', Metrics.tracked)

    # sensitivity, specificity, ... that depend on probability threshold
    metric_table = perf_measures(y_true, y_score, p_threshold=kargs.get('p_threshold', 0.5))
    for metric in metrics: 
        if not metric in metric_table: 
            # metric_table[metric] = score(y_true, y_score, metric=metric)
            run_test(metric)
    return metric_table

def analyze_precision_recall(y_true, y_score, **kargs):
    """

    Params
    ------
    y_true, y_score

    confusion_matrix: assuming that confusion_matrix is a dataframe 

    """
    def eval_entry(y_actual, y_hat):   # [todo]
        TP = FP = TN = FN = 0
        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
                TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
                FP += 1
            if y_actual[i]==y_hat[i]==0:
                TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
                FN += 1

        return (TP, FP, TN, FN)

    from sklearn.metrics import confusion_matrix

    p_threshold = kargs.get('p_threshold', 0.5)
    ep = 1e-9
    
    # need to convert probability scores to label predictions 
    y_pred = np.zeros(len(y_true))
    for i, p in enumerate(y_score): 
        # if i == 0: print('... p: {0}, pth: {1}'.format(p, p_threshold))
        if p >= p_threshold: 
            y_pred[i] = 1

    # cm = confusion_matrix(y_true, y_pred)
    # FP = cm.sum(axis=0) - np.diag(cm)  
    # FN = cm.sum(axis=1) - np.diag(cm)
    # TP = np.diag(cm)
    # TN = cm.sum() - (FP + FN + TP)  # remove values for numpy array

    TP, FP, TN, FN = eval_entry(y_true, y_pred)

    metrics = {}
    # print('... nTP: %s, nTN: %s, nFP: %s, nFN: %s' % (TP, TN, FP, FN))

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN+ep)
    metrics['sensitivity'] = metrics['TPR'] = metrics['recall'] = TPR 

    # Specificity or true negative rate
    TNR = TN/(TN+FP+ep) 
    metrics['specificity'] = metrics['TNR'] = TNR

    # Precision or positive predictive value
    PPV = TP/(TP+FP+ep)
    metrics['precision'] = metrics['PPV'] = PPV

    # Negative predictive value
    NPV = TN/(TN+FN+ep)
    metrics['NPV'] = NPV

    # Fall out or false positive rate
    FPR = FP/(FP+TN+ep)
    metrics['FPR'] = FPR

    # False negative rate
    FNR = FN/(TP+FN+ep)
    metrics['FNR'] = FNR

    # False discovery rate
    FDR = FP/(TP+FP+ep)
    metrics['FDR'] = FDR

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN+ep) 
    metrics['accuracy'] = ACC

    return metrics
#######################
# alias 
perf_measures = analyze_precision_recall
##############################################

def perf_measures2(entries, ep=1e-9): 
    TP, FP, TN, FN = entries['TP'], entries['FP'], entries['TN'], entries['FN']

    metrics = {}
    # print('... nTP: %s, nTN: %s, nFP: %s, nFN: %s' % (TP, TN, FP, FN))

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN+ep)
    metrics['sensitivity'] = metrics['TPR'] = metrics['recall'] = TPR 

    # Specificity or true negative rate
    TNR = TN/(TN+FP+ep) 
    metrics['specificity'] = metrics['TNR'] = TNR

    # Precision or positive predictive value
    PPV = TP/(TP+FP+ep)
    metrics['precision'] = metrics['PPV'] = PPV

    # Negative predictive value
    NPV = TN/(TN+FN+ep)
    metrics['NPV'] = NPV

    # Fall out or false positive rate
    FPR = FP/(FP+TN+ep)
    metrics['FPR'] = FPR

    # False negative rate
    FNR = FN/(TP+FN+ep)
    metrics['FNR'] = FNR

    # False discovery rate
    FDR = FP/(TP+FP+ep)
    metrics['FDR'] = FDR

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN+ep) 
    metrics['accuracy'] = ACC

    return metrics

def plot_roc(cv_data, **kargs):
    """
    
    Params
    ------
    cv_data: a list of (y_true, y_score) obtained from a completed CV process (e.g. datasink)

    **kargs
    -------
    file_name

    Memo
    ----
    1.  Run classifier with cross-validation and plot ROC curves
            cv = StratifiedKFold(n_splits=6)
            classifier = svm.SVC(kernel='linear', probability=True,
                            random_state=random_state)
    """
    from scipy import interp
    from sklearn.metrics import roc_curve, auc

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    n_fold = len(cv_data)
    if not n_fold: 
        print('(plot_roc) No CV data. Aborting ...')
        return

    plt.clf()
    for i, (y_true, y_score) in enumerate(cv_data):
        # probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score) # roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))

    ### plotting
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    
    # plt.show()
    file_name = kargs.get('file_name', 'roc')
    saveFig(plt, PerformanceMetrics.plot_path(name=file_name), dpi=300)

    return

def plot_roc_crossval(X, y, classifier, **kargs):
    """

    **kargs
    -------
    cv 
    n_fold 
    file_name

    Memo
    ----
    1.  Run classifier with cross-validation and plot ROC curves
            cv = StratifiedKFold(n_splits=6)
            classifier = svm.SVC(kernel='linear', probability=True,
                            random_state=random_state)
    """
    from scipy import interp
    from sklearn.metrics import roc_curve, auc 
    from sklearn.model_selection import StratifiedKFold

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    cv = kargs.get('cv', StratifiedKFold(n_splits=kargs.get('n_fold', 5)))  # 
    for i, (train, test) in enumerate(cv.split(X, y)):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))

    ### plotting
    plt.clf()

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    
    # plt.show()
    file_name = kargs.get('file_name', 'roc')
    saveFig(plt, PerformanceMetrics.plot_path(name=file_name), dpi=300)

    return

def runENetStacker(n_classifiers=5, n_bags=10, seed=0, fold=0, per_fold=True): 
    """
    
    Memo
    ----
    1. Alternatively, use stacking module. 
    """
    def getXY(ts, label='label'):
        X = ts.drop(label, axis=1).values
        y = ts[label].values 
        return (X, y)
    def get_ts(l1_dir, split='valid'): # split: {'valid', 'test', }
        if per_fold: 
            fpath = os.path.join(l1_dir, '%s-nbp%d-f%i-s%i.csv' % (split, nBP, fold, seed))
        else:
            fpath = os.path.join(l1_dir, '%s-nbp%d-s%i.csv' % (split, nBP, seed))

        assert os.path.exists(fpath), "Invalid path: %s" % fpath
        ts = pd.read_csv(fpath, sep=',', header=0, index_col=False, error_bad_lines=True)
        return ts

    from sklearn.metrics import roc_curve, roc_auc_score

    # input
    nBP = n_classifiers * n_bags
    project_path = '/Users/chiup04/Documents/work/data/diabetes_cf'
    l1_dir = os.path.join(project_path, 'LEVEL1')
    
    # naming: number of base predictors, seed number
    ts = get_ts(l1_dir, split='valid')
    print('runENetStacker> input level-1 data dim: %s' % str(ts.shape))

    # [todo] pass model
    model = visualize_coeffs(ts, file_name='sgd_enet_coefficients-s%d' % seed)

    # load test set
    ts = get_ts(l1_dir, split='test')
   
    X_test, y_test = getXY(ts, label='label')
    # model.predict(X_test)

    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    fmax = fmax_score(y_test, y_pred, beta = 1.0, pos_label = 1)
    print('runENetStacker> fmax: %f, AUC score: %f' % (fmax, auc_score))

    ##Computing false and true positive rates
    # fpr, tpr,_=roc_curve(model.predict(X_test), y_test, drop_intermediate=False)

    # plt.figure(1)
    plt.clf()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='ENet')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')

    # test 
    # [todo] customize classifier
    # params = {'penalty': 'elasticnet', 'alpha': 0.001, 'loss': 'modified_huber', 'fit_intercept': True, 'tol': 1e-3}
    # model = SGDClassifier(**params)
    # model.fit(X_train, y_train)  # predict(X_test)

    file_name = 'roc-enet_stacker-nbp%d-f%d-s%d' % (nBP, fold, seed)
    if per_fold: 
        file_name = 'roc-enet_stacker-nbp%d-s%d' % (nBP, seed)
    saveFig(plt, plot_path(name=file_name), dpi=300)
    
    return

def t_metrics(): 
    import numpy as np
    
    M = Metrics(op=np.mean)

    for i in range(100): 
        n = np.random.choice(range(10), 1)[0]
        v = np.random.choice(range(100), 1)[0]
        # np.random.choice(range(10), n)
        M.add( (n, v) )

    for k, v in M.records.items():
        print('[key=%s] %s' % (k, v)) 
    print('....... size: %d, size(bags): %d' % (M.size(), M.size_bags()))

    print('... sort by frequencies:\n%s\n' % M.sort_by_freq())

    Mp = M.aggregate()
    print('... aggregated => each entry is reduced to a single value')

    for k, v in Mp.items():
        print('[key=%s] %s' % (k, v)) 

    print('>>> construct a multibag first  ...')
    # construct a multibag first 
    bags = {}
    for i in range(100): 
        n = np.random.choice(range(10), 1)[0]
        v = np.random.choice(range(100), 1)[0]
        if not n in bags: bags[n] = []
        bags[n].append(v)

    M = Metrics(bags, op=np.median)
    print('....... size: %d, size(bags): %d' % (M.size(), M.size_bags()))

    Mp = Metrics(bags, op=np.median).aggregate()
    for k, v in Mp.items():
        print('[key=%s] %s' % (k, v)) 
    

    return

def estimateLabelMatrix(R, L=[], p_th=[], pos_label=1, neg_label=0, ratio_small_class=0.01):
    """
    Similar to estimateLabels() but compute estimated labels per classifier (i.e per row)
    instead of estimating a single label set (through mean predictions or other aggregate methods). 
    """ 
    ### verify
    k = -1
    policy = 0  # dummy method p_th = 0.5
    if (hasattr(p_th, '__iter__') and len(p_th) > 0):
        policy = 'thresholding' # classifier-specific probability thresholding (e.g. p_th ~ fmax)
        assert R.shape[0] == len(p_th)
    elif isinstance(p_th, float): # heuristic, unsupervised but not practical
        policy = 'thresholding'
        p_th = [p_th, ] * R.shape[0]
    elif len(L) > 0:  # top k probability thresholding (highest probabilities => positive)
        # use L to estimate proba thresholds
        policy = 'prior'  # use L to estimate class ratio, from which to estimate p_th
    else: 
        policy = 'ratio'


        # use case: 
        #    R can be the rating matrix from training split or test split 
        #    L must be from the training split
        
        # use L to estimate k, which is then used to estimate prob threshold

        # assert R.shape[1] == len(L)
        # k = np.sum(L == pos_label)  # top k probabilities => positive
    # print('(estimateLabelMatrix) policy: {}'.format(policy))

    ### computation starts here
    Lh = np.zeros(R.shape).astype(int)
    if policy.startswith('thr'): 
        # print("(estimateLabelMatrix) policy: 'thresholding' > thresholds pre-computed:\n{0}\n".format(p_th))
        for i in range(R.shape[0]):  # foreach user/classifeir
            cols_pos = R[i] >= p_th[i]
            Lh[i, cols_pos] = pos_label
    elif policy.startswith(('pri', 'top')):
        assert len(L) == R.shape[1] 
        print("(estimateLabelMatrix) policy: 'prior' > use the top k probs as a gauge for positives...")

        p_th = estimateProbThresholds(R, L=L, pos_label=pos_label, policy='prior')
        for i in range(R.shape[0]):  # foreach user/classifeir
            cols_pos = R[i] >= p_th[i]
            Lh[i, cols_pos] = pos_label
        # k = np.sum(L == pos_label)  # top k probabilities => positive
        
        # for i in range(R.shape[0]):  # foreach user/classifier
        #     # use topk prob scores to determine the threshold
        #     p_th = R[i, np.argsort(R[i])[:-k-1:-1]][-1]  # min of top k probs
        #     cols_pos = R[i] >= p_th
        #     Lh[i, cols_pos] = pos_label

        # may need to return the resulting thresholds
    elif policy == 'ratio': 
        # ratio_small_class estimated externally (e.g. from training set R)
        assert ratio_small_class > 0, "Minority class ratio must be > 0 in order to get an estimate of top k probabilities."
        p_th = estimateProbThresholds(R, L=[], pos_label=pos_label, policy='ratio', ratio_small_class=ratio_small_class)

        for i in range(R.shape[0]):  # foreach user/classifeir
            cols_pos = R[i] >= p_th[i]
            Lh[i, cols_pos] = pos_label
    else: 
        median_predictions = np.median(R, axis=0)
        mean_predictions = np.mean(R, axis=0)
        for j in range(R.shape[1]): 
            if median_predictions[j] > mean_predictions[j]:  # median > mean, skew to the left (mostly high probabilities)
                L_est[i] = pos_label

    return Lh

def estimateProbThresholds(R, L=[], pos_label=1, neg_label=0, policy='prior', ratio_small_class=0.01):
    """

    Memo
    ----

    1. suppose a = [0, 1, 2, ... 9] 
       a[:-5:-1] => the last 4 (-1, -2, -3, -4) elements ~> 9, 8, 7, 6 

       the last k elements 
       a[:-k-1: -1], which then include the last kth element

       a[:-3]: a[0] up to but not include the last 3rd element 
              => [0, 1, 2, 3, 4, 5, 6]

       a[:-3: -1]: counting from the back, up to but not include the last 3rd element 
               => [9, 8]  (excluding 7)

    1a. select columsn and rows by conditions 

            >>> a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
            >>> a
            array([[ 1,  2,  3,  4],
                   [ 5,  6,  7,  8],
                   [ 9, 10, 11, 12]])

            >>> a[a[:,0] > 3] # select rows where first column is greater than 3
            array([[ 5,  6,  7,  8],
                   [ 9, 10, 11, 12]])

            >>> a[a[:,0] > 3][:,np.array([True, True, False, True])] # select columns
            array([[ 5,  6,  8],
                   [ 9, 10, 12]])

            # fancier equivalent of the previous
            >>> a[np.ix_(a[:,0] > 3, np.array([True, True, False, True]))]
            array([[ 5,  6,  8],
                   [ 9, 10, 12]])

    """
    if not policy: 
        if len(L) > 0:   
            policy = 'prior'  
        else: 
            policy = 'ratio'  # unsupervised, given only an estimted ratio of the minority class (typically positive class)  

    thresholds = []
    if policy.startswith( ('prior', 'top' )):  # i.e. top k probabilities as positives where k = n_pos examples in the training split
        assert len(L) > 0, "Labels must be given in 'prior' mode (in order to estimate the lower bound of top k probabilities for positive class)"
        k = np.sum(L == pos_label)  # top k probabilities => positive 
        for i in range(R.shape[0]):  # foreach user/classifier
            # use topk prob scores to determine the threshold
            p_th = R[i, np.argsort(R[i])[:-k-1:-1]][-1]  # min of top k probs
            thresholds.append(p_th)
    elif policy == 'fmax':
        print("(estimateProbThresholds) policy: fmax")
        from evaluate import findOptimalCutoff   #  p_th <- f(y_true, y_score, metric='fmax', pos_label=1)
        thresholds = findOptimalCutoff(L, R, metric=policy, beta=1.0, pos_label=1) 
    
    elif policy == 'ratio':  # user provided an estimated ratio of the minority class, perhaps more conservative than 'prior' 
        assert ratio_small_class > 0, "'ratio_small_class' was not set ... "
        k = math.ceil(ratio_small_class * R.shape[1])  # find k candidates in the item direction
        for i in range(R.shape[0]):  # foreach row/user/classifier
            
            cols_high = np.argsort(R[i])[:-k-1: -1]  # indices of highest probabilities (these are likely to be better probability estimates of the positive)
            p_th = R[i, cols_high][-1] # min of the largest probs
            thresholds.append(p_th)

            ### another thresholds for the lowest probs (negative classe)
            # cols_low = np.argsort(R[i])[:k]  # column indices of the lowest probs
            # p_th_lower = R[i, cols_low][-1] # max of the lowest probs
            # thresholds_lower.append(p_th_lower)
    assert len(thresholds) == R.shape[0], "Expecting {n} p-thresholds (= n_users/n_classifiers) but got {np}".format(len(thresholds), R.shape[0])
    return thresholds

def estimateLabels(T, L=[], p_th=[], Eu=[], pos_label=1, neg_label=0, M=None, labels=[], policy='', ratio_small_class=0, joint_model=None):
    """
    Estimate the labels based on the input rating matrix. 

    Params
    ------
    M: message from the training split (R); used to estimate P(y=1|Lh)
    joint_model: a joint model for answering queries P(y=1|Lh) computed externally

    Memo
    ----
    1. collections.Counter(Lh[:, i]) returns a list of 2-tuples: (label, count)

    2. Pass T and p_th estimated externally

    3. logic chain 
       class prior (or external estimate of class ratio) -> p-thresholds -> Lh -> lh
    """
    import operator
    # import collections
    # probs = np.mean(T, axis=0)  
    ### verify 
    if not policy:  
        # precedence: proba thresholds -> labels -> ratio_small_class -> other unsupervised methods (e.g. mean vs median)
        if hasattr(p_th, '__iter__') and len(p_th) > 0: # supervised (using fmax(L) to pre-compute prob threshold)
            policy = 'thresholding'
            assert len(p_th) == T.shape[0], "Each user/classifier has its own threshold. n(p_th): %d but n_users: %d" % (len(p_th), T.shape[0])
        elif isinstance(p_th, float): # heuristic, unsupervised but not practical
            policy = 'thresholding'
            p_th = [p_th, ] * T.shape[0]
        elif len(L) > 0:  # weakly supervised (using L to determine prob threshold)
            # p_th = estimateProbThresholds(R, L=L, pos_label=pos_label, policy='ratio', ratio_small_class=ratio_small_class)
            p_th = estimateProbThresholds(T, L=L, pos_label=pos_label, policy='prior')
            policy = 'thresholding'
        else: # Neither 'p_th' nor 'L' was given ... 
            # policy = 'mean-median' # does not work well 
            policy = 'ratio'

    tHasMessageFromTrainingSplit = False
    if M is not None or joint_model is not None: 
        # X_train, L_train, *rest = M
        tHasMessageFromTrainingSplit = True
        policy = 'joint'
    
    ### label estimate starts here
    L_est = np.zeros(T.shape[1]).astype(int)
    if policy.startswith('thr'):  # given the probability thresholds (a vector in which each component represents a classifier-specific prob threshold, lg than which is positive, otherwise negative)
        # p_th is a vector/numpy array
        # print("(estimateLabels) Using 'majority vote' given proba thresholds ...")

        ######################################
        # Todo: 
        if len(Eu) > 0: # excluse the votes from those classifiers that did not produce any correct predictions
            pass
        ######################################

        for j in range(T.shape[1]): 
            # majority vote of the user/classifiers for data j is True => positive
            if collections.Counter(T[:, j] >= p_th).most_common(1)[0][0]: # if it's that majory vote says positive
                L_est[j] = pos_label

    # elif policy.startswith('f'): # p_th is a fixed probability score (e.g. 0.5)
    #     L_est[np.where(np.median(T, axis=0) >= p_th)] = pos_label
    else:  # p: None or [] => use mean prediction as an estimate
        # print("(estimateLabels) policy: 'prior' > use the top k probs as a gauge for positives...")
        if policy.startswith( ('pri', 'top') ): ### weakly supervised
            assert len(L) > 0, "Need labels (L) to use this mode; if not, use 'ratio' instead by providing an estimate of the ratio of the minority class via 'ratio_small_class' ... "
            # need to estimate the label per classifier => label matrix
            p_th = estimateProbThresholds(T, L=L, pos_label=pos_label, policy='prior')

            for j in range(T.shape[1]): 
                # majority vote of the user/classifiers for data j is True => positive
                if collections.Counter(T[:, j] >= p_th).most_common(1)[0][0]: # if it's that majory vote says positive
                     L_est[j] = pos_label

        elif policy.startswith('joint'):
            # the problem of this strategy is that P(y=1 | lv ) is probably 0 even if the queried prediction vector (pv) matches lv
            # => pv matching lv is too weak of a condition to conclude a case for a minority class
            # => need to 're-calibrate' the probability

            print("(estimateLabels) Estimate labels via 'joint model' (P(y=1|Lh)) ...")
            assert len(p_th) > 0, "We must assume that proba thresholds have been estimated prior to this call"
            assert M is not None, "Need access to the training data info to proceed (e.g. class prior, joint model)"
            
            r_min = r_max = 0.0
            #####################################
            # jointModel = joint_model
            # if jointModel is None: 
            X_train, L_train, *rest = M
            # if p_th is given, then use it; otherwise, use what 'policy_threshold' specifies to estimate p_th, from which to estimate Lh
            jointModel = estimateProbaByLabelMatrix(X_train, L_train, p_th=p_th, pos_label=1) # policy_threshold='prior' 
            #####################################
            ret = classPrior(L_train, labels=[0, 1], ratio_ref=0.1, verbose=False)
            r_min, r_max = ret['r_min'], ret['r_max']
            min_class, max_class = ret['min_class'], ret['max_class']
            #####################################
            Lh = estimateLabelMatrix(T, L=[], p_th=p_th)
            
            # now make queries into jointModel
            ulabels = [0, 1]
            # for i, (lv, u) in enumerate(jointModel.items()): 
            #     ulabels.update(u.keys())
            #     if i > 10: break 

            # L_est = np.zeros(T.shape[1]).astype(int)
            n_min = int(T.shape[1] * r_min) 
            print('... Expecting to observe {n} minority class examples in T (n={N}) | min_class: {cls}'.format(n=n_min, N=T.shape[1], cls=min_class))

            n_unmodeled = n_unmodeled_pos = n_unmodeled_neg = 0
            for j in range(T.shape[1]):
                lv = tuple(Lh[:, j]) 
                if lv in jointModel: 
                    # print('... test: {value}'.format(value=jointModel[lv]))
                    Py = {label: jointModel[lv][label] for label in ulabels}
                    
                    L_est[j] = max(Py.items(), key=operator.itemgetter(1))[0] # the label with higher proba is the estimated label
                    # ... chances are that the argmax leads to the majority class 
                
                    # [test]
                    if j < 10: 
                        print('... Py: {dict} | argmax: {max}'.format(dict=Py, max=L_est[j]))
                else: 
                    # reverse back to majority vote
                    n_unmodeled += 1 
                    if collections.Counter(T[:, j] >= p_th).most_common(1)[0][0]: 
                        L_est[j] = pos_label
                        n_unmodeled_pos += 1
                    else: 
                        n_unmodeled_neg += 1
            print('... (debug) n_unmodeled: {n0}, n_unmodeled_pos: {n1}, n_unmodeled_neg: {n2}'.format(n0=n_unmodeled, n1=n_unmodeled_pos, n2=n_unmodeled_neg))
            # [observation] 
            #   ... unmodeled cases are relatively less frequent

            ##################################### 
            # ... calibration
            n_min_prior_calib = sum(L_est == min_class) 
            print('... Found only {n} estimated minority class instances prior to calibration!'.format(n=n_min_prior_calib))
            if n_min_prior_calib < n_min: 
                min_index_proba = collections.Counter()
                for j in range(T.shape[1]):
                    lv = tuple(Lh[:, j]) 
                    if lv in jointModel: 
                        Py = {label: jointModel[lv][label] for label in ulabels}
                        
                        if Py[min_class] > 0.0: # [todo] use min-max heap to reduce MEM 
                            min_index_proba[j] = Py[min_class] # append( (j, Py[min_class]) )
                    
                    else: 
                        # indices associated with the majority vote is not subject to re-calibration
                        pass
                n_min_eff = n_min - n_unmodeled
                print('... (debug) Found {n} indices subject to be re-calibrated'.format(n=n_min_eff))
                candidates = min_index_proba.most_common(n_min_eff)
                print('... top {n} indices and their proba:\n{alist}\n...'.format(n=n_min_eff, alist=candidates[:200]))
                
                for j, proba in candidates: 
                    L_est[j] = min_class
            n_min_post_calib = sum(L_est == min_class)
            print('... Found {n} estimated minority class instances AFTER calibration | expected: {ne}'.format(n=n_min_post_calib, ne=n_min))
  
        elif policy == 'ratio':
            # ratio_small_class estimated externally (e.g. from training set R)
            assert ratio_small_class > 0, "Minority class ratio must be > 0 in order to get an estimate of the lower bound of the top k probabilities."
            p_th = estimateProbThresholds(T, L=[], pos_label=pos_label, policy='ratio', ratio_small_class=ratio_small_class)

            for j in range(T.shape[1]): 
                # majority vote of the user/classifiers for data j is True => positive
                if collections.Counter(T[:, j] >= p_th).most_common(1)[0][0]: # if it's that majory vote says positive
                     L_est[j] = pos_label

        elif policy.startswith(('uns', 'mean-m')): ### unsupervised 
            median_predictions = np.median(T, axis=0)
            mean_predictions = np.mean(T, axis=0)
            for i in range(T.shape[1]): 
                if median_predictions[i] > mean_predictions[i]:  # median > mean, skew to the left (mostly high probabilities)
                    L_est[i] = pos_label
        else: 
            raise NotImplementedError('Unrecognized policy: %s' % policy)
                
    return L_est 

def classSummaryStats(R, L, labels=[0, 1], ratio_ref=0.1, policy='prior'):
    ret = classPrior(L, labels=labels, ratio_ref=ratio_ref)
    ret['thresholds'] = estimateProbThresholds(R, L=L, pos_label=labels[1], policy=policy) # policy: {'prior', 'fmax'}
    
    n_users, n_items = R.shape[0], R.shape[1]
    assert len(ret['thresholds']) == R.shape[0], "Each user/classifier has a prob threshold."

    return ret

def classPrior(L, labels=[0, 1], ratio_ref=0.1, verify=True, verbose=True):  # assuming binary class
    # import collections 
    if not labels: labels = np.unique(L)
    lstats = collections.Counter(L)
    ret = {} # {l:0.0 for l in labels}
    if len(labels) == 2: # binary class 
        neg_label, pos_label = labels

        ret['n_pos'] = nPos = lstats[pos_label] # np.sum(L==pos_label)
        ret['n_neg'] = nNeg = lstats[neg_label] # np.sum(L==neg_label)
        ret['n_min_class'] = nPos if nPos <= nNeg else nNeg
        ret['n_max_class'] = nNeg if nPos <= nNeg else nPos
        ret['n'] = nPos + nNeg
        if verify: assert len(L) == ret['n'], "n(labels) do not summed to total!"

        ret['r_pos'] = ret[pos_label] = rPos = nPos/(len(L)+0.0)
        # nNeg = len(L) - nPos 
        ret['r_neg'] = ret[neg_label] = rNeg = nNeg/(len(L)+0.0) # rNeg = 1.0 - rPos
        ret['r_min'] = ret['r_minority'] =  min(rPos, rNeg)
        ret['r_max'] = ret['r_majority'] = 1. - ret['r_min']
        ret['min_class'] = ret['minority_class'] = pos_label if rPos < rNeg else neg_label
        ret['max_class'] = ret['majority_class'] = neg_label if rPos < rNeg else pos_label

        ret['r_max_to_min'] = ret['multiple'] = ret['n_max_class']/(ret['n_min_class']+0.0)
        
        if min(rPos, rNeg) < ratio_ref:  # assuming pos labels are the minority
            if verbose: print('(class_prior) Imbalanced class distribution: n(+):{0}, n(-):{1}, r(+):{2}, r(-):{3}'.format(nPos, nNeg, rPos, rNeg))
            ret['is_balanced'] = False

        # print('... class distribution > n_pos: {0} (total: {1}, ratio: {2})'.format(nPos, len(L), rPos))
        # print('...                      n_neg: {0} (total: {1}, ratio: {2})'.format(nNeg, len(L), rNeg))
    else: # single class or multiclass problems
        raise NotImplementedError
    return ret  # keys: n_pos, n_neg, r_pos, r_neg, 0/neg_label, 1/pos_label, is_balanced, r_minority, min_class

def generate_mock_data(shape=(5, 5), p_th=[]):
    def the_other(l):
        return 1 - l
    # test function for other subroutines (e.g. analyzePerf())

    Z = np.zeros(shape) 
    nu, ni = Z.shape[0], Z.shape[1]

    for i in range(nu):
        Z[i, :] = np.random.uniform(0, 1, ni)

    # simulate labels
    if not p_th: 
        p_th = [0.5, ] * nu

    L = estimateLabels(Z, p_th=p_th, L=[], pos_label=1, neg_label=0)
    uL = np.unique(L)
    if len(uL) < 2: 
        ith = np.random.choice(range(ni), 1)[0]
        L[ith] = the_other(uL[0])

    return (Z, L)

def t_performance_metrics(**kargs):
    def tune(L, Th, p_th=0.5, ep=0.01):
        for i in range(Th.shape[0]):
            # foreach row in Th, wherever L==1, set the proba to a reasonable value
            Th[i, np.where(L==1)] = np.random.uniform(p_th+ep, 1.0-ep, sum(L==1))
        return Th 

    from tabulate import tabulate
    np.set_printoptions(precision=3)

    domain = config.domain
    analysis_path = config.analysis_path
    # if not os.path.exists(project_path): 
    #     os.mkdir(project_path)
    # if not os.path.exists(analysis_path): 
    #     os.mkdir(analysis_path)

    dim = (10, 5)
    ###################################
    params = {'n_factors': 100, 'alpha': 100, 'conf_measure': 'brier', 'setting': 3}
    method_id = 'supernova'
    # 'dummy_regression'
    ###################################

    n_cycles = 10
    t = np.random.choice(n_cycles, 1)[0]
    for i in range(n_cycles): 
        T, L = generate_mock_data(dim, p_th=[])
        Th, _ = generate_mock_data(dim, p_th=[])

        ########################################
        # perfect Th assuming p_th = 0.5
        Th2, _ = generate_mock_data(dim, p_th=[])
        tune(L, Th2)
        ########################################

        # print('> T:\n{T}\n> L:\n{L}\n------'.format(T=T, L=L))
        # print('> Th:\n{T}\n------'.format(T=Th))
        perf, pv = analyzePerf(L, Th=Th, T=T, method=method_id, aggregate_func=np.mean, fold=i, 
                    save=False, analysis_path=analysis_path) # True if i == t else False
        print('> Cycle {c} | perf:\n{t}\n'.format(c=i, t=tabulate(perf.table, headers='keys', tablefmt='psql')))

        div("Do we always see positive increments between score and score_h if Th is 'perfect'? ")  # ... ok
        perf2, pv2 = analyzePerf(L, Th=Th2, T=T, method=method_id, aggregate_func=np.mean, fold=i, 
                    save=False, analysis_path=analysis_path) # True if i == t else False
        print('> Cycle {c} | perf2:\n{t}\n'.format(c=i, t=tabulate(perf.table, headers='keys', tablefmt='psql')))

    return 

def test(): 

    # stacker via logistic regression with elastic net penalty
    # for per_fold in [True, False]: 
    #     for fold in range(5): 
    #         runENetStacker(n_classifiers=5, n_bags=10, seed=0, fold=fold, per_fold=per_fold)

    # class Metrics 
    # t_metrics()

    # performance metrics and class PerformanceMetrics
    t_performance_metrics()

    return

if __name__ == "__main__": 
    test()



