import copy
import sys, math, re
from collections import defaultdict

def precook(s, n=4, out=False):
    """Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well."""
    if isinstance(s, list):
        words = s  # Directly use the list if it's already a list of words
    else:
        words = s.split()  # Split the string into words if it's a string
    counts = defaultdict(int)
    for k in range(1, n+1):  # Python 3: xrange -> range
        for i in range(len(words)-k+1):  # Python 3: xrange -> range
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return (len(words), counts)

def cook_refs(refs, eff=None, n=4): 
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.'''

    reflen = []
    maxcounts = {}
    for ref in refs:
        rl, counts = precook(ref, n)
        reflen.append(rl)
        for (ngram,count) in counts.items():  # Python 3: iteritems() -> items()
            maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)

    if eff == "shortest":
        reflen = min(reflen)
    elif eff == "average":
        reflen = float(sum(reflen)) / len(reflen)

    return (reflen, maxcounts)

def cook_test(test, reflen_refmaxcounts, eff=None, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.'''

    testlen, counts = precook(test, n, True)

    result = {}
    
    if eff == "closest":
        result["reflen"] = min((abs(l - testlen), l) for l in reflen_refmaxcounts[0])[1]
    else:
        result["reflen"] = reflen_refmaxcounts[0]

    result["testlen"] = testlen
    result["guess"] = [max(0, testlen - k + 1) for k in range(1, n+1)]  # Python 3: xrange -> range
    result['correct'] = [0]*n
    for (ngram, count) in counts.items():  # Python 3: iteritems() -> items()
        result["correct"][len(ngram) - 1] += min(reflen_refmaxcounts[1].get(ngram, 0), count)

    return result

class BleuScorer(object):
    """Bleu scorer."""

    __slots__ = "n", "crefs", "ctest", "_score", "_ratio", "_testlen", "_reflen", "special_reflen"

    def __init__(self, test=None, refs=None, n=4, special_reflen=None):
        ''' singular instance '''
        self.n = n
        self.crefs = []
        self.ctest = []
        self.cook_append(test, refs)
        self.special_reflen = special_reflen

    def _single_reflen(self, reflens, option=None, testlen=None):
        """Helper method to calculate single reference length"""
        if option == "shortest":
            reflen = min(reflens)
        elif option == "average":
            reflen = float(sum(reflens)) / len(reflens)
        elif option == "closest":
            reflen = min((abs(l-testlen), l) for l in reflens)[1]
        else:
            raise ValueError(f"Unsupported reflen option {option}")
        return reflen
    def cook_append(self, test, refs):
        """Called to append references and test sentences to BleuScorer."""
        if refs is not None:
            # Cook reference sentences
            self.crefs.append(cook_refs(refs))
            if test is not None:
                # Cook test sentence
                cooked_test = cook_test(test, self.crefs[-1])
                self.ctest.append(cooked_test)
            else:
                self.ctest.append(None)  # lens of crefs and ctest have to match

        self._score = None  ## need to recompute
    def compute_score(self, option=None, verbose=0):
        n = self.n
        small = 1e-9
        tiny = 1e-15  ## so that if guess is 0 still return 0
        bleu_list = [[] for _ in range(n)]

        if self._score is not None:
            return self._score

        if option is None:
            option = "average" if len(self.crefs) == 1 else "closest"

        self._testlen = 0
        self._reflen = 0
        totalcomps = {'testlen': 0, 'reflen': 0, 'guess': [0] * n, 'correct': [0] * n}

        for comps in self.ctest:
            testlen = comps['testlen']
            self._testlen += testlen

            if self.special_reflen is None:
                reflen = self._single_reflen(comps['reflen'], option, testlen)
            else:
                reflen = self.special_reflen

            self._reflen += reflen

            for key in ['guess', 'correct']:
                for k in range(n):
                    totalcomps[key][k] += comps[key][k]

            bleu = 1.
            for k in range(n):
                bleu *= (float(comps['correct'][k]) + tiny) / (float(comps['guess'][k]) + small)
                bleu_list[k].append(bleu ** (1. / (k + 1)))
            ratio = (testlen + tiny) / (reflen + small)
            if ratio < 1:
                for k in range(n):
                    bleu_list[k][-1] *= math.exp(1 - 1 / ratio)

        totalcomps['reflen'] = self._reflen
        totalcomps['testlen'] = self._testlen

        bleus = []
        bleu = 1.
        for k in range(n):
            bleu *= float(totalcomps['correct'][k] + tiny) / (totalcomps['guess'][k] + small)
            bleus.append(bleu ** (1. / (k + 1)))
        ratio = (self._testlen + tiny) / (self._reflen + small)
        if ratio < 1:
            for k in range(n):
                bleus[k] *= math.exp(1 - 1 / ratio)

        self._score = bleus
        return self._score, bleu_list

