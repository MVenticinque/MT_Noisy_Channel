from collections import defaultdict, Counter
import math


class KneserNeyLM:
    def __init__(self, n, discount=None, unk_token="<unk>", unk_threshold=0):
        """
        n:             order of the top-level model (e.g. 3 for trigram)
        discount:      fixed discount d in [0,1]. If None, 0.75
        unk_threshold: words with training frequency <= this are replaced
                       with unk_token (helps handle OOV at test/decode time).
        """
        assert n >= 1
        self.n = n
        self.discount = discount
        self.unk_token = unk_token
        self.unk_threshold = unk_threshold

        self.bos = "<s>"
        self.eos = "</s>"

        # raw_counts[o], for a given order we get (o-gram -> frequency)
        self.raw_counts = {o: defaultdict(int) for o in range(1, n + 1)}

        # cont_counts[o][o_gram] = number of distinct left-words context
        self.cont_counts = {o: defaultdict(int) for o in range(1, n)}
        # cont_denom[o][context]   = sum_w cont_counts[o][context + (w,)]
        self.cont_denom = {o: defaultdict(int) for o in range(1, n)}

        # n1plus[o][context] = number of distinct words following context
        self.n1plus = {o: defaultdict(int) for o in range(1, n + 1)}

        # context_total[o][context] = sum_w count(context, w)
        self.context_total = {o: defaultdict(int) for o in range(1, n + 1)}

        self.vocab = set()
        self._V = 0

    # ---------------- training ----------------

    def _replace_rare(self, tokenized_sentences):
        # Function to count per word frequency in corpus and substitute rare words (i.e. freq(w) <= unk_tresh) with <unk>

        freq = Counter(w for sent in tokenized_sentences for w in sent) # per word freq
        rare = {w for w, c in freq.items() if c <= self.unk_threshold} # rare word set

        return [
            [self.unk_token if w in rare else w for w in sent]
            for sent in tokenized_sentences
        ]

    def _pad(self, tokens):
        # pads tokenized sentence to the left be coherent with the order of ngram taken into account.
        # Example: n = 2, Sentence: 'the black cat', Result: ['<s>', 'the', 'black', 'cat', '</s>']
        # Pairs:

        # '<s>', 'the'
        # 'the', 'black'
        # 'black', 'cat'
        # 'cat', '</s>'

        return [self.bos] * (self.n - 1) + list(tokens) + [self.eos]

    def train(self, tokenized_sentences):

        """tokenized_sentences: list[list[str]] (lowercased and tokenized)."""

        tokenized_sentences = self._replace_rare(tokenized_sentences)

        # Vocab as BOW set
        self.vocab = {w for sent in tokenized_sentences for w in sent} 
        self.vocab.add(self.eos)
        self.vocab.add(self.unk_token)
        self._V = len(self.vocab) # |V|
        # <s> is context-only hence excluded from vocab

        padded = [self._pad(sent) for sent in tokenized_sentences]

        # raw counts for every order 1 to n
        for sent in padded:
            for o in range(1, self.n + 1): # 1 ... n
                for i in range(len(sent) - o + 1): # 0 ... len - o + 1
                    gram = tuple(sent[i:i + o]) # for o = 1, we have unigrams, o = 2 bigrams etc...
                    if gram[-1] == self.bos:
                        continue  # if last element is <s> skip. We dont predict bos tokens
                    self.raw_counts[o][gram] += 1 # (order, ngram) pair ++

        # ontext totals + n1plus for the top order (from raw_count)
        for gram, c in self.raw_counts[self.n].items():
            context, word = gram[:-1], gram[-1]
            self.context_total[self.n][context] += c
            self.n1plus[self.n][context] += 1

        for o in range(1, self.n): # 1 ... n-1
            higher = self.raw_counts[o + 1] # if o = 1, higher is 2 etc...
            leftwords = defaultdict(set)  # o-gram suffix -> set of left words
            for gram in higher.keys():    # unique types, not tokens
                left, suffix = gram[0], gram[1:] # left = w1, suffix w2, w3 ...
                leftwords[suffix].add(left) # dict with keys: suffix, values: left word(s)
            for suffix, lw in leftwords.items():
                self.cont_counts[o][suffix] = len(lw) # how many unique (o + 1)-grams a specific suffix completes
                context = suffix[:-1] # prefix of the suffix, i.e. ('the', 'black') -> ('the',)
                self.cont_denom[o][context] += len(lw) # how many unique (o + 1)-grams (context + any word) completes
                # Previous example (o=2): number of unique trigrams (left_word, 'the', any_word)
                self.n1plus[o][context] += 1 # number of unique o-grams (e.g. bigrams ('the', any_word)) starting with context

        # heuristic discount if not provided
        if self.discount is None:
            self.discount = 0.75 # value from Church and Gale (1991)

    # ---------------- scoring ----------------

    def _prob(self, word, context, order):
        """Recursive interpolated Kneser-Ney probability at a given order."""
        if order == 0:
            return 1.0 / self._V if self._V > 0 else 1e-10  # last backoff

        ctx = tuple(context[-(order - 1):]) if order > 1 else () # at any order, context of ogram consist of order-1 words. Trigram 'the black cat' -> 'cat' | 'the black'

        if order == self.n: # at top order we want to check how many times ctx + word appears
            count = self.raw_counts[order].get(ctx + (word,), 0) # raw istances
            total = self.context_total[order].get(ctx, 0) # all occurrencies of context
            types = self.n1plus[order].get(ctx, 0) # all unique words after context
        else: # we backed-off, we no longer use raw frequencies but continuation counts. How many times ctx + word completed an o+1gram vs all unique o+1grams completed by ctx + any w
            count = self.cont_counts[order].get(ctx + (word,), 0)
            total = self.cont_denom[order].get(ctx, 0)
            types = self.n1plus[order].get(ctx, 0)

        # recursivelly backoff at lower order
        lower = self._prob(word, context, order - 1) 
        if total == 0:
            return lower  # if context is unseen in this order we backoff to lower

        # discount formula
        d = self.discount
        discounted = max(count - d, 0) / total
        lam = (d * types) / total # lambda in KN formula
        return discounted + lam * lower # KN formula

    def logscore(self, word, context):
        """log2 P(word | context). context = list/tuple of preceding tokens."""
        if word not in self.vocab:
            word = self.unk_token
        context = [w if w in self.vocab or w == self.bos else self.unk_token
                   for w in context]
        p = max(self._prob(word, context, self.n), 1e-12)
        return math.log2(p)

    def score_sentence(self, tokens):
        """
        Total and per-word average log2 probability of a full sentence.

        Returns (total_log2prob, avg_log2prob_per_word).
        """
        padded = [self.bos] * (self.n - 1) + list(tokens) + [self.eos]
        total = 0.0
        for i in range(self.n - 1, len(padded)): # start from self.n - 1 guarantees to have bos as context
            context = padded[i - (self.n - 1):i]
            word = padded[i]
            total += self.logscore(word, context) # log chain rule
        n_words = len(tokens) + 1  # +1 for </s>
        return total, total / n_words


if __name__ == "__main__":
    # tiny test
    toy_corpus = [
        "the black cat sat on the mat".split(),
        "the black dog sat on the rug".split(),
        "a small cat ran".split(),
        "the cat and the dog played".split(),
    ]
    lm = KneserNeyLM(n=3)
    lm.train(toy_corpus)
    print("estimated discount:", lm.discount)
    print("P(cat | the, black) log2:", lm.logscore("cat", ["the", "black"]))
    print("P(zzz | the, black) log2:", lm.logscore("zzz", ["the", "black"]))
    total, avg = lm.score_sentence("the black cat sat".split())
    print("sentence total/avg log2prob:", total, avg)
