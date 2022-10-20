class NGram:
    def __init__(self, n=1):
        self.n = n

    def ngram(self, text):
        words = []
        for word in text.split():
            words.append(word)
        temp = zip(*[words[i:] for i in range(0, self.n)])
        ans = [' '.join(n) for n in temp]
        return ans