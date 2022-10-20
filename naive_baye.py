import numpy as np 
import math
from tqdm import tqdm
from ngram import NGram

class Nb_ngram:
    def __init__(self, textPreProcessor = None, ngram=1):
        self.nbClass = 0
        self.isCompile = False
        self.isTrain = False
        self.BoT = dict()
        self.classesProb = []
        self.nGram = NGram(ngram)
        self.textPreProcessor = textPreProcessor
    
    def get_classes_occurences(self, Y):
        classes = dict()
        for cl in Y:
            if cl not in classes:
                classes[cl] = 0
            classes[cl] += 1
        return classes
        
    def compile(self, X, Y):
        self.X = X
        self.Y = Y
        self.classes = np.unique(Y)
        self.nbClass = len(self.classes)
        
        if self.textPreProcessor != None:
            print("Preprocessing text:")
            for i, sen in enumerate(tqdm(self.X)):
                self.X[i] = self.textPreProcessor.cleanText(sen)

        def create_bag_of_word(X, Y):
            bags_of_ngram = dict();
            for i in self.classes:
                bags_of_ngram[i] = dict()
            print("Creating bags of words...")
            for lab, sen in tqdm(zip(Y, X)):
                ngram_sentence = self.nGram.ngram(sen)
                for t in ngram_sentence:
                    if t not in bags_of_ngram[lab]:
                        bags_of_ngram[lab][t] = 0
                    bags_of_ngram[lab][t] += 1
            return bags_of_ngram

        self.BoT = create_bag_of_word(self.X, self.Y)

    def get_classes_probabilites_log(self, Y):
        def get_classes_proba_log(classes, nb_samples):
            classes_occ = dict()
            for cl, occ in classes.items():
                classes_occ[cl] = math.log(float(occ) / float(nb_samples))
            return classes_occ
        
        self.classes_proba = get_classes_proba_log( classes = self.get_classes_occurences(Y), nb_samples = len(Y) )

    def train(self):
        self.words_by_classes = dict();
        self.vocab = dict()
        print("Extracting vocab:")
        for cl, dic in self.BoT.items():
            if cl not in self.words_by_classes:
                self.words_by_classes[cl] = 0
            for tok, val in tqdm(dic.items()):
                self.words_by_classes[cl] += val
                self.vocab[tok] = 1
        print(len(self.vocab))
        self.vocab_len = len(self.vocab)

        self.get_classes_probabilites_log(self.Y)
        
        self.denominators = dict()
        print("Calculating classes denominators for probabilites:")
        for cl in tqdm(self.classes):
            self.denominators[cl]  = self.words_by_classes[cl] + self.vocab_len + 1
        
        self.Y_info = [(self.BoT[cl], self.classes_proba[cl], self.denominators[cl]) for cl in self.classes] 
        self.Y_info = np.array(self.Y_info) 

    def predict(self, text):
        likelihood_prob = np.zeros(self.classes.shape[0])
        if self.textPreProcessor != None:
            text = self.textPreProcessor.cleanText(text)
        for cl_i, cl in enumerate(self.classes):                 
            for tok in self.nGram.ngram(text):     
                if tok in self.vocab:                   
                    tok_counts = self.Y_info[cl_i][0].get(tok, 0) + 1 # We add 1 due to the formula to not get 0 probabilities                        
                    tok_prob = tok_counts/float(self.Y_info[cl_i][2])                              
                    likelihood_prob[cl_i] += math.log(tok_prob)
                                                
        post_prob = np.empty(self.classes.shape[0])
        for cl_i, cl in enumerate(self.classes):
            post_prob[cl_i] = likelihood_prob[cl_i] + self.Y_info[cl_i][1]                              
        
        return post_prob