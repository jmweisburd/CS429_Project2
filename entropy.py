import numpy as np
from classifier import *
import math

v_total = 61188
l_total = 20

#entropy equation
#input: real, positive number
#output: entropy of number
def entropy_eq(n):
    return (-n)*(math.log(n,2))

'''
The entropy class is responsible for calculating the entropy of each word.
It does so by using the conditional entropy equation detailed in my project report:
H(Y|X) = -sum(p(X|Y)*log(P(X|Y)))
Also detailed at: https://www.cs.unm.edu/~williams/cs530/mutual2.pdf
'''
class entropy:

    def __init__(self, nb_c, p_y, tm):
        self.f = np.vectorize(entropy_eq, otypes=[np.float]) #vectorize a function

        self.t_mat = tm.t_mat

        self.p_X = np.sum(self.t_mat, axis = 0)
        self.p_X = self.p_X + (1+(1/v_total) #add hallucinated counts
        self.p_X = self.p_X / (np.sum(self.p_X))

        self.p_X_given_Y = np.copy(nb_c.p_X_given_Y) #p(X|Y) matrix copied from the MAP estimates
        self.p_Y_given_X = None #matrix we are trying to calculate

        self.p_y = p_y.reshape((20,1)) #reshape so we can multiply it with P(X|Y) matrix

        self.fillpYgivX()

        self.entropy_array = np.zeros((v_total))
        self.word_array = []
        self.top_100_words = []
        self.readVocabulary() #read in the vocabulary list
        self.calculateEntropy()

    #function to calculate p(y|x) matrix
    def fillpYgivX(self):
        #self.p_Y_given_X = self.p_X_and_Y / self.p_X #p(y|x) = p(x,y)/p(x)
        self.p_X_given_Y = self.p_y * self.p_X_given_Y
        self.p_Y_given_X = self.p_X_given_Y / self.p_X

    #function to calculate the entropy of each label y given word x
    #sums of the entropies of each word into total entropies
    def calculateEntropy(self):
        self.p_Y_given_X = self.f(self.p_Y_given_X) #apply entropy equation to matrix
        self.entropy_array = np.sum(self.p_Y_given_X, axis=0) #sum up entropies of each word for a given y into length 61188 array, holds entropy of each word

    #function to read in all of the vocabular words into an array
    def readVocabulary(self):
        with open('data/vocabulary.txt') as f:
            for line in f:
                self.word_array.append(line.rstrip())

    #function to get the 100 words with the lowest entropy
    def getTop100(self):
        top_100_index = self.entropy_array.argsort()[:100] #get indicies of words with lowest entropies

        for i in range(len(top_100_index)):
            self.top_100_words.append(self.word_array[top_100_index[i]]) #get words from the vocabulary list

        print(self.top_100_words)
