import numpy as np
from classifier import *
import math

v_total = 61188
l_total = 20

#def entropyLogEq(frac):
    #if frac == 0:
        #return 0
    #else:
        #return (-frac)*(math.log(frac,2))

def entropyLogEq(n):
    if n < 0:
        return 0
    else:
        return (-n) * (math.log(n,2))

class entropy:

    def __init__(self, nb_c, p_y):
        self.p_Y_given_X = nb_c.class_mat
        self.p_y = p_y
        self.p_X_and_Y = np.zeros((l_total, v_total))

        self.multiplybyp_y()
        self.sum_cols = np.sum(self.p_X_and_Y, axis=0)
        self.divideBySum()

        self.entropy_array = np.zeros((v_total))
        self.word_array = []
        self.top_100_words = []
        self.readVocabulary()

    def multiplybyp_y(self):
        for i in range(l_total):
            p_y_i = self.p_y[i]
            for j in range(v_total):
                self.p_X_and_Y[i, j] = p_y_i * self.p_Y_given_X[i,j]

    def divideBySum(self):
        for i in range(l_total):
            sum_i = sum_cols[i]
            for j in range(v_total):
                self.p_Y_given_X[i,j] = self.p_Y_given_X[i,j]/sum_i

    def entropyCalculation(self):
        for i in range(l_total):
            for j in range(v_total):
                n = self.p_Y_given_X[i,j]
                if n <= 0:
                    self.p_Y_given_X[i,j] = 0
                else:
                    self.p_Y_given_X[i,j] = (-n) * (math.log(n,2))

        self.entropy_array = np.sum(self.p_Y_given_X, axis=0)

    def readVocabulary(self):
        with open('data/vocabulary.txt') as f:
            for line in f:
                self.word_array.append(line.rstrip())

    def getTop100(self):
        top_100_index = self.entropy_array.argsort()[:100]
        #top_100_index = (-self.entropy_array).argsort()[:100]
        print(top_100_index)
        for i in range(len(top_100_index)):
            self.top_100_words.append(self.word_array[top_100_index[i]])

        print(self.top_100_words)
