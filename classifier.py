import numpy as np
import math

v_total = 61188
l_total = 20

#def MAP(count_x, total_words_y, alpha_m_1, alpha_t_v):
    #a = (count_x + alpha_m_1)/(total_words_y + alpha_t_v)
    #if a == 0:
        #return 0
    #else:
        #return math.log(a, 2)

def MAP(count_x, total_words_y, alpha_m_1, alpha_t_v):
    return (count_x + alpha_m_1)/(total_words_y + alpha_t_v)

class classifier:

    def __init__(self, training_matrix, likelihood):
        self.likelihood = likelihood
        self.log_likelihood= np.zeros((l_total))
        self.makeLogLikelihood()
        self.t_mat = training_matrix.t_mat
        self.sum_word_y = training_matrix.sum_word_y
        self.class_mat = np.zeros((l_total, v_total))
        self.class_doc_vector = []
        self.conf_mat = np.zeros((l_total, l_total))

    #Used for Entropy Calculations
    def makeMAPMatrix(self, alpha):
        alpha_m_1 = (alpha-1) #alpha-1
        alpha_t_v = (alpha_m_1 * v_total) #(alpha-1) * length of vocab
        self.makeMAPMatrixHelper(alpha_m_1, alpha_t_v)

    def resetClassMat(self):
        self.class_mat = np.zeros((l_total, v_total))

    #Used for Classification Calculations
    def makeLogMAPMatrix(self, alpha):
        alpha_m_1 = (alpha-1)
        alpha_t_v = (alpha_m_1 * v_total)
        self.makeLogMAPMatrixHelper(alpha_m_1, alpha_t_v)

    def makeMAPMatrixHelper(self, alpha_m_1, alpha_t_v):
        for i in range(l_total):
            total_word_y = self.sum_word_y[i]
            for j in range(v_total):
                self.class_mat[i,j] = MAP(self.t_mat[i,j], total_word_y, alpha_m_1, alpha_t_v)

    def makeLogMAPMatrixHelper(self, alpha_m_1, alpha_t_v):
        for i in range(l_total):
            total_word_y = self.sum_word_y[i]
            for j in range(v_total):
                n = MAP(self.t_mat[i,j], total_word_y, alpha_m_1, alpha_t_v)
                if n > 0:
                    self.class_mat[i,j] = math.log(n, 2)
                else:
                    self.class_mat[i,j] = 0

    def makeLogLikelihood(self):
        for i in range(l_total):
            self.log_likelihood[i] = math.log(self.likelihood[i], 2)

    def pXsFor1Line(self, doc_vector, line_vector):
        word_id = (line_vector[1])-1
        word_count = line_vector[2]
        for i in range(l_total):
            doc_vector[i] += (word_count * self.class_mat[i, word_id])

    def addLogLikelihood(self, doc_vector):
        for i in range(l_total):
            doc_vector[i] += self.log_likelihood[i]

    def parseLine(self, ls):
        ls[:] = [int(y) for y in ls]
        return ls

    def classify(self):
        doc_id = 1
        doc_vector = np.zeros((l_total))
        with open('data/test.data') as f:
            for line in f:
                line_vector = line.split()
                line_vector = self.parseLine(line_vector)
                if line_vector[0] != doc_id:
                    self.addLogLikelihood(doc_vector)
                    index_max = doc_vector.argmax()
                    self.class_doc_vector.append(index_max+1)
                    doc_vector[:] = 0
                    doc_id += 1
                    self.pXsFor1Line(doc_vector, line_vector)
                else:
                    self.pXsFor1Line(doc_vector, line_vector)

        self.addLogLikelihood(doc_vector)
        index_max = doc_vector.argmax()
        self.class_doc_vector.append(index_max+1)


    def calculateAccuracy(self, confusion):
        right = 0
        current_index = 0
        with open('data/test.label') as f:
            for line in f:
                line = line.split()
                line = self.parseLine(line)
                class_label = self.class_doc_vector[current_index]-1
                real_label = line[0]-1

                if real_label == class_label:
                    right += 1
                    if confusion == True:
                        self.conf_mat[real_label, class_label] += 1
                else:
                    if confusion == True:
                        self.conf_mat[real_label, class_label] += 1

                current_index += 1

        if confusion == True:
            np.savetxt("confusion_matrix.csv", self.conf_mat, fmt = '%i',)

        return (right/len(self.class_doc_vector))
