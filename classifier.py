import numpy as np
import math

v_total = 61188
l_total = 20

#Function to find the MAP estimate of a word given a label y
#input: #of words x, total words under label y, alpha - 1 and (alpha-1)* length of vocab
#output: the MAP estimate of a word given a label y
def MAP(count_x, total_words_y, alpha_m_1, alpha_t_v):
    return (count_x + alpha_m_1)/(total_words_y + alpha_t_v)

'''
This class is responsible for all of the naive bayes logic and calculation. It holds and calculates
the p(x|y) matrix from training data and has the functions necessary to read in and classify testing
documents, specifically test.data and test.label
'''
class classifier:

    def __init__(self, training_matrix, p_Y):
        self.p_Y = p_Y #array of likelihoods calculated in naive_bayes.py
        self.log_p_Y= np.zeros((l_total))
        self.makeLogpY()
        self.t_mat = training_matrix.t_mat #pointer to the t_mat made in training_matrix.py
        self.sum_word_y = training_matrix.sum_word_y #pointer to the list of the sum of word under label y
        self.p_X_given_Y = np.zeros((l_total, v_total)) #p(x|y) matrix
        self.pred_label_vector = [] #vector of predicted labels
        self.test_label_vector = [] #vector of actual labels
        self.conf_mat = np.zeros((l_total, l_total)) #20x20 confusion matrix
        self.readTestLabels()


    #Wrapper function, called to make the p(x|y) matrix
    #input: alpha value used for direlecht distribution
    def makeMAPMatrix(self, alpha):
        alpha_m_1 = (alpha-1) #alpha-1
        alpha_t_v = (alpha_m_1 * v_total) #(alpha-1) * length of vocab
        self.makeMAPMatrixHelper(alpha_m_1, alpha_t_v)

    #Helper function to make the p(x|y) matrix
    #input: alpha-1, (alpha-1)*(vocab length), all values used in the MAP estimate
    def makeMAPMatrixHelper(self, alpha_m_1, alpha_t_v):
        for i in range(l_total):
            total_word_y = self.sum_word_y[i]
            for j in range(v_total):
                #find the MAP estimate for P(X_i|Y_j)
                self.p_X_given_Y[i,j] = MAP(self.t_mat[i,j], total_word_y, alpha_m_1, alpha_t_v)

    #Dumb function to find the logs of all the likelihoods. These values are used
    #during classification
    def makeLogpY(self):
        for i in range(l_total):
            self.log_p_Y[i] = math.log(self.p_Y[i], 2)

    #Function which classifies the data in test/data
    def classify(self):
        doc_id = 1
        #length 20 vector. keeps track of the sum of the probabilities of a doc being label y=k
        doc_vector = np.zeros((l_total))
        #vector to store our predictions in
        self.pred_label_vector = []

        with open('data/test.data') as f:
            for line in f:
                line_vector = line.split()
                line_vector = self.parseLine(line_vector)
                if line_vector[0] != doc_id: #if we're reading a new document
                    self.addLogpY(doc_vector) #add log_p_y element wise to our doc_vector
                    index_max = doc_vector.argmax() #find the index of our maximum probability
                    self.pred_label_vector.append(index_max) #add index to our pred_label_vector
                    doc_vector[:] = 0 #reset doc vector
                    doc_id += 1
                    self.pXsFor1Line(doc_vector, line_vector) #read process the for new document
                else:
                    self.pXsFor1Line(doc_vector, line_vector) #if we're not reading a new doc, just process the line

        #process the final line
        self.addLogpY(doc_vector)
        index_max = doc_vector.argmax()
        self.pred_label_vector.append(index_max+1)


    #function which process a line of data from the data/test.data file
    #input: doc_vector (20 length array which keeps track of the sum of probabilities for a particular doc),
        # line_vector (int list from a line from data/test.data file)
    def pXsFor1Line(self, doc_vector, line_vector):
        word_id = (line_vector[1])-1
        word_count = line_vector[2]
        for i in range(l_total):
            if self.p_X_given_Y[i, word_id] > 0:
                doc_vector[i] += (word_count * math.log(self.p_X_given_Y[i, word_id],2))


    #add the log likelihood to element wise to a doc_vector
    #input: doc_vector (array which keeps trak of the sum of probabilites for a particular doc)
    def addLogpY(self, doc_vector):
        for i in range(l_total):
            doc_vector[i] += self.log_p_Y[i]

    #helper function to turn a string of ints into a list of ints
    #input: string of ints
    #output: list of ints
    def parseLine(self, ls):
        ls[:] = [int(y) for y in ls]
        return ls

    #function to read in the labels of the test documents
    #index of self.test_label_vector is the doc_id
    def readTestLabels(self):
        with open('data/test.label') as f:
            for line in f:
                line = line.split()
                line = self.parseLine(line)
                self.test_label_vector.append(line[0]-1)


    #if passed in True, will fill up a confusion_matrix and save to csv
    #otherwise, just calculates the accuracy of the classification
    #input: True, if user wants to fill up a confusion_matrix and get accuracy
        #False, if user just wants accuracy
    #output: accuracy of classification
    def calculateAccuracy(self, confusion):
        right = 0 #number of docs the classifier correctly identified

        for i in range(len(self.test_label_vector)):
            real_label = self.test_label_vector[i]
            pred_label = self.pred_label_vector[i]
            if real_label == pred_label:
                right += 1
                if confusion == True:
                    self.conf_mat[real_label, pred_label] += 1
            else:
                if confusion == True:
                    self.conf_mat[real_label, pred_label] += 1

        if confusion == True:
            print("Saving confusion_matrix to confusion_matrix.csv")
            print("")
            np.savetxt("confusion_matrix.csv", self.conf_mat, fmt = '%i',)

        return (right/len(self.test_label_vector))
