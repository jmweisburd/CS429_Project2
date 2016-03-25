import numpy as np

v_total = 61188
l_total = 20

'''
training_matrix is just data class I use to read in the training data into a 20x61188 matrix and to count
how many words each label has. This data is passed eventually to the classifier to make the P(X|Y) matrix through MAP
estimates
'''
class training_matrix:

    def __init__(self):
        #rows labels   cols  words
        self.t_mat = np.zeros((l_total, v_total))
        self.readTrainingData()
        self.sum_word_y = np.sum(self.t_mat, axis =1) #sum up how many words under each newsgroup label

    #Function which reads in the  data from training.data
    #Saves the data the t_mat array
    def readTrainingData(self):
        with open('data/train.data') as f:
            for line in f:
                a = line.split()
                a = self.parseLine(a)
                doc_type = self.lookupDocType(a[0])-1
                self.t_mat[doc_type, a[1]-1] += a[2]

    #helper function which takes a string and returns a list of ints found in the string
    #input: string containing ints
    #output: list of ints in string
    def parseLine(self, ls):
        ls[:] = [int(y) for y in ls]
        return ls

    #terrible way to look up the label of a training document without reading a file
    #or using much memory because I'm a terrible programmer
    #input: training document id
    #output: label of inputed training document
    def lookupDocType(self, doc_id):
        if doc_id >= 1 and doc_id < 481:
            return 1
        elif doc_id >= 481 and doc_id < 1062:
            return 2
        elif doc_id >= 1062 and doc_id < 1634:
            return 3
        elif doc_id >= 1634 and doc_id < 2221:
            return 4
        elif doc_id >= 2221 and doc_id < 2796:
            return 5
        elif doc_id >= 2796 and doc_id < 3388:
            return 6
        elif doc_id >= 3388 and doc_id < 3970:
            return 7
        elif doc_id >= 3970 and doc_id < 4562:
            return 8
        elif doc_id >= 4562 and doc_id < 5158:
            return 9
        elif doc_id >= 5158 and doc_id < 5752:
            return 10
        elif doc_id >= 5752 and doc_id < 6350:
            return 11
        elif doc_id >= 6350 and doc_id < 6944:
            return 12
        elif doc_id >= 6944 and doc_id < 7544:
            return 13
        elif doc_id >= 7544 and doc_id < 8129:
            return 14
        elif doc_id >= 8129 and doc_id < 8722:
            return 15
        elif doc_id >= 8722 and doc_id < 9321:
            return 16
        elif doc_id >= 9321 and doc_id < 9866:
            return 17
        elif doc_id >= 9866 and doc_id < 10430:
            return 18
        elif doc_id >= 10430 and doc_id < 10894:
            return 19
        else:
            return 20
