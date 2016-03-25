from training_matrix import *
from classifier import *
from entropy import *

#read in file number of files for file likelihood
train_doc_total = 11269
#number of words
v_total = 61188

#values for direlecht distribution
beta = 1/(v_total)
alpha = (1+beta)


##code to find the likelihood p_y
#p_y = [0] * 20
#with open('data/train.label') as f:
    #for line in f:
        #line = line.replace("\n","")
        #line = int(line)-1
        #p_y[line] = p_y[line] + 1

#p_y[:] = [y/float(train_doc_total) for y in p_y]

#just hard coded after I found the likelihood for each label because I'm a terrible programmer
p_y = np.array([0.04259472890229834, 0.05155736977549028, 0.05075871860857219, 0.05208980388676901, 0.051024935664211554, 0.052533498979501284, 0.051646108794036735, 0.052533498979501284, 0.052888455053687104, 0.0527109770165942, 0.05306593309078002, 0.0527109770165942, 0.05244475996095483, 0.0527109770165942, 0.052622237998047744, 0.05315467210932647, 0.04836276510781791, 0.05004880646020055, 0.04117490460555506, 0.033365870973467035])


#read in the training data
print("reading the traing data...")
print("")
tm = training_matrix()

nb_c = classifier(tm, p_y)

#classifying... really slow because I'm a terrible programmer
print("classifying... this may take a while on UNM machines. It takes like 15s on my personal laptop, but it took like 2 mins on campus")
print("")
#make p_x_given_y matrix
nb_c.makeMAPMatrix(alpha)
#classify
nb_c.classify()
print("accuracy: ")
print(nb_c.calculateAccuracy(True))

print("classifiers 100 most important words: ")
ent = entropy(nb_c, p_y, tm)
ent.getTop100()


#code for finding accuracies at 20 different beta values. really slow
#commented out, not recommend to run
'''
b = 0
beta_test = 0
beta_vector = []
accuracy_vector = []


while b != -5:
    beta_test = math.pow(10, b)
    beta_vector.append(beta_test)
    alpha = (1 + beta_test)
    nb_c.makeMAPMatrix(alpha)
    nb_c.classify()
    accur = nb_c.calculateAccuracy(False)
    accuracy_vector.append(accur)
    b = b - 0.25

print(beta_vector)
print(accuracy_vector)
#I just printed out the arrays, copied them from the terminal, and used the semilogx function in MATLAB
#as reccomended by the project guide
'''
