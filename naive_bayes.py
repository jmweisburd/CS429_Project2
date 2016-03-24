from training_matrix import *
from classifier import *
from entropy import *

#read in file number of files for file likelihood
train_doc_total = 11269
v_total = 61188

beta = 1/(v_total)
alpha = (1+beta)

alpha_vector = []
accuracy_vector = []


#p_y = [0] * 20
#with open('data/train.label') as f:
    #for line in f:
        #line = line.replace("\n","")
        #line = int(line)-1
        #p_y[line] = p_y[line] + 1

#p_y[:] = [y/float(train_doc_total) for y in p_y]

#just hard coded after I found the likelihood for each label
p_y = [0.04259472890229834, 0.05155736977549028, 0.05075871860857219, 0.05208980388676901, 0.051024935664211554, 0.052533498979501284, 0.051646108794036735, 0.052533498979501284, 0.052888455053687104, 0.0527109770165942, 0.05306593309078002, 0.0527109770165942, 0.05244475996095483, 0.0527109770165942, 0.052622237998047744, 0.05315467210932647, 0.04836276510781791, 0.05004880646020055, 0.04117490460555506, 0.033365870973467035]


tm = training_matrix()
nb_c = classifier(tm, p_y)

#nb_c.makeMAPMatrix(alpha)
#ent = entropy(nb_c, p_y)
#ent.calculateEntropy()
#ent.getTop100()

#nb_c.makeLogMAPMatrix(alpha)
#nb_c.classify()
#print(nb_c.calculateAccuracy(True))

b = 0
beta_test = 0
test = 1

while b != -5:
    print(test)
    test += 1
    beta_test = math.pow(10, b)
    print(beta_test)
    alpha_vector.append(beta_test)
    alpha = 1 + beta_test
    nb_c.makeLogMAPMatrix(alph)
    nb_c.classify()
    accur = nb_c.calculateAccuracy(False)
    print(accur)
    accuracy_vector.append(accur)
    a = a - 0.05

print(alpha_vector)
print(accuracy_vector)
