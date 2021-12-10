import numpy as np
import os
from scipy.sparse import coo_matrix, data
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix
import torch.optim as optim

#from igraph import Graph
#from IPython.display import SVG, display

from gtda.graphs import GraphGeodesicDistance
from gtda.homology import VietorisRipsPersistence, SparseRipsPersistence, FlagserPersistence
from gtda.diagrams import PersistenceEntropy,  BettiCurve
from gtda.diagrams import NumberOfPoints
from gtda.diagrams import Amplitude

import random
import copy
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from tda_net import *

#data
DATASET = 'NCI1' #NCI1, ENZYMES, PROTEINS, MUTAG
DATAPATH = './data' #'./gitcode/Capstone/data'

#tests
tree_method_test = False
fcnet_test = False
#SETTINGS - best results with padding and tda; without concat and diag. Also with data_aug_perms > 0
#  Concat takes longer, does not really yield better results. using diagonal matrix actually gives worse results
add_padding = True
concat = False
with_tda = True
diag = False

N_SAMPLES = -1 #use up to this many for testing, -1 for all
data_aug_perms = 1
percent_train = .7


#given data_path, create adjacency matrices and store in npy file for later
def create_npy_adj_features(data_name, path):
    loc = path + "/" + data_name + "/" + data_name
    npy_file = loc + '.npy'
    #if npy file exists, return it
    if os.path.exists(npy_file):
        with open(npy_file, 'rb') as f:
            save_data = np.load(f, allow_pickle=True)
            return save_data
    #print(loc)
    edges_lines = open(loc + "_A.txt").readlines() #each line represents edge i,j
    graph_ids_lines = open(loc + "_graph_indicator.txt").readlines() #line i represents what graph node i is in
    node_label_lines = open(loc + "_node_labels.txt").readlines() #line i represents label for node i

    #get graph lengths
    graph_lengths = [0]
    for line in graph_ids_lines:
        graph_id = int(line.strip())
        if len(graph_lengths) < graph_id:
            graph_lengths.append(1)
        else:
            graph_lengths[-1] += 1

    print(graph_lengths[:10])
    #construct adjacency matrix for each graph
    adjacency_matrices = []
    sz = graph_lengths[0]
    current_adjacency = np.zeros((sz, sz))
    offset = 0
    for line in edges_lines:
        v1,v2 = [int(x) for x in line.strip().split(',')]
        if v1 > offset+sz or v2 > offset+sz:
            #append + construct new graph
            adjacency_matrices.append(current_adjacency)
            offset += sz
            sz = graph_lengths[len(adjacency_matrices)]
            current_adjacency = np.zeros((sz, sz))
            
        #1+offset to get back to 0-index
        v1 -= 1 + offset
        v2 -= 1 + offset
        current_adjacency[v1, v2] = 1
        current_adjacency[v2, v1] = 1

    #add last one
    adjacency_matrices.append(current_adjacency)
    print(len(adjacency_matrices))

    #get node labels, feature_matrices[i] = X / node features for the ith graph
    feature_matrices = []
    sz = graph_lengths[0]
    current_features = np.zeros((sz, 1))
    offset = 0
    i = 0
    for line in node_label_lines:
        label = int(line.strip())
        i += 1
        ind = i - 1 - offset
        if ind >= sz:
            #append + construct new feature matrix
            feature_matrices.append(current_features)
            offset += sz
            sz = graph_lengths[len(feature_matrices)]
            current_features = np.zeros((sz, 1))
            
        #1+offset to get back to 0-index
        ind = i - 1 - offset
        current_features[ind,0] = label

    #add last one
    feature_matrices.append(current_features)
    print(len(feature_matrices))

    #save
    adjacency_matrices = np.array(adjacency_matrices)
    feature_matrices = np.array(feature_matrices)
    with open(npy_file, 'wb') as f:
        save_data = np.array([adjacency_matrices, feature_matrices])
        np.save(f, save_data)
    return save_data


def get_class_labels(data_name, path):
    loc = path + "/" + data_name + "/" + data_name
    graph_labels_lines = open(loc + "_graph_labels.txt")
    return [ int(l.strip()) for l in graph_labels_lines.readlines()]

def data_aug(X, Y, n_perms_wanted=3):
    print("Creating extra data samples...")
    samples = len(Y)
    lines_output = 20
    for i in range(samples):
        if i % (samples//lines_output) == 0:
            print("{}/{}".format(i,samples))
        permutations = []
        Xi_adj = X[0][i]
        Xi_features = X[1][i]
        yi = Y[i]
        #print("Xi adj, feat", Xi_adj.shape, Xi_features.shape)
        n = len(Xi_adj)
        #get permutations
        perm = list(range(n))
        n_perms = min(n_perms_wanted, n*(n-1)/2)
        while len(permutations) < n_perms:
            np.random.shuffle(perm)
            if perm not in permutations:
                permutations.append(copy.copy(perm))
        
        #create new data
        for p in permutations:
            p_matrix = np.zeros((n,n))
            for j in range(n):
                p_matrix[j][p[j]] = 1
            #print("Dims adj, feat, p", Xi_adj.shape, Xi_features.shape, p_matrix.shape)
            Xi_adj_perm =  p_matrix @ Xi_adj #row swap
            Xi_features_perm =  p_matrix @  Xi_features # Xi_featues is (n x features) 
            X[0].append(Xi_adj_perm)
            X[1].append(Xi_features_perm)
            Y.append(yi)

    return X, Y


def adjacency_to_coo(adjacency_matrices):
    ret = []
    for M in adjacency_matrices:
        C = coo_matrix(M)
        ret.append(C)
    return ret

def print_acc(y_true, y_preds):
    print("Accuracy: ", accuracy_score(y_true, y_preds))

def pad(a, l):
    b = list(a.squeeze()) + [0]*(l-len(a))
    #b = [int(el) for el in b] 
    return np.array(b)

def pad2d(a, l):
    b = np.zeros((l,l))
    n = len(a)
    b[:n,:n] = a
    return b

def label_to_onehot(a):
    a = np.array(a)
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    return b

#import dataset
X = create_npy_adj_features(DATASET, DATAPATH)
Y = get_class_labels(DATASET, DATAPATH) 
if -1 in Y:
    Y = [y+1 for y in Y] #MUTAG 
print(len(X[0]), len(X[1]))
print(len(Y))
class_dist = {y: Y.count(y) for y in Y}
print("Class distribution, ", class_dist)
#print("X0 vs X1 shape test")
#for i in range(10):
#    print(len(X[0][i]), len(X[1][i]))
graph_lens = [len(X[1][i]) for i in range(len(X[1]))]
print("Graph len max", max(graph_lens))

if N_SAMPLES == -1:
    N_SAMPLES = len(Y)
train_index = int(N_SAMPLES*percent_train)
data_permutation = np.random.permutation(len(Y)) #Shuffle data
#np.random.shuffle(data_permutation)
X_adj = list(X[0][data_permutation][:N_SAMPLES])
X_feat = list(X[1][data_permutation][:N_SAMPLES])
X = [X_adj, X_feat]
Y = np.array(Y)[data_permutation][:N_SAMPLES]
Y = list(Y) #back to list for data_aug

#Split data into train/test. Leave test set alone + use train set for data augmentation and other
Y_train = Y[:train_index]
Y_test = Y[train_index:]
X_train = [X_adj[:train_index], X_feat[:train_index]]
X_test = [X_adj[train_index:], X_feat[train_index:]]

print("X_train, Y_train, ", len(X_train[0]), len(Y_train))
class_dist = {y: Y_train.count(y) for y in Y_train}
print("Class distribution, ", class_dist)
X_train, Y_train = data_aug(X_train, Y_train, n_perms_wanted=data_aug_perms)
print("X_train, Y_train, ", len(X_train[0]), len(Y_train))
train_index = len(X_train[0]) #update index


#TEST TDA: want to check if we can get high acc on base VR->PE embedding (without X_feat)
X_adj = X_train[0] + X_test[0]
#X_adj = GraphGeodesicDistance(directed=False, unweighted=True).fit_transform(X_adj) same acc
X_adj = adjacency_to_coo(X_adj) #Take all adjacency matrices
print("VR...")
homology = [0,1,2,3]
#VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2, 3])  # check for 0,1,2,3d holes 63% test 99% train
VR = VietorisRipsPersistence(homology_dimensions=homology, metric='precomputed')  # 64% train test #precomputed -> adj not point clouds
#VR = SparseRipsPersistence(metric='precomputed')
diagrams = VR.fit_transform(X_adj)
print(diagrams.shape)

PE = PersistenceEntropy()
#BC = BettiCurve()
features_PE = PE.fit_transform(diagrams)
features_NP = NumberOfPoints().fit_transform(diagrams)
#features_wass = Amplitude(metric='bottleneck').fit_transform(diagrams) #["bottleneck", "wasserstein", "landscape", "persistence_image"]
features = np.concatenate((features_PE, features_NP), axis=1)
#features = np.concatenate((features2, features3, features_NP, features_wass), axis=1)
tda_in_dim = len(features[0])
print(features.shape)
tda_features_train = features[:train_index]
tda_features_test = features[train_index:]

if tree_method_test:
    ##TREE
    #Train set
    print("Testing w tree (training set only)")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(tda_features_train, Y_train)
    preds = clf.predict(tda_features_train)
    print_acc(Y_train, preds)
    print("Samples: {}".format(len(tda_features_train)))

    #Test set
    print("Testing w tree (on test set only)")
    preds = clf.predict(tda_features_test)
    print_acc(Y_test, preds)
    print("Samples: {}".format(len(tda_features_test)))

    ##Random forest
    print("RFC")
    model = RandomForestClassifier()
    model.fit(tda_features_train, Y_train)
    print(model.score(tda_features_train, Y_train))
    print(model.score(tda_features_test, Y_test))
    exit()

##NN with features, First generate datasets
#padding
if add_padding:
    X_feat_padded_train = [pad(X_train[1][i], max(graph_lens)) for i in range(len(X_train[1]))]  #X_feat, padded with 0s to hit max len
    X_feat_padded_test = [pad(X_test[1][i], max(graph_lens)) for i in range(len(X_test[1]))]  #X_feat, padded with 0s to hit max len
    X_feat_padded_train = np.array(X_feat_padded_train)
    X_feat_padded_test = np.array(X_feat_padded_test)
    X_adj_padded_train =  [pad2d(X_train[0][i], max(graph_lens)) for i in range(len(X_train[0]))]
    X_adj_padded_test = [pad2d(X_test[0][i], max(graph_lens)) for i in range(len(X_test[0]))]
    X_adj_padded_train = np.array(X_adj_padded_train)
    X_adj_padded_test = np.array(X_adj_padded_test)
else:
    X_feat_padded_train = X_train[1]
    X_feat_padded_test = X_test[1]
    X_adj_padded_train = X_train[0]
    X_adj_padded_test = X_test[0]
#
#print("Dims x feat tda feat: ", X_feat_padded_train.shape, tda_features_train.shape)
#X_train_all_features = np.concatenate((X_feat_padded_train, tda_features_train), axis=1) 
#X_test_all_features = np.concatenate((X_feat_padded_test, tda_features_test), axis=1) 
Y_train_onehot = label_to_onehot(Y_train)

#print("Dims X Y train", X_train_all_features.shape, Y_train_onehot.shape)
#TEST TDA: use embedded X_adj with X_feat
if fcnet_test and add_padding:
    x_feat_in_dim = max(graph_lens)
    #tda_in_dim =  len(homology)*2
    out_dim = Y_train_onehot.shape[1]
    net = TDA_FCNet(x_feat_in_dim, tda_in_dim, out_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    batch_size = 4
    trainset = TDADataset(X_feat_padded_train, tda_features_train, np.array(Y_train))
    print("Dim train, ", X_feat_padded_train.shape, tda_features_train.shape)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    for epoch in range(200):  # loop over the dataset multiple times
        print("EPOCH", epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            x, x_tda, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(x, x_tda)

            # type issues
            labels = torch.tensor(labels, dtype=torch.long)
            #outputs = outputs.float()
            #print("sizes outputs/labels", outputs.size(), labels.size())

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        print('Epoch {}. Accuracy on train set'.format(epoch))
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in trainloader:
                x, x_tda, labels = data
                # type issues
                labels = torch.tensor(labels, dtype=torch.long)
                # calculate outputs by running images through the network
                outputs = net(x, x_tda)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Accuracy: ", correct/total)

#TEST TDA with gcn: use embedded X_adj with X_feat
x_feat_in_dim = 1# max(graph_lens)
adj_dim = max(graph_lens)
#tda_in_dim =  len(homology)*2
out_dim = Y_train_onehot.shape[1]
if add_padding:
    print("WITH PADDING")
    net = TDA_GCNet_padded(x_feat_in_dim, adj_dim, tda_in_dim, out_dim, diag=diag, concat=concat, with_tda=with_tda)
else:
    print("WITHOUT PADDING")
    net = TDA_GCNet(x_feat_in_dim, adj_dim, tda_in_dim, out_dim, diag=diag, concat=concat, with_tda=with_tda)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
batch_size = 1 if not add_padding else 4
if add_padding:
    trainset = TDA_GCN_Dataset(X_feat_padded_train, X_adj_padded_train, tda_features_train, np.array(Y_train), diag=diag)
    testset = TDA_GCN_Dataset(X_feat_padded_test, X_adj_padded_test, tda_features_test, np.array(Y_test), diag=diag)
else: 
    trainset = TDA_GCN_Dataset_no_pad(X_feat_padded_train, X_adj_padded_train, tda_features_train, np.array(Y_train), diag=diag)
    testset = TDA_GCN_Dataset_no_pad(X_feat_padded_test, X_adj_padded_test, tda_features_test, np.array(Y_test), diag=diag)

#print("GCN Dim train, ", X_feat_padded_train.shape, X_adj_padded_train.shape, tda_features_train.shape)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=True, num_workers=0)

for epoch in range(200):  # loop over the dataset multiple times
    print("EPOCH", epoch)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        if diag:
            x, x_tda, x_adj, d, labels = data
        else:
            x, x_tda, x_adj, labels = data
            d = None


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(x, x_tda, x_adj, d)

        # type issues
        labels = torch.tensor(labels, dtype=torch.long)
        #outputs = outputs.float()
        #print("sizes outputs/labels", outputs.size(), labels.size())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    print('Epoch {}.'.format(epoch))
    if (epoch + 1) % 5 == 0:
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        print("Train/test")
        with torch.no_grad():
            for loader in [trainloader, testloader]:
                for data in loader:
                    if diag:
                        x, x_tda, x_adj, d, labels = data
                    else:
                        x, x_tda, x_adj, labels = data
                        d = None
                    # type issues
                    labels = torch.tensor(labels, dtype=torch.long)
                    # calculate outputs by running images through the network
                    outputs = net(x, x_tda, x_adj, d)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print("Accuracy: ", correct/total)

exit()
