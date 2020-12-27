'''
python SPOTree_tsp.py 100 0 10 0 1 3 20 SPO

Runs SPOT (greedy) / CART algorithm on shortest path dataset with nonlinear mapping from features to costs ("nonlinear")
Outputs algorithm decision costs for each test-set instance as pickle file
Also outputs optimal decision costs for each test-set instance as pickle file
Takes multiple input arguments:
  (1) n_train: number of training observations. can take values 200, 10000
  (2) eps: parameter (\bar{\epsilon}) in the paper controlling noise in mapping from features to costs.
    n_train = 200: can take values 0, 0.25
    n_train = 10000: can take values 0, 0.5
  (3) deg_set_str: set of deg parameters to try, e.g. "2-10". 
    deg = parameter "degree" in the paper controlling nonlinearity in mapping from features to costs. 
    can try values in {2,10}
  (4) reps_st, reps_end: we provide 10 total datasets corresponding to different generated B values (matrix mapping features to costs). 
    script will run code on problem instances reps_st to reps_end 
  (5) max_depth: training depth of tree, e.g. "5"
  (6) min_weights_per_node: min. number of (weighted) observations per leaf, e.g. "100"
  (7) algtype: set equal to "MSE" (CART) or "SPO" (SPOT greedy)
Values of input arguments used in paper:
  (1) n_train: consider values 200, 10000
  (2) eps:
    n_train = 200: considered values 0, 0.25
    n_train = 10000: considered values 0, 0.5
  (3) deg_set_str: "2-10"
  (4) reps_st, reps_end: reps_st = 0, reps_end = 10
  (5) max_depth: 
    n_train = 200: considered depths of 1, 2, 3, 1000
    n_train = 10000: considered depths of 2, 4, 6, 1000
  (6) min_weights_per_node: 20
  (7) algtype: "MSE" (CART) or "SPO" (SPOT greedy)
'''

import time

import numpy as np
import pickle
from SPO_tree_greedy import SPOTree
from decision_problem_solver import*
import sys
#problem parameters
n_train = int(sys.argv[1])#200
eps = float(sys.argv[2])#0
deg_set_str = sys.argv[3]
deg_set=[int(k) for k in deg_set_str.split('-')]#[2,4,6,8,10]
#evaluate algs of dataset replications from rep_st to rep_end
reps_st = int(sys.argv[4])#0 #can be as low as 0
reps_end = int(sys.argv[5])#1 #can be as high as 50
valid_frac = 0.2 #set aside valid_frac of training data for validation
########################################
#training parameters
max_depth = int(sys.argv[6])#3
min_weights_per_node = int(sys.argv[7])#20
algtype = sys.argv[8] #either "MSE" or "SPO"
########################################
trainX = np.loadtxt('train_tsp_X_100.txt')
trainY = np.loadtxt('train_tsp_Y_100.txt')
testX = np.loadtxt('test_tsp_X_100.txt')
testY = np.loadtxt('test_tsp_Y_100.txt')
n_test = 30
#############################################################################
#############################################################################
#############################################################################
assert(reps_st >= 0)
assert(reps_end <= 50)
n_reps = reps_end-reps_st

costs_deg = {}
optcosts_deg = {} #optimal costs

for deg in deg_set:
  costs_deg[deg] = np.zeros((n_reps,n_test))
  optcosts_deg[deg] = np.zeros((n_reps,n_test))
    
  for trial_num in range(reps_st,reps_end):
     train_x = trainX
     train_cost = trainY
     test_x = testX
     test_cost = testY
    
    #split up training data into train/valid split
     n_valid = int(np.floor(n_train*valid_frac))
     valid_x = train_x[:n_valid]
     valid_cost = train_cost[:n_valid]
     train_x = train_x[n_valid:]
     train_cost = train_cost[n_valid:]
    
     start = time.time()
    
    #FIT ALGORITHM
    #SPO_weight_param: number between 0 and 1:
    #Error metric: SPO_loss*SPO_weight_param + MSE_loss*(1-SPO_weight_param)
     if algtype == "MSE":
       SPO_weight_param=0.0
     elif algtype == "SPO":
       SPO_weight_param=1.0
     my_tree = SPOTree(max_depth = max_depth, min_weights_per_node = min_weights_per_node, quant_discret = 0.01, debias_splits=False, SPO_weight_param=SPO_weight_param, SPO_full_error=True)
     my_tree.fit(train_x,train_cost,verbose=False,feats_continuous=True); #verbose specifies whether fitting procedure should print progress
    #my_tree.traverse() #prints out the unpruned tree 
    
     end = time.time()
     print "Elapsed time: " + str(end-start)
    
    #PRUNE DECISION TREE USING VALIDATION SET
     my_tree.prune(valid_x, valid_cost, verbose=False, one_SE_rule=False) #verbose specifies whether pruning procedure should print progress
    #my_tree.traverse() #prints out the pruned tree

    #FIND TEST SET ALGORITHM COST AND OPTIMAL COST
     opt_decision = find_opt_decision(test_cost)['weights']
     pred_decision = my_tree.est_decision(test_x)
     np.savetxt("SPOTree_pred_100.txt", pred_decision, fmt="%.3f")
     pred_value = my_tree.est_cost(test_x)
     np.savetxt("SPOTree_predV_100.txt", pred_value, fmt="%.3f")
     
     costs_deg[deg][trial_num] = [np.sum(test_cost[i] * pred_decision[i]) for i in range(0,len(pred_decision))]
     optcosts_deg[deg][trial_num] = [np.sum(test_cost[i] * opt_decision[i,:]) for i in range(0,opt_decision.shape[0])]

    # Saving the objects occasionally:
     if trial_num % 25 == 0:
       with open(fname_out, 'wb') as output:
         pickle.dump(costs_deg, output, pickle.HIGHEST_PROTOCOL)
       with open(fname_out_opt, 'wb') as output:
         pickle.dump(optcosts_deg, output, pickle.HIGHEST_PROTOCOL)

with open(fname_out, 'wb') as output:
  pickle.dump(costs_deg, output, pickle.HIGHEST_PROTOCOL)
with open(fname_out_opt, 'wb') as output:
  pickle.dump(optcosts_deg, output, pickle.HIGHEST_PROTOCOL)
  
costs_arr = costs_deg[10]
pred_sum = np.sum(costs_arr)
avgPreLength = pred_sum / n_test
print("avgPreLength: " + str(round(avgPreLength, 2)) )

optcosts_arr = optcosts_deg[10]
opt_sum = np.sum(optcosts_arr)
avgOptLength = opt_sum / n_test
print("avgOptLength: " + str(round(avgOptLength, 2)) )

avgError = (pred_sum - opt_sum) / n_test
print("avgError: " + str(round(avgError, 2)) )
