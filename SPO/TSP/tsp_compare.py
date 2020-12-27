import time

import numpy as np
import pickle
from SPO_tree_greedy import SPOTree
from decision_problem_solver import*
import sys

pred_length = np.loadtxt('LR_tsp100_pred_100.txt')
real_length = np.loadtxt('LR_tsp100_real_100.txt')
n_test = 100

pred_decision = find_opt_decision(pred_length)['weights']
opt_decision = find_opt_decision(real_length)['weights']

costs_arr = [np.sum(real_length[i] * pred_decision[i]) for i in range(0,len(pred_decision))]
optcosts_arr = [np.sum(real_length[i] * opt_decision[i,:]) for i in range(0,opt_decision.shape[0])]


pred_sum = np.sum(costs_arr)
avgPreLength = pred_sum / n_test
print("avgPreLength: " + str(round(avgPreLength, 2)) )
#print costs_arr.shape
#print np.sum(costs_arr)

opt_sum = np.sum(optcosts_arr)
avgOptLength = opt_sum / n_test
print("avgOptLength: " + str(round(avgOptLength, 2)) )

avgError = (pred_sum - opt_sum) / n_test
print("avgError: " + str(round(avgError, 2)) )
