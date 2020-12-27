#!/usr/bin/python

# Copyright 2019, Gurobi Optimization, LLC

# Solve a traveling salesman problem on a randomly generated set of
# points using lazy constraints.   The base MIP model only includes
# 'degree-2' constraints, requiring each node to have exactly
# two incident edges.  Solutions to this model may contain subtours -
# tours that don't visit every city.  The lazy constraint callback
# adds new constraints to cut them off.

import sys
import math
import random
import itertools
from gurobipy import *
import numpy as np

# Callback - use lazy constraints to eliminate sub-tours

# Parse argument

#if len(sys.argv) < 2:
#    print('Usage: tsp.py npoints')
#    exit(1)
#n = int(sys.argv[1])

# Create n random points
nodeNum = 5
edgeNum = 10

random.seed(1)
points = [(random.randint(0,100),random.randint(0,100)) for i in range(nodeNum)]
#print points

def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = tuplelist((i,j) for i,j in model._vars.keys() if vals[i,j] > 0.5)
        # find the shortest cycle in the selected edge list
        tour = subtour(selected)
        if len(tour) < nodeNum:
            # add subtour elimination constraint for every pair of cities in tour
            model.cbLazy(quicksum(model._vars[i,j]
                                  for i,j in itertools.combinations(tour, 2))
                         <= len(tour)-1)


# Given a tuplelist of edges, find the shortest subtour

def subtour(edges):
    unvisited = list(range(nodeNum))
    cycle = range(nodeNum+1) # initial length has 1 more city
    while unvisited: # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i,j in edges.select(current,'*') if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle


def find_opt_decision(cost):
    weights = np.zeros((cost.shape[0], nodeNum), dtype=int)
    weightSol = np.zeros((cost.shape[0], edgeNum), dtype=int)
    #print weightSol.shape
    objective = np.zeros(cost.shape[0])
#    indexDict = {(0,1):0,(0,2):1,(0,3):2,(0,4):3,(1,2):4,(1,3):5,(1,4):6,(2,3):7,(2,4):8,(3,4):9,(1,0):0,(2,0):1,(3,0):2,(4,0):3,(2,1):4,(3,1):5,(4,1):6,(3,2):7,(4,2):8,(4,3):9}

    for c in range(cost.shape[0]):
#        print c
        print cost.shape
        k = 0
        dist = {}
        for i in range(nodeNum):
            for j in range(i+1, nodeNum):
                dist[(i,j)] = cost[c][k]
#                print i
#                print j
#                print cost[c][k]
                k = k + 1
        #print dist

        m = Model()

        # Create variables

        vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
        for i,j in vars.keys():
            vars[j,i] = vars[i,j] # edge in opposite direction


        m.addConstrs(vars.sum(i,'*') == 2 for i in range(nodeNum))

        m._vars = vars
        m.Params.lazyConstraints = 1
        m.optimize(subtourelim)

        vals = m.getAttr('x', vars)
        selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)

        tour = subtour(selected) # tour is a list variable
        assert len(tour) == nodeNum
        
        for i in range(nodeNum):
            weights[c][i] = tour[i]
        for j in range(nodeNum):
            if j < nodeNum - 1:
                a = weights[c][j]
                b = weights[c][j+1]
                if a < b:
                    index = (a * (2*nodeNum-a-1))/2 + b - (a+1)
                    weightSol[c][index] = 1
                else:
                    index = (b * (2*nodeNum-b-1))/2 + a - (b+1)
                    weightSol[c][index] = 1
            elif j == nodeNum - 1:
                a = weights[c][j]
                b = weights[c][0]
                index = (b * (2*nodeNum-b-1))/2 + a - (b+1)
                weightSol[c][index] = 1
#        for j in range(n):
#            if j < n - 1:
#                weightSol[c][indexDict[(weights[c][j],weights[c][j+1])]] = 1
#            elif j == n - 1:
#                weightSol[c][indexDict[(weights[c][j],weights[c][0])]] = 1
        objective[c] = m.objVal
#        print('')
#        print('Optimal tour: %s' % str(tour))
#        print('Optimal cost: %g' % m.objVal)
#        print('')
    return {'weights': weightSol, 'objective':objective}

#
#    print weights
#    print objective
    

# Dictionary of Euclidean distance between each pair of points

#distGraph = np.loadtxt('graph_50.txt')
#opt_decision = find_opt_decision(distGraph)['weights']
#opt_value = find_opt_decision(distGraph)['objective']
#print opt_decision
#print opt_value

