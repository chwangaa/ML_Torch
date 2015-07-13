#! /usr/bin/env python

# Take any number of files produced by Caffe and combine them into a single
# C header file so the model can be compiled into the program. This greatly
# reduces the time taken to set up the model on slow simulators.

# Each file represents a different layer of the network and must have the
# following format:
#   Line 1: layer parameters, space separated
#   Lines 2-n: Weights, one weight per line

# The output of this program will then contain:
#   int FILENAME_params[]
#   double FILENAME_data[]
# TODO: make weight's type configurable

import os.path
from sys import argv

def processFile(filename):
    basename = os.path.basename(filename)
    layer = os.path.splitext(basename)[0]
    
    with open(filename) as f:
        firstline = f.readline()
        params = firstline.split()
        print "int %s_params[] = {%s};" % (layer, ", ".join(params))
        
        print "double %s_data[] = {%s};" % (layer, ", ".join([line.strip() for line in f.readlines()]))

print """
#ifndef WEIGHTS_H
#define WEIGHTS_H

"""

for filename in argv[1:]:
    processFile(filename)

print "#endif"
