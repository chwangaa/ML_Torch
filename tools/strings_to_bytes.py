#! /usr/bin/env python

# Convert a file containing a list of numbers in text format into stream of
# bytes representing the same numbers. This means that expensive scanf calls
# do not need to be made in the program which uses the numbers.

from sys import argv, stdout

assert len(argv) > 1

filename = argv[1]

with open(filename) as f:
    for line in f:
        val = int(line)
        assert(val < 256) # can only deal with bytes for now
        stdout.write(chr(val))
