import numpy as np

a = []

def parseLine(ls):
    ls[:] = [int(y) for y in ls]
    return ls

def test():
    with open('data/vocabulary.txt') as f:
        for line in f:
            a.append(line.rstrip())
    print(a)

test()
