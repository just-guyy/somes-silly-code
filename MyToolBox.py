import math
variables = {}

def factorial(x):
    if x < 0:
        raise ValueError(f"yo wtf {x} is a negative number lol")
    elif x == int(x):
        return math.factorial(int(x))
    else:
        return math.gamma(x + 1)


def sin(input, accuracy=5):
    output = 0
    sign = 1
    for i in range(1, accuracy*2-1, 2):
        if sign==1:
            output += input**i / factorial(i)
            sign = 0
        elif sign==0: 
           output -= input**i / factorial(i)
           sign = 1
        else:
            raise IndexError("wtf")
    return output


def cos(input, accuracy=5):
    output = 0
    sign = 1
    for i in range(0, accuracy*2, 2):
        if sign==1:
            output += input**i / factorial(i)
            sign = 0
        elif sign==0: 
           output -= input**i / factorial(i)
           sign = 1
        else:
            raise IndexError("wtf")
    return output


def tan(input, accuracy=5):
    return(sin(input, accuracy)/cos(input, accuracy))

def radians(input):
    return input/(3.14 / 180.0)

def initialize_nn(id):
    stuff = variables[id]
    for i in range(1, stuff["layers"]):
        v = "weights" + str(i) + str(i+1)
        v2 = "layer" + str(i)
        v3 = "layer" + str(i+1)
        stuff[v] = [0] * (stuff[v2] * stuff[v3])
    for i in range(1, stuff["layers"]+1):
        v = str(i)
        vv = "layer" + v
        w = "biases" + v
        stuff[w] = [0] * stuff[vv]
    for i in range(1, stuff["layers"]+1):
        stuff["layer"+str(i)+"n"] = [0] * stuff["layer"+str(i)]
    variables[id] = stuff

def activation(input):
    #relu
    for i in range(len(list(input))):
        if input[i] <= 0:
            input[i] = 0
    return input

def nn_run(id):
    stuff = variables[id]
    stuff["temp"] = []
    for i in range(0, stuff["layers"]-1):
        del stuff["temp"]
        stuff["temp"] = []
        for l in range(0, stuff["layer"+str(i+2)]):
            stuff["temp"].append(0)
        v = str(i+1) + str(i+2)
        w = "weights" + v
        for j in range(0, len(stuff[w])):
            x = len(stuff[w])
            y = len(stuff["layer"+str(i+2)+"n"])
            z = stuff[w][j]
            jz = j%y
            jy = int((j-jz)/y)
            stuff["temp"][jz] = stuff["layer"+str(i+1)+"n"][jy] * z
        stuff["layer"+str(i+2)+"n"] = activation([x - y for x, y in zip(stuff["temp"], stuff["biases"+str(i+2)])])
    del stuff["temp"]
    variables[id] = stuff
