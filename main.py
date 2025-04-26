import MyToolBox as tb

def delusional_attention(V, Q, K):
    output = [[None]*len(V[_]) for _ in range(len(V))]
    tokens = len(V)
    for i in range(tokens):
        wants = Q[i]
        for j in range(tokens):
            #if j == i:
                #continue #no need in skipping the current token ig
            temp = [K[j][_] * wants[_] for _ in range(len(K[j]))]
            temp2 = sum(temp)
            output[i] = [V[j][_] * (temp[j]/temp2) for _ in range(len(temp))]
    print(output)

def attention(V, Q, K):
    output = [[None]*len(V[_]) for _ in range(len(V))]
    tokens = len(V)
    for i in range(tokens): 
        wants = Q[i]
        temp = []
        temp1 = 0
        for j in range(tokens):
            #if j == i:
                #continue #no need in skipping the current token ig
            temp.append(sum([K[j][_] * wants[_] for _ in range(len(K[j]))])) #holy shit what are those nested brackets
        temp1 = sum(temp) #for softmax
        temp2 = [] #after softmax
        for _ in range(len(temp)):
            temp2.append(temp[_]/temp1)
        output[i] = [
            sum([temp2[j] * V[j][dim] for j in range(tokens)])
            for dim in range(len(V[0]))
        ]
    print(output)


VV = [[1,2,3],[4,5,6]]
QQ = [[1,2,3],[4,5,6]]
KK = [[1,2,3],[4,5,6]]

tb.variables["0"] = {}
tb.variables["0"]["layers"] = 4
tb.variables["0"]["layer1"] = 50
tb.variables["0"]["layer2"] = 200
tb.variables["0"]["layer3"] = 200
tb.variables["0"]["layer4"] = 20

tb.initialize_nn("0")

tb.nn_run("0")

print(tb.variables["layer4n"])
