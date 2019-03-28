import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy.interpolate import BSpline

epsilon = np.finfo(np.float32).eps

def weight_n(u, i, j, knotvector):
    if j == 0:
        if knotvector[i] <= u <= knotvector[i+1]:
            return 1
        else:
            return 0
    if knotvector[i+j] == knotvector[i]:
        weight1 = 0
    else:
        weight1 = (u - knotvector[i]) / (knotvector[i+j] - knotvector[i])
    if knotvector[i+j+1] == knotvector[i+1]:
        weight2 = 0
    else:
        weight2 = (knotvector[i+j+1] - u) / (knotvector[i+j+1] - knotvector[i+1])

    return weight1*weight_n(u=u, i=i, j=j-1, knotvector=knotvector) + weight2 * weight_n(u=u, i=i+1, j=j-1, knotvector=knotvector)

def lengthBetweenPoints(point1, point2):
    distance = np.square(point1[:, 0] - point2[:, 0]) + np.square(point1[:, 1] - point2[:, 1])
    return np.sqrt(distance)

def ChordLengthParameterization(points):
    point1 = np.array(points[:-1:])
    point2 = np.array(points[1::])
    dis = lengthBetweenPoints(point1=point1, point2=point2)
    dis = np.absolute(dis)
    denominator = np.sum(a=dis, axis=-1)
    numerator = np.cumsum(a=dis, axis=-1, dtype=np.float)
    weights = numerator / denominator
    weights = list(weights)
    weights.append(0)
    weights = sorted(weights)
    weights = np.round(a=weights, decimals=12)
    return weights

def Ni3(u, i, knot_vector):
    global epsilon
    if u >= knot_vector[i] and u <= knot_vector[i+1]:
        return (u-knot_vector[i])**3 / float((knot_vector[i+1]-knot_vector[i])*(knot_vector[i+2]-knot_vector[i])*(knot_vector[i+3]-knot_vector[i]) + epsilon)
    elif u >= knot_vector[i+1] and u <= knot_vector[i+2]:
        tmp1 = (knot_vector[i + 2] - u) * (u - knot_vector[i]) ** 2 / float(
            (knot_vector[i + 2] - knot_vector[i + 1]) * (knot_vector[i + 3] - knot_vector[i]) * (
                        knot_vector[i + 2] - knot_vector[i]) + epsilon)
        tmp2 = (knot_vector[i + 3] - u) * (u - knot_vector[i]) * (u - knot_vector[i + 1]) / float(
            (knot_vector[i + 2] - knot_vector[i + 1]) * (knot_vector[i + 3] - knot_vector[i + 1]) * (
                        knot_vector[i + 3] - knot_vector[i]) + epsilon)
        tmp3 = (knot_vector[i + 4] - u) * (u - knot_vector[i + 1]) ** 2 / float(
            (knot_vector[i + 2] - knot_vector[i + 1]) * (knot_vector[i + 4] - knot_vector[i + 1]) * (
                        knot_vector[i + 3] - knot_vector[i + 1]) + epsilon)
        return tmp1 + tmp2 + tmp3
    elif u >= knot_vector[i+2] and u <= knot_vector[i+3]:
        tmp1 = (u - knot_vector[i]) * (knot_vector[i + 3] - u)**2 / float(
            (knot_vector[i + 3] - knot_vector[i + 2]) * (knot_vector[i + 3] - knot_vector[i + 1]) * (
                        knot_vector[i + 3] - knot_vector[i]) + epsilon)
        tmp2 = (knot_vector[i + 4] - u) * (u - knot_vector[i + 1]) * (knot_vector[i + 3] - u) / float(
            (knot_vector[i + 3] - knot_vector[i + 2]) * (knot_vector[i + 4] - knot_vector[i + 1]) * (
                        knot_vector[i + 3] - knot_vector[i + 1]) + epsilon)
        tmp3 = (u - knot_vector[i + 2]) * (knot_vector[i + 4] - u)**2 / float(
            (knot_vector[i + 3] - knot_vector[i + 2]) * (knot_vector[i + 4] - knot_vector[i + 2]) * (
                        knot_vector[i + 4] - knot_vector[i + 1]) + epsilon)
        return tmp1 + tmp2 + tmp3
    elif u >= knot_vector[i+3] and u <= knot_vector[i+4]:
        return (knot_vector[i+4] - u)**3 / float((knot_vector[i+4] - knot_vector[i+3])*(knot_vector[i+4] - knot_vector[i+2])*(knot_vector[i+4]-knot_vector[i+1]) + epsilon)
    else:
        return 0
if __name__ == "__main__":
    ###########################################
    #read the points that lie in curve
    number = 5
    n = 4
    X = [0, 1, 1, 0, 0]
    Y = [0, 0, 1, 1, 2]
#     X = np.arange(0, number*5, 5, np.float)
# #    X = np.reshape(a=X, newshape=(-1, 1))
#     Y = np.arange(0, number*2, 2, np.float)
#    Y = np.reshape(a=Y, newshape=(-1, 1))
    points = list(zip(X, Y))
    print(points)
    ###########################################

    ###########################################
    #generate knot vector in useful range
    # k to n+1
    # k is degree of the curve, n is number of control points
    k = 3
    # uniform parameterization
    knot_vector1 = np.arange(0, 1+1/(number-1), 1/(number-1), np.float)
    knot_vector1 = np.round(a=knot_vector1, decimals=12)
    # Chord length parameterization
    knot_vector2 = ChordLengthParameterization(points=points)

    # recovering the knot vectors
    a1 = np.repeat(a=knot_vector1[0], repeats=k)
    a2 = np.repeat(a=knot_vector1[-1], repeats=k)
    knot_vector1 = np.concatenate([a1, knot_vector1, a2], axis=-1)
    knot_vector2 = np.concatenate([a1, knot_vector2, a2], axis=-1)
    matrix = np.zeros(shape=(number+2, number-2+k+1))
    for i in range(number):
        weight_k = weight_n(u=knot_vector1[i+3], i=i, j=3, knotvector=knot_vector1)
        weight_k1 = weight_n(u=knot_vector1[i+3], i=i + 1, j=3, knotvector=knot_vector1)
        weight_k2 = weight_n(u=knot_vector1[i+3], i=i + 2, j=3, knotvector=knot_vector1)
        index = np.arange(i, i + k, 1, np.int)
        matrix[i+1, index] = [weight_k, weight_k1, weight_k2]

    # computing the gradient
    # h is precise interval
    h = 0.000001
    u0 = knot_vector1[3]
    tmp1 = derivative(func=Ni3, x0=u0, dx=h, n=2, args=(0, knot_vector1), order=3)
    tmp2 = derivative(func=Ni3, x0=u0, dx=h, n=2, args=(1, knot_vector1), order=3)
    tmp3 = derivative(func=Ni3, x0=u0, dx=h, n=2, args=(2, knot_vector1), order=3)
    matrix[0, 0] = tmp1
    matrix[0, 1] = tmp2
    matrix[0, 2] = tmp3
    u1 = knot_vector1[13]
    tmp11 = derivative(func=Ni3, x0=u1, dx=h, n=2, args=(10, knot_vector1), order=3)
    tmp12 = derivative(func=Ni3, x0=u1, dx=h, n=2, args=(11, knot_vector1), order=3)
    tmp13 = derivative(func=Ni3, x0=u1, dx=h, n=2, args=(12, knot_vector1), order=3)
    matrix[-1, -3] = tmp11
    matrix[-1, -2] = tmp12
    matrix[-1, -1] = tmp13
    D = np.array(points)
    D = np.concatenate([[(0, 0)], D, [(0, 0)]], axis=0)
    P = np.linalg.solve(a=matrix, b=D)
    P = np.round(a=P, decimals=8)
    print(P)
    spl = BSpline(t=knot_vector1, c=P, k=k)
    fig, ax = plt.subplots()
    xx = np.linspace(1.5, 4.5, 50)
    ax.plot(xx, spl(xx), 'b-', lw=4, alpha=0.7, label='BSpline')
    ax.grid(True)
    ax.legend(loc='best')
    plt.show()
    a = 1
    ###########################################