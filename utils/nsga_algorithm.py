import numpy as np
import math
from itertools import combinations
from math import comb


# 生成参考点
def uniformpoint(n, m):
    H1 = 1
    while comb(H1 + m - 1, m - 1) <= n:
        H1 = H1 + 1
    H1 = H1 - 1
    size = comb(H1 + m - 1, m - 1)
    W = np.array(list(combinations(range(H1 + m - 1), m - 1))) - np.tile(range(m - 1), (size, 1))
    W = (np.concatenate((W, np.tile([H1], (size, 1))), axis=1) - np.concatenate((np.zeros((size, 1)), W), axis=1)) / H1
    if H1 < m:
        H2 = 0
        while comb(H1 + m - 1, m - 1) + comb(H2 + m - 1, m - 1) <= n:
            H2 = H2 + 1
        H2 = H2 - 1
        if H2 > 0:
            size = comb(H2 + m - 1, m - 1)
            W2 = np.array(list(combinations(range(H2 + m - 1), m - 1))) - np.tile(range(m - 1), (size, 1))
            W2 = ((np.concatenate((W2, np.tile([H2], (size, 1))), axis=1)) - np.concatenate((np.zeros((size, 1)), W2),
                                                                                            axis=1)) / H2
            W2 = W2 / 2 + 1 / (2 * m)
            W = np.concatenate((W, W2), axis=0)
    W[W < 1e-6] = 1e-6
    N = W.shape[0]
    return W, N


# 非支配排序
def ndsort(popfun: np.ndarray) -> [int]:
    N, M = popfun.shape
    front = [-1] * N
    count = [0] * N
    dom = []
    for i in range(N):
        dom_i = []
        for j in range(N):
            #   目标函数值小于 等于 大于的个数
            less, equal, more = 0, 0, 0
            for k in range(M):
                if popfun[i][k] < popfun[j][k]:
                    less = less + 1
                elif popfun[i][k] == popfun[j][k]:
                    equal = equal + 1
                else:
                    more = more + 1
            #   i支配j
            if more == 0 and equal != M:
                dom_i.append(j)
            #   i被j支配
            elif less == 0 and equal != M:
                count[i] = count[i] + 1
        dom.append(dom_i)
    F = []
    f = 1
    for i in range(N):
        if count[i] == 0:
            front[i] = f
            F.append(i)
    while len(F) != 0:
        Q = []
        f = f + 1
        for point in F:
            #   dom_point是point支配的点
            for dom_point in dom[point]:
                count[dom_point] = count[dom_point] - 1
                if count[dom_point] == 0:
                    Q.append(dom_point)
                    front[dom_point] = f
        F = Q
    return front


def accumulate(mix_pop, front, popsize) -> [[int], [int], int]:
    """
    :param mix_pop: 混合种群索引id
    :param front: 混合种群的pareto层级
    :param popsize: 种群大小
    :return:
        result1: 前f_1层的geti
        result2: f层的个体
        result3: 剩余需要从第f层选出的个体
    """
    front = np.array(front).astype(int)
    maxno = np.max(front).astype(int) # 最大层
    mix_pop = np.array(mix_pop).astype(int)
    divide_pop = []

    choose_next = np.zeros_like(mix_pop).astype(bool)
    choose_last = np.zeros_like(mix_pop).astype(bool)
    for f in range(1, maxno + 1):
        index = np.ravel(np.array(np.where(front == f))).tolist()
        divide_pop.append(index)
    for i, rank in enumerate(divide_pop):
        if popsize >= len(rank):
            choose_next[rank] = True
            popsize = popsize - len(rank)
        elif popsize > 0:
            choose_last[rank] = True
            break
    return mix_pop[choose_next].tolist(), mix_pop[choose_last].tolist(), popsize


def normalize(popfun, zmin):
    N, M = popfun.shape
    #   将popfun减去zmin
    popfun = popfun - np.tile(zmin, (N, 1))
    #   找出M个理想点
    w = np.zeros((M, M)) + 1e-6 + np.eye(M)
    extreme = np.zeros(M)
    for i in range(M):
        extreme[i] = np.argmin(np.max(popfun / np.tile(w[i, :], (N, 1)), axis=1))
    #   计算截距
    extreme = extreme.astype(int)
    #   计算逆矩阵
    temp = np.array(np.linalg.pinv(np.mat(popfun[extreme, :])))
    #   计算超平面的系数
    plane = np.dot(temp, np.ones((M, 1)))
    #   计算截距
    a = 1 / plane
    a = a.T
    #   截距不存在取最大函数值
    if np.sum(a == math.nan) != 0:
        a = np.max(popfun, axis=0)
    a = a - zmin
    popfun1 = popfun / np.tile(a, (N, 1))
    return popfun1


#   计算个体与参考点的距离
def cal_distance(popfun, refer_points):
    pop_size = popfun.shape[0]
    Z_size = refer_points.shape[0]
    #   popfun shape=N*D refer_points.T=D*ND
    pmz = np.dot(popfun, refer_points.T)
    #   计算popfun每行的平方和开根号
    sum1 = np.sqrt(np.sum(popfun ** 2, axis=1)).reshape(pop_size, 1)
    sum2 = np.sqrt(np.sum(refer_points ** 2, axis=1)).reshape(1, Z_size)
    #   sum1.shape=N*1 sum2.shape=NZ*1 sum.shape=N*NZ
    sum = np.dot(sum1, sum2)
    #   计算余弦值
    cos = pmz / sum
    #   计算popfun到Z的距离
    distance = np.tile(sum1, (1, Z_size)) * np.sqrt((1 - cos ** 2))
    return distance


#   联系个体和参考点
def associate(popfun, refer_points, n):
    distance = cal_distance(popfun, refer_points)
    #   找到最短的距离和对应的参考点
    dmin = np.min(distance, axis=1)
    pointmin = np.argmin(distance, axis=1)
    #   计算参考点Z关联的解的个数
    count = np.zeros(refer_points.shape[0])
    for i in range(len(count)):
        count[i] = np.sum(pointmin[:n] == i)
    return dmin, pointmin, count


def choose(f_1th, f_th, k, zmin, refer_points, popfun):
    """
        nsga小生境选择算子，从第f层选择出剩余k个个体
    :param f_1th: 前f-1层
    :param f_th: 第f层
    :param k: 剩余个体
    :param zmin: 理想点
    :param refer_points: 参考点
    :param popfun: 前f层个体目标函数值
    :return:
        pop；所有选择个体的索引
        f_th[choose]：从第f层选择出的剩余k个个体
    """
    f_1th = np.array(f_1th, int)
    f_th = np.array(f_th, int)
    n1 = len(f_1th)
    n2 = len(f_th)
    nz = refer_points.shape[0]
    #   pop=np.concatenate((f_1th,f_th),axis=0)
    normpopfun = normalize(popfun, zmin)
    dmin, pointmin, count = associate(normpopfun, refer_points, n1)
    #   选出剩余的k个点
    choose = np.zeros(n2)
    choose = choose.astype(bool)
    zchoose = np.ones(nz)
    zchoose = zchoose.astype(bool)
    while np.sum(choose) < k:
        #  选择最不拥挤的点
        temp = np.ravel(np.array(np.where(zchoose == True)))
        jmin = np.ravel(np.array(np.where(count[temp] == np.min(count[temp]))))
        j = temp[jmin[np.random.randint(jmin.shape[0])]]
        I = np.ravel(np.array(np.where(pointmin[n1:] == j)))
        I = I[choose[I] == False]
        if (I.shape[0] != 0):
            if (count[j] == 0):
                s = np.argmin(dmin[n1 + I])
            else:
                s = np.random.randint(I.shape[0])
            choose[I[s]] = True
            count[j] = count[j] + 1
        else:
            zchoose[j] = False
    pop = np.concatenate((f_1th, f_th[choose]), axis=0)
    return pop, f_th[choose]