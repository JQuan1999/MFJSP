import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from utils.population import Population
from utils.instance import ReadInstances
from utils.evolution import EvolutionOperator
from utils.nsga_algorithm import *

np.random.seed(1)
random.seed(1)
sys.setrecursionlimit(10000)  # 设置递归深度限制为10000或更高

iter_max = 300  # 迭代次数
pop_size = 120  # 种群数目
mobj = 4  # 目标函数个数
pc, pm = 0.9, 0.1  # 交叉、变异概率
local_search_iter = 10 # 个体局部搜索深度
tourment_select_size = 10 # 局部搜索锦标赛大小
pls = 0.1 # 局部搜索概率

testdir = "./test"
resultdir = "./result/nsgals_1/version2"


def NSGALS(data: str = './test/None', result: str = './result/None') -> np.ndarray:
    refer_points, points_size = uniformpoint(pop_size, mobj)  # 生成参考点,参考点集合Z,参考点数目N
    # 读取测试问题
    job_count, machine_count, job_array, machine_array = ReadInstances(data)
    # 初始化种群得到pop种群
    population = Population(points_size, machine_count, job_array, machine_array)
    popfun = population.InitPopulation()
    # 求出理想点
    zmin = np.min(popfun, axis=0).reshape(1, mobj)
    # 创建进化算子
    ga = EvolutionOperator(pc, pm)
    for i in range(iter_max):
        print('第{}次迭代'.format(i))
        # 交叉
        ga.cross(population)
        # 变异
        ga.mutate(population)
        offfun = population.CalChildfunc(mobj)
        # 更新理想点
        zmin = np.min(np.vstack((zmin, offfun)), axis=0).reshape(1, mobj)
        # 得到融合种群目标函数
        mixpopfun = np.vstack((popfun, offfun))
        # pareto排序
        front = ndsort(mixpopfun)
        #  得到前f-1等级的个体f_1th,f等级的个体fth,k剩余k个个体
        mixpop_idx = range(0, mixpopfun.shape[0])
        f_1th, fth, k = accumulate(mixpop_idx, front, pop_size)
        if k > 0:
            fpop = f_1th + fth
            fpopfun = mixpopfun[fpop] # 取前f层个体的目标函数值
            nextpop_idx, lastk_idx = choose(f_1th, fth, k, zmin, refer_points, fpopfun)
            _, popfun = population.GenerateNextPop(nextpop_idx, mixpopfun)
        else:
            _, popfun = population.GenerateNextPop(f_1th, mixpopfun)
    population.Save(result)
    return popfun


if __name__ == "__main__":
    files = os.listdir(testdir)
    for file in files:
        testins = '/'.join([testdir, file])
        savefile = file.split('.')[0] + '.npz'
        savefile = '/'.join([resultdir, savefile])
        print(f'test instance {file} run begin')
        NSGALS(testins, savefile)
        print(f'test instance {file} run end')