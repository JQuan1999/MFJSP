import os
import numpy as np

import matplotlib.pyplot as plt

from utils.population import *
from utils.instance import ReadInstances

# 测试数据、sc结果保存路径、gd文件路径、igd文件路径
testdata = ['./MK', './la', './orb', './test']

gatta_load = './result/nsgals/chrome/'

gatta_savedir = './result/figure/gatta/'

# 对比算法名称
load_files = [
    {
        'nsga3':"./result/nsga3/version1",
        'nsgals':"./result/nsgals/version1",
        'nsgals_1':"./result/nsgals_1/version1",
        'nsgals_2': "./result/nsgals_2/version1",
    },
    {
        'nsga3':"./result/nsga3/version1",
        'nsgals':"./result/nsgals/version1",
        'nsgals_1':"./result/nsgals_1/version1",
        'nsgals_2': "./result/nsgals_2/version2",
    },
    {
        'nsga3':"./result/nsga3/version1",
        'nsga3_new':"./result/nsga3/version2",
    },
    {
        'nsgals': "./result/nsgals/version2",
        'nsga3': "./result/nsga3/version2",
    },
    {
        'nsga3':"./result/nsga3/version2",
        'nsgals':"./result/nsgals/version2",
        'nsgals_1':"./result/nsgals_1/version2",
        'nsgals_2': "./result/nsgals_2/version3",
    },
    {
        'nsgals':"./result/nsgals/chrome",
    }
]


def ShowAll(testdir, savedir):
    files = os.listdir(testdir)
    algorithm = [key for key in savedir.keys()]
    title = ['cmax', 'idletime', 'workload', 'ecost']
    for file in files:
        iter_datas = []

        for algo, res in savedir.items():
            # 拼凑保存的数据文件路径
            savefile = file.split('.')[0] + '.npz'
            savefile = '/'.join([res, savefile])
            # 加载迭代曲线值
            iter_curve = np.load(savefile)["arr_0"]
            iter_datas.append(iter_curve)

        n_obj = iter_datas[0].shape[1] # 目标函数个数
        fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True, sharey=False)
        suptitle = file.split('.')[0]
        fig.suptitle(suptitle)
        for i in range(n_obj):
            row = int(i / 2)
            col = i % 2
            for j, data in enumerate(iter_datas):
                obj_value = data[:, i].tolist() # algorithm[j]在第i个目标上的目标函数值
                axs[row][col].plot(range(len(obj_value)), obj_value, label=algorithm[j])
                axs[row][col].set_title(title[i])
                axs[row][col].legend(loc=1)

        plt.title(file)
        # plt.pause(5)
        plt.savefig('./result/figure/'+suptitle+'.png')
        # plt.close()
        plt.cla()


def ShowOne(testdir, savedir):
    files = os.listdir(testdir)
    label = ['cmax', 'idletime', 'workload', 'ecost']
    for file in files:
        iter_datas = []

        test_ins = file.split('.')[0]
        plt.gca().spines['right'].set_color('none')
        plt.gca().spines['top'].set_color('none')
        for algo, res in savedir.items():
            # 拼凑保存的数据文件路径
            savefile = test_ins + '.npz'
            savefile = '/'.join([res, savefile])
            # 加载迭代曲线值
            iter_curve = np.load(savefile)["arr_0"]
            iter_datas.append(iter_curve)

        n_obj = iter_datas[0].shape[1] # 目标函数个数
        plt.title(test_ins)
        for i in range(n_obj):
            for j, data in enumerate(iter_datas):
                obj_value = data[:, i].tolist() # algorithm[j]在第i个目标上的目标函数值
                plt.plot(range(len(obj_value)), obj_value, label=label[i])
        plt.legend(loc='upper right', fontsize='small')
        plt.title(f"{test_ins}上种群目标函数均值的迭代曲线")
        # plt.pause(5)
        plt.savefig('./result/figure/'+test_ins+'.png', dpi=400)
        # plt.close()
        plt.cla()


def ShowGatta():
    obj_names = ['camx', 'idletime', 'cworkload', 'ec']
    testdir = testdata[0] # 测试文件目录
    testins = os.listdir(testdir) # 所有测试用例
    for file in testins:
        file_path = '/'.join([testdir, file]) # 测试问题
        job_count, machine_count, job_array, machine_array = ReadInstances(file_path)
        chrome = Chromosome(machine_count, [EnumRule.RM_RULE] * 4, job_array, machine_array)

        test_name = file.split('.')[0]  # 测试问题名称Mk01

        # 读取数据
        datafile = test_name + '.npz'
        datafile = '/'.join([gatta_load, datafile])
        np_data = np.load(datafile)
        popfun = np_data['arr_1']  # 目标函数
        jccode = np_data['arr_2']  # 工件编码
        mccode = np_data['arr_3']  # 工序编码
        min_value = np.min(popfun, axis=0)
        # 分别返回四个目标取值最小的索引位置
        min_idx = np.argmin(popfun, axis=0)
        # 获取cmax最小的索引位置信息
        obj_values = popfun[min_idx[0], :].tolist()
        print(file, obj_values, min_value)
        jc = jccode[min_idx[0], :].astype(int)
        mc = mccode[min_idx[0], :].astype(int)
        # 设置编码信息
        chrome.SetJobCode(jc)
        chrome.SetMachineCode(mc)
        chrome.decode()
        # 设置标题信息格式如mk01 cmax=100 idletime = 10 cworkload = 10 ecost = 10
        title = test_name
        for i, name in enumerate(obj_names):
            title += ' ' + name + '=' + str(obj_values[i])
        # 保存文件路径
        savefile = test_name + '.png'
        savefile = '/'.join([gatta_savedir, savefile])
        chrome.machine_array.Gatta(job_count, title, savefile)


if __name__ == "__main__":
    ShowOne(testdata[0], load_files[-1])
    # ShowGatta()