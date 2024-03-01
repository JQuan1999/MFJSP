import os
import numpy as np

import matplotlib.pyplot as plt

from utils.schedule import color

# 对比算法名称
compare_algorithm = [
    ["nsga3", "nsgals","nsgals_1", "nsgals_2"],
    ["nsga3", "nsgals","nsgals_1", "new_nsgals_2"],
    ["MOEAD", "EFR", "nsgals"],
    ["nsgals", "new_nsgals"]
]

testdir = ['./la', './MK', './orb', './test']

resultdir = [
    ["./result/nsga3", "./result/new_nsgals", "./result/nsgals_1", "./result/new_nsgals_2"],
    ["./result/nsga3", "./result/nsgals", "./result/nsgals_1", "./result/new_nsgals_2"]
]


def ShowAll(testdir, algorithm, resdir):
    files = os.listdir(testdir)
    title = ['cmax', 'idletime', 'workload', 'ecost']
    for file in files:
        iter_datas = []

        for res in resdir:
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

        # plt.pause(20)
        plt.title(file)
        plt.savefig('result/figure/'+suptitle+'.png')
        plt.cla()


if __name__ == "__main__":
    # ShowAll('./MK', ['nsgals'], ['./result/nsgals/mk/'])
    ShowAll(testdir[1], compare_algorithm[1], resultdir[0])