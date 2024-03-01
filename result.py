import copy
import os
import logging
import numpy as np

import itertools
from utils.metric import *

# 对比算法名称
compare_algorithm = [
    ["nsga3", "nsgals","nsgals_1", "nsgals_2"],
    ["nsga3", "nsgals","nsgals_1", "new_nsgals_2"],
    ["nsga3", "new_nsgals","nsgals_1", "new_nsgals_2"],
    ["MOEAD", "EFR", "nsgals"],
    ["nsgals", "new_nsgals"],
    ["nsgals"]
]

# 测试数据、所有算法实验结果保存路径、sc结果保存路径、gd文件路径、igd文件路径
exeperiment_files = [
    ['./la', './result', "./result/la/sc.txt", "./result/la/gd.txt", "./result/la/igd.txt", "./result/la/log.txt"],
    ['./MK', './result', "./result/mk/sc.txt", "./result/mk/gd.txt", "./result/mk/igd.txt", "./result/mk/log.txt"],
    ['./orb', './result', "./result/orb/sc.txt", "./result/orb/gd.txt", "./result/orb/igd.txt", "./result/orb/log.txt"],
    ['./MK', './result', "./result/mk/tmp/sc.txt", "./result/mk/tmp/gd.txt", "./result/mk/tmp/igd.txt", "./result/mk/tmp/log.txt"],
]


def CompareSC(algorithm: [str], test_ins: [str], save_file: str, result_dir: str):
    comb_algorithm = list(itertools.combinations(algorithm, 2))
    logging.info(f"comb alogorithm sequence {comb_algorithm}")
    sc_results = []
    better_mp = {}
    total_mp = {}
    algo_map = {'new_nsgals_2': 'nsgals', 'nsga3': 'new_nsgals_2', 'nsgals': 'nsgals_1', 'nsgals_1': 'nsga3'}
    for instance in test_ins:
        r = []
        better_mp.clear()
        for a, b in comb_algorithm:
            # 文件名
            dataname = instance.split('.')[0] + '.npz'
            # 读取算法a保存数据
            datadir_a = '/'.join([result_dir, a, dataname])
            # 读取算法b保存的数据
            datadir_b = '/'.join([result_dir, b, dataname])
            # 读取算法a的种群目标函数
            popfuna = np.load(datadir_a)["arr_1"]
            # 读取算法b的种群目标函数
            popfunb = np.load(datadir_b)["arr_1"]
            sc_ab = SC(popfuna, popfunb)
            sc_ba = SC(popfunb, popfuna)
            r.append(sc_ab)
            r.append(sc_ba)
            if sc_ab > sc_ba:
                better = a
            elif sc_ab < sc_ba:
                better = b
            else:
                better = "equal"
            if better in better_mp:
                better_mp[better] += 1
            else:
                better_mp[better] = 1
            if better in total_mp:
                total_mp[better] += 1
            else:
                total_mp[better] = 1
            # logging.info(f"[test instance {instance} sc value]: {algo_map[a]} sc_ab = {sc_ab}, {algo_map[b]} sc_ba = {sc_ba}, better = {algo_map[better]}")
            logging.info(
                f"[test instance {instance} sc value]: {a} sc_ab = {sc_ab}, {b} sc_ba = {sc_ba}, better = {better}")
            logging.info("===============================")
        for algo, count in better_mp.items():
            logging.info(f"algo {algo} better count {count}")
        logging.info("============================")
        sc_results.append(r)

    for algo, count in total_mp.items():
        logging.info(f"total count: algo {algo} total better count {count}")

    with open(save_file, 'w') as file:
        # 第一行保存测试问题
        first_row = '\t'.join(test_ins)
        file.write(first_row+'\n')
        # 第二行保存对比算法组合
        second_row = [f'({a},{b})\t({b},{a})' for a, b in comb_algorithm]
        # second_row = [f'({algo_map[a]},{algo_map[b]})\t({algo_map[b]},{algo_map[a]})' for a, b in comb_algorithm]
        file.write('\t'.join(second_row)+'\n')
        for insrow in sc_results:
            insrow = [round(num, 5) for num in insrow]
            file.write('\t '.join(map(str, insrow))+'\n')


def CompareGD(algorithm: [str], test_ins: [str], igd_save_file: str, gd_save_file: str, result_dir: str):
    igd_results = []
    gd_results = []
    logging.info(f"alogorithm sequence {algorithm}")
    igd_better_mp = {}
    gd_better_mp = {}
    algo_map = {'new_nsgals_2': 'nsgals', 'nsga3': 'new_nsgals_2', 'nsgals': 'nsgals_1', 'nsgals_1': 'nsga3'}
    for instance in test_ins:
        igd_row = []
        gd_row = []
        popfun_array = []
        for algo in algorithm:
            dataname = instance.split('.')[0] + '.npz'
            datadir = '/'.join([result_dir, algo, dataname])
            popfun = np.load(datadir)["arr_1"]
            popfun_array.append(popfun)

        P = np.concatenate([obj for obj in popfun_array], axis=0)
        v_max = P.max(axis=0)
        v_min = P.min(axis=0)
        # 归一化
        P = (P - np.tile(v_min, (P.shape[0], 1))) / (np.tile(v_max, (P.shape[0], 1)) - np.tile(v_min, (P.shape[0], 1)) + 1e-6)
        norm_popfun = []
        begin = 0
        # norm_objs记录对比方法的目标函数归一化后的目标函数值
        for popfun in popfun_array:
            norm_popfun.append(P[begin:(begin + popfun.shape[0])])
            begin = begin + popfun.shape[0]
        pareto = get_ps(P)  # 得到对比方法组成的pareto最优解

        for i, norm_obj in enumerate(norm_popfun):
            gd = GD(norm_obj, pareto) # 计算算法的GD值
            igd = IGD(pareto, norm_obj) # 计算算法的IGD值
            gd_row.append(gd)
            igd_row.append(igd)

        # 找出gd指标最好的算法
        best_gd_algo = algorithm[np.argmin(np.array(gd_row))]
        # 找出igd指标最好的算法
        best_igd_algo = algorithm[np.argmin(np.array(igd_row))]

        # 更新gd统计数据
        if best_gd_algo in gd_better_mp:
            gd_better_mp[best_gd_algo] += 1
        else:
            gd_better_mp[best_gd_algo] = 1
        # 更新igd统计数据
        if best_igd_algo in igd_better_mp:
            igd_better_mp[best_igd_algo] += 1
        else:
            igd_better_mp[best_igd_algo] = 1

        # logging.info(f"instance {instance} gd values {gd_row} best {algo_map[best_gd_algo]}")
        # logging.info(f"instance {instance} igd values {igd_row} best {algo_map[best_igd_algo]}")
        logging.info(f"instance {instance} gd values {gd_row} best {best_gd_algo}")
        logging.info(f"instance {instance} igd values {igd_row} best {best_gd_algo}")
        logging.info("===============================")
        igd_results.append(igd_row)
        gd_results.append(gd_row)

    for algo, count in igd_better_mp.items():
        # logging.info(f"In all instance igd total better count, algo {algo_map[algo]} count {count}")
        logging.info(f"In all instance igd total better count, algo {algo} count {count}")

    logging.info("===============================")
    for algo, count in gd_better_mp.items():
        # logging.info(f"In all instance gd total better count, algo {algo_map[algo]} count {count}")
        logging.info(f"In all instance gd total better count, algo {algo} count {count}")

    logging.info("===============================")
    with open(igd_save_file, 'w') as file:
        # 第一行保存测试问题
        first_row = '\t'.join(test_ins)
        file.write(first_row + '\n')
        # 算法顺序
        # algos = [algo_map[algo] for algo in algorithm]
        file.write(' '.join(algorithm) + '\n')
        for insrow in igd_results:
            insrow = [round(num, 5) for num in insrow]
            file.write(' '.join(map(str, insrow))+'\n')

    with open(gd_save_file, 'w') as file:
        # 第一行保存测试问题
        first_row = '\t'.join(test_ins)
        file.write(first_row + '\n')
        # 算法顺序
        # algos = [algo_map[algo] for algo in algorithm]
        file.write(' '.join(algorithm)+'\n')
        for insrow in gd_results:
            insrow = [round(num, 5) for num in insrow]
            file.write(' '.join(map(str, insrow))+'\n')


if __name__ == "__main__":
    algorithm = compare_algorithm[2]
    # 测试数据、对比数据保存路径、sc结果保存路径、gd文件路径、igd文件路径
    testdir, result, sc_file, gd_file, igd_file, log_file = exeperiment_files[1]
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    testins = os.listdir(testdir)
    CompareSC(algorithm, testins, sc_file, result)
    CompareGD(algorithm, testins, igd_file, gd_file, result)