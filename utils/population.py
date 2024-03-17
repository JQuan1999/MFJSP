import copy
import math

import numpy as np

from enum import Enum
from .schedule import MachineArray, JobArray

class EnumRule(Enum):
    RM_RULE = 0  # 随机
    LML_RULE = 1  # 最小机器负载
    SPT_RULE = 2  # 最短加工时间
    LEC_RULE = 3  # 最小机器能耗


class Chromosome:
    def __init__(self, machine_num: int,
                 rule: EnumRule,
                 job_array: JobArray,
                 machine_array: MachineArray
                 ):
        self.machine_num = machine_num
        self.rule = rule
        self.job_array = copy.deepcopy(job_array)
        self.machine_array = copy.deepcopy(machine_array)

        self.opnum_array = self.job_array.opnum_array
        self.presum_opnum = np.cumsum(self.opnum_array)

        self.job_num = len(self.opnum_array)
        self.chrome_len = self.job_array.GetOperationNum()
        self._jc = None
        self._mc = None
        self._init_jc()  # 初始化工序编码部分
        self._init_mc(rule)  # 初始化机器编码部分

    def FindIdxForMachinePos(self, idx) -> [int, int]:
        """
            根据机器编码串idx位置 返回属于该位置的工件和工序索引
        :param idx: 机器编码索引位置
        :return:
        """
        job_idx = 0
        op_idx = 0
        for i in range(self.job_num):
            if self.presum_opnum[i] >= idx + 1:
                job_idx = i
                if i == 0:
                    op_idx = idx
                else:
                    op_idx = idx - self.presum_opnum[i - 1]
                break
        return job_idx, op_idx

    # 根据工件和工序编号找到机器编码位置
    def FindMachineIdx(self, job_idx, op_idx):
        idx = op_idx
        if job_idx == 0:
            return idx
        else:
            idx += self.presum_opnum[job_idx - 1]
            return idx

    def _init_mc(self, rule):
        self._mc = np.full(self.chrome_len, -1, dtype=int)

        if rule == EnumRule.RM_RULE:
            # 随机初始化
            for i in range(self.chrome_len):
                job_idx, op_idx = self.FindIdxForMachinePos(i)
                # 获取job_array[job_idx][op_idx]的加工信息
                ava_machine_k, ava_machines, _, _ = self.job_array.GetOperationMachines(job_idx, op_idx)
                self._mc[i] = ava_machines[np.random.randint(0, ava_machine_k)]
        elif rule == EnumRule.LML_RULE:
            # 最小机器负载初始化
            order = np.random.permutation(self.job_num)
            # 机器累计负载
            workload = np.zeros(self.machine_num, int)
            for job_idx in order:
                for op_idx in range(self.opnum_array[job_idx]):
                    ava_machine_k, ava_machines, ptimes, _ = self.job_array.GetOperationMachines(job_idx, op_idx)
                    inf = 1e10
                    tmp_workload = np.ones_like(workload) * inf # 临时的数组记录当前工序机器的累计负载
                    for ava_machine_idx in ava_machines:
                        tmp_workload[ava_machine_idx] = 0

                    for i in range(ava_machine_k):
                        tmp_workload[ava_machines[i]] = workload[ava_machines[i]] + ptimes[i]  # 加上对应机器的加工时间
                    # 选择可用机器中累计负载最小的机器编号
                    choose_machine = np.argmin(tmp_workload)
                    # 选择出该机器在可用机器集合里的索引位置
                    array_idx = np.where(ava_machines == choose_machine)[0][0]
                    # 计算出工序的索引位置
                    idx = self.job_array.GetOperationAbsoulteIdx(job_idx, op_idx)
                    self._mc[idx] = choose_machine
                    workload[choose_machine] += ptimes[array_idx]  # 更新累计负载
        elif rule == EnumRule.SPT_RULE:
            # 最小加工时间初始化
            for i in range(self.chrome_len):
                job_idx, op_idx = self.FindIdxForMachinePos(i)
                # 获取job_array[job_idx][op_idx]的可用机器数目
                ava_machine_m, ava_machines, ptimes, _ = self.job_array.GetOperationMachines(job_idx, op_idx)
                self._mc[i] = ava_machines[np.argmin(ptimes)]
        else:
            # 最小加工能耗初始化
            for i in range(self.chrome_len):
                job_idx, op_idx = self.FindIdxForMachinePos(i)
                # 获取job_array[job_idx][op_idx]的可用机器数目
                ava_machine_m, ava_machines, _, ecosts = self.job_array.GetOperationMachines(job_idx, op_idx)
                self._mc[i] = ava_machines[np.argmin(ecosts)]
        self.CheckMC()

    def _init_jc(self):
        self._jc = np.array([], dtype=int)
        for i in range(len(self.opnum_array)):
            self._jc = np.append(self._jc, np.full(self.opnum_array[i], i, dtype=int))  # 生成工序编码
        self._jc = np.random.permutation(self._jc)  # 打乱顺序

    def Check(self):
        self.CheckJC()
        self.CheckMC()

    def CheckJC(self):
        # 1. 检查工件出现次数是否一致
        vis_times = np.zeros(self.job_num, dtype=int)
        for job_idx in self._jc:
            vis_times[job_idx] += 1
        for i in range(self.job_num):
            assert self.opnum_array[i] == vis_times[i]

    def CheckMC(self):
        # 1. 检查工件出现次数是否一致
        # 2. 检查工序的可用机器是否包含该索引
        vis_times = np.zeros(self.job_num, dtype=int)
        for pos in range(self.chrome_len):
            job_idx, op_idx = self.FindIdxForMachinePos(pos)
            vis_times[job_idx] += 1
            _, ava_machines, _, _ = self.job_array.GetOperationMachines(job_idx, op_idx)
            find = False
            for midx in ava_machines:
                if midx == self._mc[pos]:
                    find = True
                    break
            if find == False:
                debug_info = [idx+1 for idx in ava_machines]
                print(f"工序 {job_idx+1}-{op_idx+1} 可用机器集合{debug_info} 编码机器为 {self._mc[pos]+1}")
            assert find == True
        for i in range(self.job_num):
            assert self.opnum_array[i] == vis_times[i]

    def GetMachineCode(self) -> [np.ndarray]:
        return self._mc

    def GetJobCode(self) -> [np.ndarray]:
        return self._jc

    def ChangeCode(self, mcode, jcode):
        self.SetMachineCode(mcode)
        self.SetJobCode(jcode)

    def SetJobCode(self, jcode: np.ndarray):
        self._jc = jcode

    def SetMachineCode(self, mcode: np.ndarray):
        self._mc = mcode

    def ResetGraphVar(self):
        self.job_array.ResetJobGraphVar()

    def ResetScheduleVar(self):
        self.machine_array.ResetScheduleVar()

    def CalObjective(self, objn: int):
        self.decode()
        # 检查机器上的编码和加工时间信息
        # self._printDebugInfo()
        cmax = self.job_array.GetCompleteTime()
        # worload, critical_workload, energy_cost = self.machine_array.CalObjective()
        ildetime, critical_workload, energy_cost = self.machine_array.CalObjective()
        objs = [cmax, ildetime, critical_workload, energy_cost]
        return objs[:objn]

    def decode(self):
        self.ResetScheduleVar()
        # 解码计算目标函数值
        op_idices = np.zeros(self.job_num, dtype=int)

        for job_idx in self._jc:
            # 1. 按顺序找出当前工序对应的机器编号
            op_idx = op_idices[job_idx]
            op_idices[job_idx] += 1
            idx = self.FindMachineIdx(job_idx, op_idx)
            mk = self._mc[idx]
            # 2. 将工序插入机器上
            self.machine_array.Sequence(mk, job_idx, op_idx, self.job_array)

    def _printDebugInfo(self):
        job_code = self._jc + 1
        print(f"_jc {job_code.tolist()}")
        op_ididces = [0] * self.job_num
        # nc保存每个位置 工件和机器一对一分配关系
        nc = [-1] * len(self._jc)
        for i, job_idx in enumerate(self._jc):
            op_idx = op_ididces[job_idx]
            op_ididces[job_idx] += 1
            abs_idx = self.job_array.GetOperationAbsoulteIdx(job_idx, op_idx)
            nc[i] = self._mc[abs_idx] + 1
        print(f"_mc {nc}")
        # self.machine_array.DebugInfo()
        self.machine_array.Gatta(t=100)
        print("======================")


class Population:
    def __init__(self,
                 N: int,  # 种群大小
                 machine_num: int,  # 加工机器大小
                 job_array: JobArray,
                 machine_array: MachineArray,
                 ratio: [float] = None,
                 objective_num: int = 4,
                 ):
        self.N = N
        self.machine_num = machine_num
        self.job_array = job_array
        self.machine_array = machine_array
        self.job_num = job_array.JobNum()
        if ratio is None:
            self.ratio = [0.25] * 4
        else:
            self.ratio = ratio
        self.objn = objective_num
        self.popfun = np.array([])
        self.offfun = np.array([])
        self.pop: [Chromosome] = []
        self.offspring: [Chromosome] = []
        self.obj_curve = []

    def InitPopulation(self, rules=None) -> np.ndarray:
        """
        :param rules: 按rules初始化种群
        :return: 返回初始化种群的目标函数值
        """
        if rules is None:
            rules = [EnumRule.LML_RULE, EnumRule.SPT_RULE, EnumRule.LEC_RULE, EnumRule.RM_RULE]
        sizes = [0] * len(self.ratio)

        assert len(sizes) == len(rules)

        for i, r in enumerate(self.ratio):
            if i != len(self.ratio) - 1:
                sizes[i] = round(float(self.N) * r)
            else:
                sizes[i] = self.N - sum(sizes)

        for i, size in enumerate(sizes):
            for j in range(size):
                self.pop.append(Chromosome(self.machine_num, rules[i], self.job_array, self.machine_array))

        self.popfun = self.CalParentfunc(self.objn)
        init_ave = np.average(self.popfun, axis=0).tolist()
        self.AppendRecord(init_ave)
        return self.popfun

    def InitRandomPopulation(self):
        rules = [EnumRule.RM_RULE] * len(self.ratio)
        self.InitPopulation(self, rules)

    def Size(self):
        return len(self.pop)

    def GetJobNum(self):
        return self.job_num

    def GetParentChrome(self, idx) -> [Chromosome]:
        return self.pop[idx]

    def GetChildChrome(self, idx):
        return self.offspring[idx]

    def UpdateParent(self, pop: [Chromosome]):
        self.pop = pop

    def UpdateOffSpring(self, off: [Chromosome]):
        self.offspring = off

    def UpdateOneOffSprint(self, pos: int, off: Chromosome):
        self.offspring[pos] = off

    def GenerateNextPop(self, idices: np.ndarray, mix_popfun: np.ndarray) -> [[Chromosome], np.ndarray]:
        """
        :param idices: 下一代种群索引
        :param mix_popfun: 融合种群的目标函数
        :return:
        """
        merge_chromes = np.array([*self.pop, *self.offspring])
        self.pop = merge_chromes[idices].tolist()
        self.offspring.clear()
        self.popfun = mix_popfun[idices] # 更新种群目标函数
        ave_obj = np.average(self.popfun, axis=0).tolist()
        self.AppendRecord(ave_obj)
        return self.pop, self.popfun

    def _calObjective(self, chromes: [Chromosome], objn):
        n = len(chromes)
        popfun = []
        for i in range(n):
            objv = chromes[i].CalObjective(objn)
            popfun.append(objv)
        return np.array(popfun)

    def CalParentfunc(self, objn) -> np.ndarray:
        self.popfun = self._calObjective(self.pop, objn)
        return self.popfun

    def CalPartChildfunc(self, objn: int, idices: [int]):
        chromes = np.array(self.offspring)[idices]
        return self._calObjective(chromes, objn)

    def CalChildfunc(self, objn: int):
        self.offfun = self._calObjective(self.offspring, objn)
        return self.offfun

    def CalMergePop(self, objn, idices: [int]):
        merge_chromes = np.array([*self.pop, *self.offspring])
        choose_chromes = merge_chromes[idices].tolist()
        return self._calObjective(choose_chromes, objn)

    def AppendRecord(self, record: [int]):
        self.obj_curve.append(record)

    def Save(self, file: str):
        obj_values = np.array(self.obj_curve)
        popfun = self.popfun
        code_len = self.job_array.OperationNum()
        jc_code = np.array([], dtype=int)
        mc_code = np.array([])
        for chrome in self.pop:
            jc_code = np.concatenate((jc_code, chrome.GetJobCode()))
            mc_code = np.concatenate((mc_code, chrome.GetMachineCode()))
        jc_code = jc_code.reshape((-1, code_len))
        mc_code = mc_code.reshape((-1, code_len))
        np.savez(file, obj_values, popfun, jc_code, mc_code)


def Normalize(popfun: np.ndarray, zmin: np.ndarray, zmax:  np.ndarray) -> np.ndarray:
    if len(popfun.shape) == 1:
        popfun = popfun.reshape(1, -1)
    m, n = popfun.shape
    zmax_fun = np.tile(np.array(zmax).reshape(1, n), (m, 1))
    zmin_fun = np.tile(np.array(zmin).reshape(1, n), (m, 1)) + 1e-6
    norm_popfun = (popfun - zmin_fun) / (zmax_fun - zmin_fun)
    if norm_popfun.shape[0] == 1:
        norm_popfun = norm_popfun.reshape(-1)
    return norm_popfun


def GetMaxAndMinObj(popfun: np.ndarray, zmin, zmax) -> [np.ndarray, np.ndarray]:
    zmin = np.min(np.vstack((zmin, popfun)), axis=0).reshape(1, -1)
    # 求出最大值点
    zmax = np.max(np.vstack((zmax, popfun)), axis=0).reshape(1, -1)
    return zmin, zmax