import math
import random

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文

color = [0.77, 0.18, 0.78,
         0.21, 0.33, 0.64,
         0.88, 0.17, 0.56,
         0.20, 0.69, 0.28,
         0.26, 0.15, 0.47,
         0.83, 0.27, 0.44,
         0.87, 0.85, 0.42,
         0.85, 0.51, 0.87,
         0.99, 0.62, 0.76,
         0.52, 0.43, 0.87,
         0.00, 0.68, 0.92,
         0.26, 0.45, 0.77,
         0.98, 0.75, 0.00,
         0.72, 0.81, 0.76,
         0.77, 0.18, 0.78,
         0.28, 0.39, 0.44,
         0.22, 0.26, 0.24,
         0.64, 0.52, 0.64,
         0.87, 0.73, 0.78,
         0.94, 0.89, 0.85,
         0.85, 0.84, 0.86]

class Operation:
    def __init__(self,
                 job_idx: int = -1,
                 op_idx: int = -1,
                 k: int = -1,
                 low: int = -1,
                 up: int = -1,
                 marray=None,
                 parray=None):
        """
        :param job_idx: 所属工件索引
        :param op_idx: 工序索引
        :param k: 可用加工机器数量
        :param low: 加工能耗生成参数
        :param up: 加工能耗生成参数
        :param marray: 可用加工机器
        :param parray: 加工能耗
        """
        if parray is None:
            parray = []
        if marray is None:
            marray = []
        self.job_idx = job_idx
        self.op_idx = op_idx
        self.k = k
        self.low = low
        self.up = up
        self.marray = np.array(marray, dtype=int)
        self.parray = np.array(parray, dtype=int)
        self._init_schedule_variale()
        self._init_graph_variable()
        self._init_energy_cost()

    def _init_energy_cost(self):
        earray: list = []
        for i in range(0, len(self.parray)):
            cost = self.up + random.randint(round(self.low / 2), self.low) - self.parray[i]  # 生成能耗
            earray.append(cost)
        self.earray = np.array(earray)

    def _init_graph_variable(self):
        self.job_next: Operation = None
        self.job_rnext: Operation = None
        self.machine_next: Operation = None
        self.machine_rnext: Operation = None
        self.pre_cnt = 0
        self.rpre_cnt = 0
        self.ebegin_time = 0
        self.lbegin_time = math.inf

    def _init_schedule_variale(self):
        self.begin_t = 0
        self.end_t = 0
        self.process_time = 0
        self.choose_machine_idx: int = -1
        self.choose_array_idx: int = -1
        self.is_scheduled = 0

    def ResetGraphVar(self):
        self.machine_next = None
        self.machine_rnext = None
        self.ebegin_time = 0
        self.pre_cnt = 0
        self.rpre_cnt = 0
        self.lbegin_time = math.inf

    def ResetScheduleVar(self):
        self._init_schedule_variale()

    def Set(self, begin_t, end_t, choose_idx):
        self.begin_t = begin_t
        self.end_t = end_t
        self.process_time = self.end_t - self.begin_t
        self.choose_machine_idx = choose_idx
        find = False
        for idx, midx in enumerate(self.marray):
            if midx == choose_idx:
                find = True
                self.choose_array_idx = idx
                break
        assert find == True
        self.is_scheduled = 1

    # TODO 设置插入机器、开始加工时间和完工时间
    def InsertMachine(self, machine_idx: int):
        """
            工序插入新机器
        :param machine_idx:
        :return:
        """
        pos = np.where(self.marray == machine_idx)[0]
        assert len(pos) == 1
        self.choose_machine_idx = machine_idx # 设置新的加工机器
        for idx, midx in enumerate(self.marray):
            if midx == machine_idx:
                self.choose_array_idx = idx # 设置该机器在可用机器集合中的索引位置

    def GetJobIdx(self):
        return self.job_idx

    def GetOperationIdx(self):
        return self.op_idx

    def GetMachines(self) -> [int, np.ndarray, np.ndarray, np.ndarray]:
        return self.k, self.marray, self.parray, self.earray

    def GetProcessTime(self, idx):
        idx = np.where(self.marray == idx)[0]
        assert len(idx) == 1
        return self.parray[idx[0]]

    def GetEnergyCost(self, idx):
        idx = np.where(self.marray == idx)[0]
        assert len(idx) == 1
        return self.earray[idx[0]]

    def GetECompleteTime(self):
        return self.ebegin_time + self.process_time

    def GetLStartTime(self):
        return self.lbegin_time

    def GetNext(self):
        pass


class Job:
    def __init__(self,
                 job_idx: int,
                 opnums: int,
                 low: int,
                 up: int,
                 karray: [],
                 mdoubleArray: [[]],
                 pdoubleArray: [[]]
                 ):
        self.operations: [Operation] = []
        self.job_idx: int = job_idx
        self.opnums = opnums
        self.low = low
        self.up = up
        self.karray = karray
        self.mdoubleArray = mdoubleArray
        self.pdoubleArray = pdoubleArray
        self._init_operation()

    def _init_operation(self):
        for i in range(0, self.opnums):
            self.operations.append(Operation(self.job_idx, i, self.karray[i], self.low, self.up, self.mdoubleArray[i],
                                             self.pdoubleArray[i]))

    def InitGraph(self):
        for i in range(0, self.opnums):
            self.operations[i].pre_cnt = 0
            self.operations[i].rpre_cnt = 0
            self.operations[i].ebegin_time = 0
            self.operations[i].lbegin_time = math.inf

        for i in range(0, self.opnums):
            if i != self.opnums - 1:
                self.operations[i].job_next = self.operations[i + 1]
                self.operations[i+1].pre_cnt += 1 # i+1正向入度+1
            if i != 0:
                self.operations[i].job_rnext = self.operations[i - 1]
                self.operations[i-1].rpre_cnt += 1 # i-1反向入度+1

    def GetOperationMachines(self, idx) -> [int, np.ndarray, np.ndarray, np.ndarray]:
        return self.operations[idx].GetMachines()

    def GetOperationFinishTime(self, idx):
        return self.operations[idx].end_t

    def GetOperation(self, idx):
        return self.operations[idx]

    def GetJobCompleteTime(self):
        op = self.operations[-1]
        return op.end_t


class JobArray:
    def __init__(self, jobs: [Job],
                 opnum_array,
                 job_karrays,
                 job_marrays,
                 job_parrays):
        self.jobs: [Job] = jobs
        self.opnum_array = opnum_array
        self.job_karrays = job_karrays
        self.jobs_marrays = job_marrays
        self.job_parrays = job_parrays

    def GetOperationNum(self) -> [int]:
        return np.sum(self.opnum_array)

    def GetOperationMachines(self, job_idx, op_idx) -> [int]:
        """
        :return:
            可用机器数目, 可用机器编号集合, 机器处理时间, 机器加工能耗
        """
        return self.jobs[job_idx].GetOperationMachines(op_idx)

    def InitGraph(self):
        for job in self.jobs:
            job.InitGraph()

    def JobNum(self):
        return len(self.jobs)

    def OperationNum(self):
        return sum(self.opnum_array)

    def GetOperationNumArray(self):
        return self.opnum_array

    def GetJobOperationFinishTime(self, job_idx, op_idx):
        """
        :param job_idx: 工件编号
        :param op_idx: 工序编号
        :return:  返回工序O_{ij}的结束时间 0或c_{ij}
        """
        assert job_idx >= 0 and op_idx >= 0
        return self.jobs[job_idx].GetOperationFinishTime(op_idx)

    def GetOperation(self, job_idx, op_idx) -> Operation:
        return self.jobs[job_idx].GetOperation(op_idx)

    def GetCompleteTime(self):
        ctime = []
        for job in self.jobs:
            ctime.append(job.GetJobCompleteTime())
        return max(ctime)

    def ResetJobGraphVar(self):
        for job in self.jobs:
            for op in job.operations:
                op.ResetScheduleVar()

    def GetOperationAbsoulteIdx(self, job_idx, op_idx) -> int:
        """
            获取工序在所有工序中的绝对顺序
        :param job_idx:
        :param op_idx:
        :return:
        """
        presum = np.cumsum(self.opnum_array)
        if job_idx == 0:
            return op_idx
        else:
            return presum[job_idx - 1] + op_idx


class Machine:
    def __init__(self, idx: int):
        self.sequence: [Operation] = []  # 工序序列
        self.begin_times: [int] = []  # 工序开始加工时间
        self.end_times: [int] = []  # 工序结束加工时间
        self.costs: [int] = []  # 工序能耗
        self.idx = idx

    def reset_variable(self):
        # for op in self.sequence:
        #     op.ResetScheduleVar()
        self.sequence.clear()
        self.begin_times.clear()
        self.end_times.clear()
        self.costs.clear()

    def greedy_insert(self, job_idx, op_idx, job_array: JobArray):
        """
        :param job_idx: 工序索引
        :param op_idx: 工件索引
        :return:
        """
        op = job_array.GetOperation(job_idx, op_idx)
        job_preop_t = 0
        if op_idx == 0:
            job_preop_t = 0
        else:
            job_preop_t = job_array.GetJobOperationFinishTime(job_idx, op_idx - 1)  # 工件上紧前工序的完工时间
        mach_preop_t = 0 if len(self.end_times) == 0 else self.end_times[-1]  # 机器上紧前工序的完工时间
        if mach_preop_t == 0:
            self._insert(job_preop_t, job_idx, op) # 在机器末尾插入
        else:
            # 在中间寻找插入位置
            pt = op.GetProcessTime(self.idx)
            found = False
            # 0, 1, ..., n 一共有n个插入位置
            # 插入位置i需要满足 max(end_{i-1}, end_{idx-1}) + pt <= begin_{i}
            for i, end in enumerate(self.end_times):
                machine_preop_t = 0 # 插入位置i的前一工序结束加工时间
                if i != 0:
                    machine_preop_t = self.end_times[i-1]
                begin_t = max(job_preop_t, machine_preop_t)
                end_t = self.begin_times[i] # 插入位置工序的开始时间
                if end_t - begin_t >= pt:
                    # 满足插入条件
                    self._insert_at(i, begin_t, job_idx, op)
                    found = True
                    break
            if not found:
                # 在最尾端插入
                begin_t = max(job_preop_t, mach_preop_t)
                self._insert(begin_t, job_idx, op)
            # begin_t = max(job_preop_t, mach_preop_t)
            # self._insert(begin_t, job_idx, op)

    def _insert_at(self, pos, begin_t, job_idx, op: Operation):
        """
        :param pos: 插入位置
        :param begin_t: 开始加工时间
        :param job_idx: 插入工件索引
        :param op: 工序
        :return:
        """
        ptime = op.GetProcessTime(self.idx)
        ecost = op.GetEnergyCost(self.idx)
        # op插入pos位置
        self.sequence.insert(pos, op)
        self.begin_times.insert(pos, begin_t)
        self.end_times.insert(pos, begin_t + ptime)
        self.costs.insert(pos, ecost)
        op.Set(begin_t, begin_t + ptime, self.idx)

    def _insert(self, begin_t, job_idx, op: Operation):
        """
            在机器加工顺序末尾插入工序
        :param begin_t: 开始加工时间
        :param job_idx: 工件索引
        :param op: 插入工序
        :return:
        """
        pos = len(self.sequence)
        self._insert_at(pos, begin_t, job_idx, op)

    def GetBeginTime(self):
        if len(self.begin_times) == 0:
            return -1
        else:
            return 0

    def CalWorkload(self):
        pt = [self.end_times[i] - self.begin_times[i] for i in range(len(self.sequence))]
        return sum(pt)

    def CalEnergyCost(self):
        return sum(self.costs)

    def CalIdleTime(self):
        """
            计算机器上的空闲时间
        :return:
        """
        if len(self.sequence) == 0:
            return 0
        idle_times = self.begin_times[0]
        for i in range(1, len(self.sequence)):
            idle_times += self.begin_times[i] - self.end_times[i-1]
        return idle_times


class MachineArray:
    def __init__(self, machine_num: int):
        self.m = machine_num
        self.machines: [Machine()] = [Machine(i) for i in range(self.m)]
        self.color = np.array(color).reshape((-1, 3))

    def GetMachineKBeginTime(self, k):
        return self.machines[k].GetBeginTime()

    def Sequence(self, k, job_idx, op_idx, job_array):
        self.machines[k].greedy_insert(job_idx, op_idx, job_array)

    def CalWorkLoad(self):
        pass

    def CalCriticalWorkload(self):
        pass

    def EnergyCost(self):
        pass

    def ResetScheduleVar(self):
        for i in range(self.m):
            self.machines[i].reset_variable()

    def InitGraph(self) -> [[[Operation]]]:
        sequences: [[Operation]] = self.GetJobSequence()  # 获取机器上的加工顺序
        # 重置机器上的边
        for seq in sequences:
            for op in seq:
                op.machine_next = None
                op.machine_rnext = None
        for seq in sequences:
            for i in range(0, len(seq)):
                # 添加机器上的相邻边
                if i != len(seq) - 1:
                    # 正向边
                    seq[i].machine_next = seq[i + 1]
                    seq[i+1].pre_cnt += 1 # i+1正向入度+1
                if i != 0:
                    # 反向边
                    seq[i].machine_rnext = seq[i - 1]
                    seq[i-1].rpre_cnt += 1 # i-1反向入度+1
        return sequences

    def Gatta(self, job_count: int, title: str = '', savefile: str = ''):
        """
            绘制机器上的加工甘特图
        :param job_count: 工件数目
        :param title: 甘特图标题
        :param savefile: 甘特图保存文件路径
        :return:
        """
        labels = [f"J{idx+1}" for idx in range(job_count)]
        colormap = plt.get_cmap('RdYlGn')
        category_colors = colormap(np.linspace(0.1, 0.85, job_count))
        label_set = set()
        max_end = 0
        plt.gca().spines['right'].set_color('none')
        plt.gca().spines['top'].set_color('none')
        for i in range(len(self.machines)):
            m = self.machines[i]
            for j in range(len(m.sequence)):
                op = m.sequence[j]
                job_index, op_index = op.job_idx, op.op_idx
                start, end = m.begin_times[j], m.end_times[j]
                pt = end - start
                label = labels[job_index]
                max_end = max(max_end, end)
                if label not in label_set:
                    plt.barh(i + 0.5, pt, height=1, left=start, align='center', color=category_colors[job_index],
                             edgecolor='grey', label=labels[job_index])
                    label_set.add(label)
                else:
                    plt.barh(i + 0.5, pt, height=1, left=start, align='center', color=category_colors[job_index],
                             edgecolor='grey')
                # plt.text(start + pt / 8, i+0.5, '{}-{}\n{}'.format(job_index + 1, op_index + 1, pt), fontsize=10, color='tan')
                # if pt <= 2:
                #     plt.text(start + pt / 8, i + 0.5, '{}-{}\n{}'.format(job_index + 1, op_index + 1, pt), fontsize=10, color='tan')
                # else:
                #     plt.text(start + pt / 2, i + 0.5, '{}-{}\n{}'.format(job_index + 1, op_index + 1, pt), fontsize=10, color='tan')
        plt.title(title, y=1.05, fontsize='small')
        plt.yticks(np.arange(len(self.machines) + 1))
        # plt.ticklabel_format(axis='both', style='sci', scilimits=[-1, 2])
        plt.xlim(0, max_end + max_end / 10)
        plt.xlabel('Time', fontsize='small')
        plt.ylabel('Machine', fontsize='small')
        plt.legend(ncol=2, bbox_to_anchor=(1.15, 1.15), loc='upper right', fontsize='small')
        if savefile != '':
            plt.savefig(savefile, dpi=400, format='png')
        else:
            plt.pause(10)
        plt.close()

    def DebugInfo(self):
        for m in self.machines:
            seq = m.sequence
            debug_info = []
            for op in seq:
                debug_info.append((op.job_idx, op.op_idx, op.begin_t, op.end_t))
            print(f"in machine [{m.idx} seq = [{debug_info}]")

    def CalObjective(self) -> [int, int, int, int]:
        """
            计算机器上的空闲时间、关键机器负载、加工能耗三个目标函数
        :return:
        """
        workload_array = []
        idletime_array = []
        energy_cost = 0
        for i in range(self.m):
            workload_array.append(self.machines[i].CalWorkload())
            idletime_array.append(self.machines[i].CalIdleTime())
            energy_cost += self.machines[i].CalEnergyCost()
        critical_workload = max(workload_array)
        workload = sum(workload_array)
        idletime = sum(idletime_array)
        # return workload, critical_workload, energy_cost
        return idletime, critical_workload, energy_cost

    def GetJobSequence(self) -> [[[Operation]]]:
        """
        :return:
            machine_sequences二维数组所有机器上工件的加工顺序
        """
        machine_sequences = []
        for machine in self.machines:
            machine_sequences.append(machine.sequence)
        return machine_sequences

    def GetJobSequenceIdx(self):
        sequences_idx = []
        for machine in self.machines:
            seq = []
            for op in machine.sequence:
                seq.append((op.job_idx, op.op_idx))
            sequences_idx.append(seq)
        return sequences_idx

    def GetMachineJobSequence(self, idx) -> [[Operation]]:
        """
            返回idx机器上工序
        :param idx:
        :return:
        """
        return self.machines[idx].sequence

    def GetMachineJobSequenceIdx(self, idx):
        """
            返回idx机器上工序的索引
        :param idx: 加工机器索引
        :return:
        """
        sequence = self.GetMachineJobSequence(idx)
        op_idices = []
        for op in sequence:
            op_idices.append((op.job_idx, op.op_idx))
        return op_idices