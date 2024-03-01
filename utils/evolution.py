import copy
import random

import numpy as np

from .population import Chromosome, Population


class AbstractEvolutionOperator:
    def __init__(self, pc, pm: float):
        self.pc: float = pc
        self.pm: float = pm

    # 交叉
    def cross(self, pop: Population):
        pass

    # 变异
    def mutate(self, pop: Population):
        pass


class EvolutionOperator(AbstractEvolutionOperator):
    def __init__(self, pc, pm):
        super(EvolutionOperator, self).__init__(pc, pm)

    def chooseTwo(self, pop: Population):
        idx1 = np.random.randint(0, pop.Size())
        idx2 = idx1
        while idx1 == idx2:
            idx2 = np.random.randint(0, pop.Size())
        return pop.GetParentChrome(idx1), pop.GetParentChrome(idx2)

    # 机器编码部分采用均匀交叉
    def uniform_machine(self, chrome1: Chromosome, chrome2: Chromosome):
        mc1 = chrome1.GetMachineCode()
        mc2 = chrome2.GetMachineCode()
        # 交换次数
        r = np.random.randint(0, len(mc1))
        # 交换位置
        changed = set()
        while len(changed) != r:
            pos = np.random.randint(0, len(mc1)) # 生成交换位置
            if pos in changed:
                continue
            tmp = mc1[pos]
            mc1[pos] = mc2[pos]
            mc2[pos] = tmp
            changed.add(pos)
        chrome1.SetMachineCode(mc1)
        chrome2.SetMachineCode(mc2)
        # 检查是否合法
        chrome1.Check()
        chrome2.Check()

    # 工件编码部分采用IPOX交叉
    def ipox_job(self, job_num: int, chrome1: Chromosome, chrome2: Chromosome):
        jc1 = chrome1.GetJobCode()
        jc2 = chrome2.GetJobCode()
        # 生成两个工件集合
        order = np.random.permutation(job_num)
        seperate = np.random.randint(0, job_num)
        left = order[:seperate]
        right = order[seperate:]
        # 交换属于这两个工件的部分
        child_jc1 = np.full(len(jc1), -1, dtype=int)
        child_jc2 = np.full(len(jc1), -1, dtype=int)

        codeInLeft = np.isin(jc1, left)  # 找出jc1工件编码在left中的元素
        codeInRight = np.isin(jc2, right)  # 找出jc2中工件编码在right中的元素

        child_jc1[codeInLeft] = jc1[codeInLeft]  # 复制jc1编码在left中的元素到child1
        child_jc1[~codeInLeft] = jc2[codeInRight]  # 复制不在left中的元素依次到child1的剩余位置

        child_jc2[codeInRight] = jc2[codeInRight]
        child_jc2[~codeInRight] = jc1[codeInLeft]

        chrome1.SetJobCode(child_jc1)
        chrome2.SetJobCode(child_jc2)

        chrome1.Check()
        chrome2.Check()

    def cross(self, pop: Population):
        nextPop: [Chromosome] = []
        N = pop.Size()
        job_num = pop.GetJobNum()
        while len(nextPop) < N:
            chrome1, chrome2 = self.chooseTwo(pop)
            p1 = copy.deepcopy(chrome1)
            p2 = copy.deepcopy(chrome2)
            r = random.random()
            if r < self.pc:
                self.ipox_job(job_num, p1, p2)
                self.uniform_machine(p1, p2)
            # print(f"pop size= {len(nextPop)}")
            nextPop.append(p1)
            nextPop.append(p2)
        pop.UpdateOffSpring(nextPop)

    # 两点变异
    def _two_point_mutate(self, chrome: Chromosome):
        mc = chrome.GetMachineCode()
        pos1 = np.random.randint(0, len(mc))
        pos2 = pos1
        while pos1 == pos2:
            pos2 = np.random.randint(0, len(mc))

        job_idx, op_idx = chrome.FindIdxForMachinePos(pos1)
        machine_k, ava_machine, _, _ = chrome.job_array.GetOperationMachines(job_idx, op_idx)
        mc[pos1] = ava_machine[np.random.randint(0, machine_k)]

        job_idx, op_idx = chrome.FindIdxForMachinePos(pos2)
        machine_k, ava_machine, _, _ = chrome.job_array.GetOperationMachines(job_idx, op_idx)
        mc[pos2] = ava_machine[np.random.randint(0, machine_k)]
        chrome.SetMachineCode(mc)

        chrome.Check()

    # 两点变异
    def _os_change_mutate(self, chrome: Chromosome):
        jc = chrome.GetJobCode()
        # 随机选择变异的基因工序
        pos1 = np.random.randint(0, len(jc))
        pos2 = pos1
        while pos1 == pos2:
            pos2 = np.random.randint(0, len(jc))
        tmp = jc[pos1]
        jc[pos1] = jc[pos2]
        jc[pos2] = tmp
        chrome.SetMachineCode(jc)
        # 修正mc对应的机器号
        chrome.Check()

    # 工序顺序变异
    def _os_order_mutate(self, chrome: Chromosome):
        jc = chrome.GetJobCode()
        # 随机选择变异的基因工序
        pos = np.random.randint(0, len(jc))
        job_idx = jc[pos]
        left = -1
        for i in reversed(range(0, pos)):
            if jc[i] == job_idx:
                left = i
                break
        right = len(jc)
        for i in range(pos+1, len(jc)):
            if jc[i] == job_idx:
                right = i
                break
        # 随机生成left+1, right-1之间的位置
        if left + 1 >= right - 1:
            return
        change_pos = pos
        while change_pos == pos:
            change_pos = np.random.randint(left+1, right)
        if change_pos < pos:
            # 将[change_pos, pos-1]区间移动到[change_pos+1, pos]
            # 即 [change_pos+1:pos+1] = [change_pos:pos]
            jc[change_pos+1:pos+1] = jc[change_pos:pos]
            jc[change_pos] = job_idx
        else:
            # 将[pos+1, change_pos]左移到[pos, change_pos-1]
            # 即[pos:change_pos] = [pos+1:change_pos+1]
            jc[pos: change_pos] = jc[pos+1: change_pos+1]
            jc[change_pos] = job_idx
        chrome.SetJobCode(jc)
        # 检查是否合法
        chrome.Check()

    def mutate(self, pop: Population):
        N = pop.Size()
        for i in range(N):
            chrome = pop.GetChildChrome(i)
            r = random.random()
            if r < self.pm:
                self._os_order_mutate(chrome)
                self._two_point_mutate(chrome)
