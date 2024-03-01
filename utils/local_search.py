import copy
import math

import numpy as np

from .population import *
from .schedule import Operation


class Graph:
    def __init__(self, objn: int, weight_vector: [float] = None, chrome: Chromosome = None):
        self.objn = objn
        self.weight = weight_vector
        self.chrome = chrome
        self.graph_size = chrome.chrome_len
        self.cmax = -1
        self.init_op = []
        self.rinit_op = []
        self.graph_update = False
        self.InitGraph()

    def ResetGraphVar(self):
        self.init_op.clear()
        self.rinit_op.clear()
        self.cmax = -1

    def CheckOperationDegree(self):
        for op in self.graph:
            assert op.pre_cnt <= 2 and op.pre_cnt >= 0
            assert op.rpre_cnt <= 2 and op.rpre_cnt >= 0

    # TODO 优化工序图相关变量重置
    # 利用染色体编码构建graph
    def InitGraph(self):
        assert self.graph_update == False
        self.ResetGraphVar() # 重置变量

        self.chrome.decode()
        self.chrome.job_array.InitGraph()
        sequences = self.chrome.machine_array.InitGraph()
        self.graph = [Operation()] * self.graph_size
        for seq in sequences:
            for i in range(0, len(seq)):
                idx = self.chrome.job_array.GetOperationAbsoulteIdx(seq[i].job_idx, seq[i].op_idx) # 绝对工序索引位置
                self.graph[idx] = seq[i]
                if seq[i].pre_cnt == 0:
                    self.init_op.append(seq[i])  # 正向图入度为0
                if seq[i].rpre_cnt == 0:
                    self.rinit_op.append(seq[i])  # 反向图入度为0
        self.CheckOperationDegree()
        self.ChcekOperationTime()

    def topological_sort(self) -> [[Operation]]:
        queue: [Operation] = self.init_op
        for op in queue:
            op.ebegin_time = 0  # 虚拟起始节点出发，初始节点的最早开始时间设置为0

        visited = [] # 工序的访问序列
        visited_flag = [False] * self.graph_size

        while len(queue) != 0:
            nextq = []
            for op in queue:
                idx = self.chrome.job_array.GetOperationAbsoulteIdx(op.job_idx, op.op_idx)
                assert visited_flag[idx] == False # assert失败表示有环
                visited_flag[idx] = True
                visited.append(op)

                # op.ebegin_time + op.parray[op.choose_array_idx] = op的最早完工时间
                self.cmax = max(self.cmax, op.ebegin_time + op.parray[op.choose_array_idx])  # 更新完工时间
                for next_op in [op.job_next, op.machine_next]:
                    if next_op is not None:
                        begin_t = next_op.ebegin_time
                        pt = op.parray[op.choose_array_idx]  # op -> next_op连接边的权值
                        next_op.ebegin_time = max(begin_t, op.ebegin_time + pt)  # 更新next_op的最早开始时间
                        next_op.pre_cnt -= 1  # 入度数目-1
                        if next_op.pre_cnt == 0:
                            nextq.append(next_op)  # 加入下一层
            queue = nextq  # 访问下一层
        assert len(visited) == self.graph_size
        self._check_sort_pre_cnt()
        # visited_sequence = [(op.job_idx+1, op.op_idx+1) for op in visited]
        # print(f"top sort sequence: {visited_sequence}") # debug info
        return visited

    def _check_sort_pre_cnt(self):
        count = [0] * self.graph_size
        check_failed = False
        operation_info = []
        operation_neighbor_info = [[] for _ in range(self.graph_size)]
        for i, op in enumerate(self.graph):
            # 检查拓扑排序后的正向入度
            if op.pre_cnt != 0:
                check_failed = True
            count[i] = op.pre_cnt
        if check_failed:
            for i, op in enumerate(self.graph):
                operation_info.append((op.job_idx+1, op.op_idx+1))
                if op.job_next is not None:
                    operation_neighbor_info[i].append((op.job_next.job_idx+1, op.job_next.op_idx+1))
                if op.machine_next is not None:
                    operation_neighbor_info[i].append((op.machine_next.job_idx + 1, op.machine_next.op_idx + 1))
            print("After top sort check pre_cnt failed")
            for i in range(self.graph_size):
                print(f"job_info {operation_info[i]} degree {count[i]} neighbor info {operation_neighbor_info[i]}")
            raise Exception("After top sort check pre_cnt failed")

    def rtopological_sort(self) -> [[Operation]]:
        queue = self.rinit_op
        queue_opinfo = []
        for op in queue:
            op.lbegin_time = self.cmax - op.parray[op.choose_array_idx]  # 虚拟终止节点出发cmax - pt就是初始入度为0的节点的最晚开始时间
            queue_opinfo.append((op.job_idx+1, op.op_idx+1))
        # print(f"rtop sort init queue info: {queue_opinfo}")
        visited = [] # 工序的访问序列
        visited_flag = [False] * self.graph_size
        while len(queue) != 0:
            nextq = []
            for op in queue:
                idx = self.chrome.job_array.GetOperationAbsoulteIdx(op.job_idx, op.op_idx)
                assert visited_flag[idx] == False
                visited_flag[idx] = True
                visited.append(op)
                for next_op in [op.job_rnext, op.machine_rnext]:
                    if next_op is not None:
                        lbegin_t = next_op.lbegin_time  # 最晚开始时间
                        next_pt = next_op.parray[next_op.choose_array_idx]  # op->next_op反向连接边的权值
                        next_op.lbegin_time = min(lbegin_t, op.lbegin_time - next_pt)  # 更新next_op的最晚开始时间
                        next_op.rpre_cnt -= 1  # 反向入度数目-1
                        if next_op.rpre_cnt == 0:
                            nextq.append(next_op)  # 加入下一层
            queue = nextq  # 访问下一层
        assert len(visited) == self.graph_size
        # visited_sequence = [(op.job_idx + 1, op.op_idx + 1) for op in visited]
        # print(f"rtop sort sequence: {visited_sequence}")  # debug info

        for op in self.graph:
            assert op.rpre_cnt == 0 # 检查拓扑排序后的反向入度
        return visited

    def GetCriticalPath(self):
        self.InitGraph()

        # 正向拓扑排序
        self.topological_sort()
        # 逆向拓扑排序
        self.rtopological_sort()
        # 求出最早开始时间和最晚开始时间相等的工序
        critical_op = []
        for op in self.graph:
            assert op.ebegin_time <= op.lbegin_time
            if op.lbegin_time == op.ebegin_time:
                critical_op.append(op)
        return critical_op

    def DeleteEdge(self, idx: int) -> [Operation, Operation, int]:
        """
            删除途中索引idx位置的机器边 并重置机器信息
        :param idx: 工序索引
        :return:
        """
        self.graph_update = True
        op = self.graph[idx]
        pre = op.machine_rnext  # op机器紧前工序
        next = op.machine_next  # op机器紧后工序
        # 断开边
        op.machine_next = None
        op.machine_rnext = None
        # 连接pre和next双向边
        if pre != None:
            pre.machine_next = next
        if next != None:
            next.machine_rnext = pre

        machine_idx = op.choose_machine_idx # 机器索引信息
        op.ResetScheduleVar() # 重置op选择的机器信息
        return pre, next, machine_idx

    def ReConnectEdge(self, idx: int, machine_idx, pre: [Operation], next: [Operation]):
        """
            重连索引idx位置的机器边
        :param idx:
        :param machine_idx:
        :param pre:
        :param next:
        :return:
        """
        self.graph_update = False
        op = self.graph[idx]
        op.InsertMachine(machine_idx)
        if pre != None:
            op.machine_rnext = pre
            pre.machine_next = op
        if next != None:
            op.machine_next = next
            next.machine_rnext = op


    def UpdateChrome(self) -> [Chromosome]:
        return self.Encode()

    def Encode(self) -> [Chromosome]:
        """
            将self.graph重新编码成染色体
        :return:
        """
        if self.graph_update is False:
            return self.chrome

        self.graph_update = False
        # 重新构建graph
        self.cmax = -1
        self.init_op.clear()
        self.rinit_op.clear()
        # 重连工件部分的边
        self.chrome.job_array.InitGraph()
        # 重连机器部分的边
        for op in self.graph:
            if op.machine_next is not None:
                op.machine_next.pre_cnt += 1  # 下一个工序入度+1
            if op.machine_rnext is not None:
                op.machine_rnext.rpre_cnt += 1  # 反向下一个工序入度+1

        self.CheckOperationDegree()
        for op in self.graph:
            if op.pre_cnt == 0:
                self.init_op.append(op)
            if op.rpre_cnt == 0:
                self.rinit_op.append(op)

        visited = self.topological_sort() # 获取工件的拓扑排序序列
        visited_job_order = [] # 工件编码
        visited_machine_order = [-1] * self.graph_size # 机器编码
        for op in visited:
            visited_job_order.append(op.job_idx)
            idx = self.chrome.job_array.GetOperationAbsoulteIdx(op.job_idx, op.op_idx)
            visited_machine_order[idx] = op.choose_machine_idx

        self.chrome.SetMachineCode(np.array(visited_machine_order))
        self.chrome.SetJobCode(np.array(visited_job_order))
        self.chrome.Check()
        return self.chrome

    def CalObjective(self):
        if self.graph_update:
            self.UpdateChrome()
        objs = self.chrome.CalObjective(self.objn)
        return objs, np.dot(np.array(self.weight), np.array(objs))

    def Insert(self, insert_op, target_op ,machine_idx):
        """
            op<-> insert_op <-> op 中间位置
            insert_op <-> op 第一个位置
            op<-> insert_op 最后一个位置
        :param insert_idx: 插入工序
        :param target_idx: 目标工序
        :param machine_idx: 插入的机器索引
        :return:
        """
        self.graph_update = True
        # 插入位置的机器紧前工序
        insert_pre = insert_op.machine_rnext
        # 修改target_op的机器紧前工序 连接反向边
        target_op.machine_rnext = insert_pre
        if insert_pre is not None:
            # 连接插入位置的机器紧前工序 连接正向边
            insert_pre.machine_next = target_op

        # 连接插入位置工序
        target_op.machine_next = insert_op
        insert_op.machine_rnext = target_op

        target_op.InsertMachine(machine_idx)
        # 检查节点入度数目
        # self.CheckOperationDegree()
        # 检查该机器上属于该工序的访问顺序 检查是否有环
        self.CheckOperationSequence(target_op)

    def ChcekOperationTime(self):
        for op in self.graph:
            assert op.ebegin_time == 0
            assert op.lbegin_time == math.inf

    def CheckOperationSequence(self, op: [Operation]):
        """
            检查op在机器上的访问顺序是否存在 op->pre_op或者 post_op -> op这样的路径
        :param op: 检查的工序
        :return:
        """
        begin_op = op
        # begin_op设置为机器上的起始工序
        times = 0
        while begin_op.machine_rnext is not None:
            begin_op = begin_op.machine_rnext
            times += 1
            assert times < 100
        # 和op同在一个工件内的工序索引
        op_idices = []
        times = 0
        while begin_op is not None:
            if begin_op.job_idx == op.job_idx:
                op_idices.append(begin_op.op_idx)
            begin_op = begin_op.machine_next
            assert times < 100
        # 检查一个工序是否被包含多次
        abs_idx_set = set(op_idices)
        assert len(abs_idx_set) == len(op_idices)
        # 检查工序顺序是否一致
        sorted_idx = sorted(op_idices)
        assert sorted_idx == op_idices

    def GetOperationECompleteTime(self, idx):
        """
            获取idx位置工序的最早完成时间
        :param idx:
        :return:
        """
        op = self.graph[idx]
        return op.GetECompleteTime()

    def GetOperationLStartTime(self, idx):
        """
            获取idx位置工序的最晚开始时间
        :param idx:
        :return:
        """
        op = self.graph[idx]
        return op.GetLStartTime()

    def GetOperation(self, idx) -> [Operation]:
        return self.graph[idx]


class LocalSearch:
    def __init__(self, iter_max, select_size, p=0.1, objn=3):
        self.iter_max = iter_max
        self.select_size = select_size
        self.p = p
        self.objn = objn
        self.choose_idx = []

    def TournamentSelection(self, weight, population: [Population], zmin: np.ndarray, zmax: np.ndarray) -> [int, Chromosome]:
        assert len(self.choose_idx) >= self.select_size
        # 在未选的个体中随机挑选
        tour_select = np.random.permutation(np.array(self.choose_idx))[:self.select_size].tolist()
        # 计算个体目标函数
        select_popfun = population.CalPartChildfunc(self.objn, tour_select)
        # 归一化目标函数
        norm_popfun = Normalize(select_popfun, zmin, zmax)
        # 计算加权目标函数值
        weight_popfunc = np.dot(norm_popfun, weight).reshape(-1)
        # 选择加权目标函数值最小的个体
        idx = tour_select[np.argmin(weight_popfunc)]
        # 无回放的选择
        self.choose_idx.remove(idx)
        chrome = population.GetChildChrome(idx)
        return idx, chrome

    def LocalSearch(self, weight_set, population: [Population], zmin: np.ndarray, zmax: np.ndarray) -> [Population]:
        """
            对种群中的个体进行局部搜索
            1. 对子代进行局部搜索并替换
            2. 对子代进行局部搜索产生新种群
            3. 采用局部搜索过程中产生的非支配解集合作为新种群
        :param weight_set: 权重向量集合
        :param population: 种群
        :return:
            返回局部搜索后新的种群
        """
        self.choose_idx = [i for i in range(population.Size())]
        size = round(population.Size() * self.p)
        weight = np.array(weight_set)
        wsize = weight.shape[0]
        for i in range(size):
            random_weight = weight[np.random.randint(0, wsize)]
            idx, chrome = self.TournamentSelection(random_weight, population, zmin, zmax)
            chrome = copy.deepcopy(chrome)
            new_chrome, find = self.LocalSearchForIndividual(random_weight, chrome, zmin, zmax)
            if find:
                population.UpdateOneOffSprint(idx, chrome)
        return population

    # 对个体进行局部搜索
    def LocalSearchForIndividual(self, weight_vector, chrome: Chromosome, zmin: np.ndarray, zmax: np.ndarray) -> [Chromosome, int]:
        """
            对chrome进行局部搜索 返回新的chrome个体和是否找到flag
        :param weight_vector: 权重向量 计算目标函数值使用
        :param chrome: 局部搜索的个体
        :return:
            chrome: 新个体
            find: 是否找到新个体
        """
        # 初始化图
        graph = Graph(self.objn, weight_vector, chrome)
        best_jc, best_mc = chrome.GetJobCode(), chrome.GetMachineCode()

        best_obj = np.array(chrome.CalObjective(self.objn)).reshape(1, -1)
        # 归一化目标函数
        best_obj = Normalize(best_obj, zmin, zmax)
        # 计算加权best value
        best_value = np.dot(np.array(weight_vector), np.array(best_obj))
        assert best_value >= 0
        # print(f"==========local serarch for individual init best obj {best_obj} value {best_value}============")
        iter = 0
        flag = 0
        while iter < self.iter_max:
            graph, chrome, find = self.KInsert(graph, chrome) # 领域搜索
            if find:
                # 计算新的目标函数值
                new_obj = np.array(chrome.CalObjective(self.objn)).reshape(1, -1)
                # 更新极值点
                zmin, zmax = GetMaxAndMinObj(new_obj, zmin, zmax)
                # 计算归一化目标函数值
                norm_obj = Normalize(new_obj, zmin, zmax)
                # 计算加权值
                new_value = np.dot(np.array(weight_vector), np.array(norm_obj))
                assert new_value >= 0
                # print(f"[iter={iter}] local search for individual find new graph {new_obj} value {new_value}")
                if new_value < best_value:
                    # 更新工件编码和机器编码
                    best_jc, best_mc = chrome.GetJobCode(), chrome.GetMachineCode()
                    best_value = new_value
                    flag = 1
            else:
                # 邻域搜索失败直接返回
                # print(f"[iter={iter}] local search for individual failed")
                break
            iter += 1
        # 设置新的jc和mc编码
        chrome.SetJobCode(best_jc)
        chrome.SetMachineCode(best_mc)
        return chrome, flag

    # 对graph删除和插入新边之后 返回的新的graph编码和图不对应
    # KInsert在local Serach Individual中被重复调用多次
    def KInsert(self, graph: Graph, chrome: Chromosome) -> [Graph, Chromosome, bool]:
        critical_op = graph.GetCriticalPath() # 获取关键路径
        assert len(critical_op) != 0

        critical_opinfo = [(op.job_idx+1, op.op_idx+1) for op in critical_op]
        # print(f"critical operation path: {critical_opinfo}") # debug info
        graph.chrome.decode()
        sequences = graph.chrome.machine_array.GetJobSequenceIdx()
        for i, op in enumerate(critical_op):
            # remove_graph = copy.deepcopy(graph) # 复制一个临时图
            op_absIdx = chrome.job_array.GetOperationAbsoulteIdx(op.job_idx, op.op_idx) # 获取该工序的索引位置
            # remove_graph.DeleteEdge(op_absIdx)
            m_pre, m_next, m_idx = graph.DeleteEdge(op_absIdx) # 删除边 获取机器上的紧前和紧后工序及机器id
            # TODO 对邻域机器按目标函数进行排序
            for machine_idx in op.marray:
                # 在可用机器上进行邻域搜索
                if self.InsertOperationOnMachine(graph, op_absIdx, machine_idx, sequences):
                    # 插入新位置后贪婪插入式解码可能会将工序重新插入到之前的位置
                    # 更新chrome
                    new_chrome = graph.UpdateChrome()
                    return graph, new_chrome, True
                    # chrome = remove_graph.Encode()
                    # chrome.decode()
                    # chrome.machine_array.Gatta(10) # debug
            graph.ReConnectEdge(op_absIdx, m_idx, m_pre, m_next) # 插入失败重新恢复删除的机器边

        return graph, chrome, False

    def InsertOperationOnMachine(self, graph, target_idx, machine_idx, sequence_idx):
        """
        :param graph: 邻域搜索的图
        :param target_idx: 将target_idx位置的工序插入机器machine_idx上
        :param machine_idx: 机器索引
        :param sequence_idx: 所有机器上的加工工序索引信息
        :return: graph, find
        """
        target_op = graph.GetOperation(target_idx)
        # 得到集合Q_k中的工序索引
        sequences = []
        for job_idx, op_idx in sequence_idx[machine_idx]:
            idx = graph.chrome.job_array.GetOperationAbsoulteIdx(job_idx, op_idx)
            if idx == target_idx: # 如果插入机器是当前已选择机器 则过滤掉该工序
                continue
            sequences.append(graph.GetOperation(idx))

        # 对机器k上的工序进行排序得到集合Q_k
        sorted(sequences, key=lambda op: op.ebegin_time)
        R_k = []  # 集合R_k
        L_k = []  # 集合L_k
        for pos, op in enumerate(sequences):
            if op.ebegin_time + op.parray[op.choose_array_idx] > target_op.ebegin_time:
                R_k.append(pos)
            if op.lbegin_time < target_op.lbegin_time:
                L_k.append(pos)

        # print("集合R_k, ", R_k, " 集合L_k, ", L_k)
        # 集合 L_k \ R_k
        left_set = set(L_k) - set(R_k)
        # 集合 R_k \ L_k
        right_set = set(R_k) - set(L_k)
        # print("集合left_set, ", left_set, " 集合right_set, ", right_set)
        # 找到插入位置
        begin = -1
        if len(left_set) != 0:
            begin = max(left_set)
        end = len(sequences)
        if len(right_set) != 0:
            end = min(right_set)

        for pos in range(begin + 1, end):
            insert_pos_op = sequences[pos] # 插入位置的工序
            op_machine_pre = insert_pos_op.machine_rnext # 插入位置机器紧前工序

            # max(插入位置的前一工序最早完成时间, 工件紧前工序的最早完成时间) + pij
            #                       <= min(pos位置工序的最晚开始时间, insert_op工件紧后工序最晚开始时间)
            m_preop_ecomplete = 0
            if op_machine_pre is not None:
                m_preop_ecomplete = op_machine_pre.GetECompleteTime() # 插入位置的前一工序最早完成时间
            j_preop_ecomplete = 0
            if target_op.job_rnext is not None:
                j_preop_ecomplete = target_op.job_rnext.GetECompleteTime() # 工件紧前工序的最早完成时间
            ec_time = max(m_preop_ecomplete, j_preop_ecomplete) + target_op.GetProcessTime(machine_idx) # target_op在插入位置的最早完成时间

            insert_postop_lbegint = insert_pos_op.GetLStartTime()
            target_postop_lbegint = math.inf
            if target_op.job_next is not None:
                target_postop_lbegint = target_op.job_next.GetLStartTime()
            lb_time = min(insert_postop_lbegint, target_postop_lbegint) # target_op在插入位置的允许的最晚开始时间
            if ec_time <= lb_time:
                # debug info
                # interval_info = [(idx[0]+1, idx[1]+1) for idx in sequence_idx[machine_idx]]
                # print(f"插入机器 {machine_idx + 1}, 插入工件索引{target_op.job_idx + 1}-{target_op.op_idx + 1}, 插入区间 [{begin + 1}, {end-1}] 原加工顺序为 {interval_info} 插入区间工序为 f{interval_info}")
                graph.Insert(insert_pos_op, target_op ,machine_idx)
                # 打印该机器上起始工序到截至工序信息
                # debug info
                # op = target_op
                # while op.machine_rnext is not None:
                #     op = op.machine_rnext
                # seq = []
                # while op is not None:
                #     seq.append((op.job_idx+1, op.op_idx+1))
                #     op = op.machine_next
                # print(f"插入之后该机上的访问顺序依次为 seq = {seq}")
                # debug info
                return True
        return False
