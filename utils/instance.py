from .schedule import JobArray, MachineArray, Job


def ReadInstances(file: str) -> [int, int, JobArray, MachineArray]:
    jobs: [Job] = []
    opnum_array = []  # 一维数组
    job_karrays = []  # 二维数组
    job_marrays = []  # 三维数组
    job_parrays = []  # 三维数组

    low = 1e9  # 记录测试用例的最小加工时间
    up = 0  # 最大加工时间
    # 打开文件
    with open(file, 'r') as file:
        # 逐行读取文件内容
        lines = file.readlines()
    # 读取首行3个数字
    first = lines[0].strip().split()
    job_count, machine_count = [int(first[i]) for i in range(0, 2)]
    for i in range(1, len(lines)):
        line = lines[i].strip().split(' ')
        line = [int(num) for num in line if num.isdigit()]
        operations = int(line[0])  # 当前工件的工序数目
        karray = []  # 当前工件所有工序的可用机器数目
        marrays = []  # 当前工件所有工序的可用机器集合
        parrays = []  # 当前工件所有工序的可用机器加工时间
        idx = 1
        while idx < len(line):
            k = int(line[idx])  # 工序可用机器数目
            karray.append(k)
            idx += 1
            marray = []
            parray = []
            for j in range(0, k):
                machine_idx = int(line[idx]) - 1 # 机器编号 idx = idx - 1索引下标从0开始
                ptime = int(line[idx+1])  # 加工时间
                low = min(low, ptime)
                up = max(up, ptime)
                marray.append(machine_idx)
                parray.append(ptime)
                idx += 2
            marrays.append(marray)
            parrays.append(parray)

        opnum_array.append(operations)
        job_karrays.append(karray)
        job_marrays.append(marrays)
        job_parrays.append(parrays)
    for i in range(job_count):
        jobs.append(Job(i, opnum_array[i], low, up, job_karrays[i], job_marrays[i], job_parrays[i]))

    job_array = JobArray(jobs, opnum_array, job_karrays, job_marrays, job_parrays)
    machine_array = MachineArray(machine_count)

    return job_count, machine_count, job_array, machine_array
