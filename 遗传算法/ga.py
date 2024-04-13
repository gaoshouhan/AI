import random

# 各种超参数
city_dist_mat = None
# 城市个数
gene_len = 8
# 种群中个体数
individual_num = 60
# 迭代轮数
gen_num = 100
# 变异概率
mutate_prob = 0.25

# 模拟深拷贝
def copy_list(old_arr: [int]):
    new_arr = []
    for element in old_arr:
        new_arr.append(element)
    return new_arr


# 个体类（城市路线）
class Individual:
    # genes为基因序列 这里指城市路线
    def __init__(self, genes=None):
        # 初始种群初始化路线
        if genes is None:
            genes = [i for i in range(gene_len)]
            random.shuffle(genes)
        self.genes = genes
        self.fitness = self.evaluate_fitness()

    # 计算个体基因适应度 适应度高（路程最短）被选择
    def evaluate_fitness(self):
        fitness = 0.0
        for i in range(gene_len-1):
            # 起始城市和目标城市
            from_idx = self.genes[i]
            to_idx = self.genes[i+1]
            fitness += city_dist_mat[from_idx, to_idx]
        # 还要计算回到原点
        fitness += city_dist_mat[self.genes[-1], self.genes[0]]
        return fitness


class Ga:
    # 传入城市距离矩阵
    def __init__(self, input_):
        global city_dist_mat
        city_dist_mat = input_
        # 当代的最佳个体（最佳路径）
        self.best = None
        # 每一代的个体列表（种群）
        self.individual_list = []
        # 每一代对应的解（最佳路径）
        self.result_list = []
        # 每一代对应的最佳适应度
        self.fitness_list = []


    @staticmethod
    def fill(son, gene, index1, index2):
        have = son[index1:index2]
        for i in range(0, index1):
            for n in gene:
                if n not in have:
                    son[i] = n
                    have.append(n)
                    break

        for i in range(index2, len(son)):
            for n in gene:
                if n not in have:
                    son[i] = n
                    have.append(n)
                    break


    # 交叉
    def cross(self):
        # 顺序交叉法（order crossover）
        new_gen = []
        # 首先打乱当前种群
        random.shuffle(self.individual_list)
        # 遍历种群个体 一次取两个
        for i in range(0, individual_num-1, 2):
            # 父代基因
            genes1 = self.individual_list[i].genes
            genes2 = self.individual_list[i+1].genes
            # 随机起始位置
            index1 = random.randint(0, gene_len//2)
            index2 = random.randint(index1+1, gene_len-1)
            # 子代基因 要深拷贝
            son1 = copy_list(genes1)
            son2 = copy_list(genes2)
            # 填充子代基因除了区间内剩余部分
            self.fill(son1, genes2, index1, index2)
            self.fill(son2, genes1, index1, index2)
            print(son1)
            new_gen.append(Individual(son1))
            new_gen.append(Individual(son2))
        return new_gen


    # 变异
    def mutate(self, new_gen):
        for individual in new_gen:
            # 变异概率
            if random.random() < mutate_prob:
                # 翻转切片
                old_genes = copy_list(individual.genes)
                index1 = random.randint(0, gene_len - 2)
                index2 = random.randint(index1, gene_len - 1)
                genes_mutate = old_genes[index1:index2]
                genes_mutate.reverse()
                individual.genes = old_genes[:index1] + genes_mutate + old_genes[index2:]
        # 两代合并
        self.individual_list += new_gen


    # 选择 轮盘赌算法 锦标赛算法
    def select(self):
        # 使用锦标赛算法选择individual_num个体
        '''
            锦标赛算法
            定义10个小组 默认60个个体 则每个小组选择6个人 10进6
            不能一次选择60个人 那样会进入到一种局部最优的状态
        '''
        group_num = 10  # 小组数
        group_size = 10  # 每小组人数
        group_winner = individual_num // group_num  # 每小组获胜人数
        winners = []  # 锦标赛结果
        for i in range(group_num):
            group = []
            for j in range(group_size):
                # 随机组成小组
                player = random.choice(self.individual_list)
                player = Individual(player.genes)
                group.append(player)
            group = Ga.rank(group)
            # 取出获胜者
            winners += group[:group_winner]
        self.individual_list = winners


    @staticmethod
    def rank(group):
        # 冒泡排序
        for i in range(1, len(group)):
            for j in range(0, len(group) - i):
                if group[j].fitness > group[j + 1].fitness:
                    group[j], group[j + 1] = group[j + 1], group[j]
        return group


    # 更新种群
    def next_gen(self):
        # 交叉
        new_gen = self.cross()
        # 变异
        self.mutate(new_gen)
        # 选择
        self.select()
        # 获得这一代的结果
        for individual in self.individual_list:
            # 适应度为距离 越小越好
            if individual.fitness < self.best.fitness:
                self.best = individual


    def train(self):
        # 初代种群（individual_num个初始城市路线）
        self.individual_list = [Individual() for _ in range(individual_num)]
        self.best = self.individual_list[0]
        # 迭代
        for i in range(gen_num):
            # 从当代种群中交叉、变异、选择出适应度高的个体（城市路线），获得新的种群
            self.next_gen()
            # 连接首尾
            result = copy_list(self.best.genes)
            result.append(result[0])

            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)
        return self.result_list, self.fitness_list