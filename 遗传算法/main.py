import numpy as np
import matplotlib.pyplot as plt
import random

# 城市个数
gene_len = 8
# 种群中个体数
individual_num = 60
# 迭代轮数
gen_num = 100
# 变异概率
mutate_prob = 0.25

# 模拟深拷贝
def copy_list(old_arr: list[int]):
    """
    深拷贝列表的函数。

    Parameters:
        old_arr (list[int]): 要复制的列表。

    Returns:
        list[int]: 复制的新列表。
    """
    new_arr = []
    for element in old_arr:
        new_arr.append(element)
    return new_arr


# 个体类（城市路线）
class Individual:
    """
    个体类，表示城市路线。

    Attributes:
        genes (list[int]): 表示城市路线的基因序列。
        fitness (float): 个体的适应度评分。
    """
    def __init__(self, genes=None):
        """
        初始化个体类。

        Parameters:
            genes (list[int], optional): 基因序列，如果未提供则随机生成一个序列。

        Returns:
            None
        """
        if genes is None:
            genes = [i for i in range(gene_len)]
            random.shuffle(genes)
        self.genes = genes
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        """
        计算个体的适应度评分。

        Returns:
            float: 个体的适应度评分，值越小越好。
        """
        fitness = 0.0
        for i in range(gene_len-1):
            from_idx = self.genes[i]
            to_idx = self.genes[i+1]
            fitness += city_dist_mat[from_idx, to_idx]
        fitness += city_dist_mat[self.genes[-1], self.genes[0]]  # 回到起点的距离
        return fitness


class Ga:
    """
    遗传算法类，用于解决旅行商问题。

    Attributes:
        best (Individual): 当前迭代中的最佳个体（最佳路径）。
        individual_list (list[Individual]): 当前种群中的个体列表。
        result_list (list[list[int]]): 每一代的最佳路径列表。
        fitness_list (list[float]): 每一代的最佳适应度列表。
    """
    def __init__(self, input_):
        """
        初始化遗传算法类。

        Parameters:
            input_ (numpy.ndarray): 城市之间的距离矩阵。

        Returns:
            None
        """
        global city_dist_mat
        city_dist_mat = input_
        self.best = None
        self.individual_list = []
        self.result_list = []
        self.fitness_list = []

    @staticmethod
    def fill(son, gene, index1, index2):
        """
        辅助函数，用于填充子代基因。

        Parameters:
            son (list[int]): 子代基因序列。
            gene (list[int]): 另一个父代的基因序列。
            index1 (int): 交叉的起始索引。
            index2 (int): 交叉的结束索引。

        Returns:
            None
        """
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

    def cross(self):
        """
        交叉操作，使用顺序交叉法（order crossover）。

        Returns:
            list[Individual]: 交叉后得到的新个体列表。
        """
        new_gen = []
        random.shuffle(self.individual_list)  # 首先打乱当前种群
        for i in range(0, individual_num-1, 2):
            genes1 = self.individual_list[i].genes
            genes2 = self.individual_list[i+1].genes
            index1 = random.randint(0, gene_len//2)
            index2 = random.randint(index1+1, gene_len-1)
            son1 = copy_list(genes1)
            son2 = copy_list(genes2)
            self.fill(son1, genes2, index1, index2)
            self.fill(son2, genes1, index1, index2)
            new_gen.append(Individual(son1))
            new_gen.append(Individual(son2))
        return new_gen

    def mutate(self, new_gen):
        """
        变异操作。

        Parameters:
            new_gen (list[Individual]): 新一代的个体列表。

        Returns:
            None
        """
        for individual in new_gen:
            if random.random() < mutate_prob:  # 根据变异概率进行变异
                old_genes = copy_list(individual.genes)
                index1 = random.randint(0, gene_len - 2)
                index2 = random.randint(index1, gene_len - 1)
                genes_mutate = old_genes[index1:index2]
                genes_mutate.reverse()  # 翻转切片
                individual.genes = old_genes[:index1] + genes_mutate + old_genes[index2:]
        self.individual_list += new_gen  # 将新一代个体加入到种群中

    def select(self):
        """
        选择操作，使用锦标赛算法。

        Returns:
            None
        """
        group_num = 10
        group_size = 10
        group_winner = individual_num // group_num
        winners = []
        for i in range(group_num):
            group = []
            for j in range(group_size):
                player = random.choice(self.individual_list)  # 随机选择个体组成小组
                player = Individual(player.genes)
                group.append(player)
            group = Ga.rank(group)
            winners += group[:group_winner]  # 取出每个小组的获胜者
        self.individual_list = winners  # 更新种群

    @staticmethod
    def rank(group):
        """
        对个体进行排名（根据适应度评分排序）。

        Parameters:
            group (list[Individual]): 要排名的个体列表。

        Returns:
            list[Individual]: 排序后的个体列表。
        """
        for i in range(1, len(group)):
            for j in range(0, len(group) - i):
                if group[j].fitness > group[j + 1].fitness:
                    group[j], group[j + 1] = group[j + 1], group[j]
        return group

    def next_gen(self):
        """
        生成下一代种群。

        Returns:
            None
        """
        new_gen = self.cross()  # 交叉
        self.mutate(new_gen)  # 变异
        self.select()  # 选择
        for individual in self.individual_list:
            if individual.fitness < self.best.fitness:  # 更新最佳个体
                self.best = individual
        print(f"当前最佳个体：{self.best.genes}, 适应度：{self.best.fitness}")

    def train(self):
        """
        训练函数，运行遗传算法解决旅行商问题。

        Returns:
            tuple: 包含最佳路径列表和每一代最佳适应度列表的元组。
        """
        self.individual_list = [Individual() for _ in range(individual_num)]  # 初代种群
        self.best = self.individual_list[0]
        for i in range(gen_num):  # 迭代
            print(f"第{i+1}代")
            self.next_gen()
            result = copy_list(self.best.genes)
            result.append(result[0])  # 将最佳路径连接首尾形成闭环
            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)
        return self.result_list, self.fitness_list

city_dist_mat = np.array([
    [0, 49, 25, 19, 63, 74, 26, 39],
    [49, 0, 26, 48, 65, 36, 42, 55],
    [25, 26, 0, 26, 21, 24, 78, 49],
    [19, 48, 26, 0, 45, 44, 57, 62],
    [63, 65, 21, 45, 0, 47, 48, 54],
    [74, 36, 24, 44, 47, 0, 47, 65],
    [26, 42, 78, 57, 48, 47, 0, 47],
    [39, 55, 49, 62, 54, 65, 47, 0]
])

if __name__ == "__main__":
    # 运行遗传算法
    ga = Ga(city_dist_mat)
    # 获取每代的最佳路线和最佳适应度
    result_list, fitness_list = ga.train()
    # 输出最终结果
    result = result_list[-1]
    res_dis = fitness_list[-1]
    print("最佳路线：{}".format(result))
    print("最短距离：{}".format(res_dis))

    # 绘制适应度曲线
    plt.rcParams['font.family'] = 'SimHei' 
    plt.figure()
    plt.plot(fitness_list)
    plt.title(u'适应度曲线')
    plt.show()
