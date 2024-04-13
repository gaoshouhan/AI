import numpy as np
import matplotlib.pyplot as plt
import random

# 城市数量
num_vertices = 8
# 种群大小
pop_size = 60
# 迭代次数
num_gens = 100
# 变异概率
mut_prob = 0.25

# 深度复制列表的函数
def clone_list(old_list: list[int]):
    """
    函数用于深度复制列表。

    参数:
        old_list (list[int]): 要复制的列表。

    返回:
        list[int]: 复制后的新列表。
    """
    new_list = []
    for element in old_list:
        new_list.append(element)
    return new_list


# 表示个体（城市路线）的类
class Traveler:
    """
    表示个体的类，代表一条城市路线。

    属性:
        path (list[int]): 表示城市路线的基因序列。
        score (float): 个体的适应度评分。
    """
    def __init__(self, path=None):
        """
        初始化Traveler类。

        参数:
            path (list[int], optional): 基因序列，如果不提供，则生成随机序列。

        返回:
            None
        """
        if path is None:
            path = [i for i in range(num_vertices)]
            random.shuffle(path)
        self.path = path
        self.score = self.evaluate_score()

    def evaluate_score(self):
        """
        计算个体的适应度评分。

        返回:
            float: 个体的适应度评分，值越小越好。
        """
        score = 0.0
        for i in range(num_vertices-1):
            from_idx = self.path[i]
            to_idx = self.path[i+1]
            score += city_distance_matrix[from_idx, to_idx]
        score += city_distance_matrix[self.path[-1], self.path[0]]  # 返回到起点的距离
        return score


class GeneticOptimizer:
    """
    用于解决旅行商问题的遗传算法类。

    属性:
        champ (Traveler): 当前迭代中的最佳个体（最佳路线）。
        tribe (list[Traveler]): 当前种群中的个体列表。
        journey_log (list[list[int]]): 每一代最佳路线的列表。
        fitness_log (list[float]): 每一代最佳适应度评分的列表。
    """
    def __init__(self, distance_matrix):
        """
        初始化GeneticOptimizer类。

        参数:
            distance_matrix (numpy.ndarray): 城市之间的距离矩阵。

        返回:
            None
        """
        global city_distance_matrix
        city_distance_matrix = distance_matrix
        self.champ = None
        self.tribe = []
        self.journey_log = []
        self.fitness_log = []

    @staticmethod
    def fill_route(son, gene, index1, index2):
        """
        填充后代基因的辅助函数。

        参数:
            son (list[int]): 后代的基因序列。
            gene (list[int]): 另一个父代的基因序列。
            index1 (int): 交叉起始索引。
            index2 (int): 交叉结束索引。

        返回:
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

    def mate(self):
        """
        使用顺序交叉进行交叉操作。

        返回:
            list[Traveler]: 交叉后生成的后代列表。
        """
        next_gen = []
        random.shuffle(self.tribe)  # 首先对当前种群进行洗牌
        for i in range(0, pop_size-1, 2):
            genes1 = self.tribe[i].path
            genes2 = self.tribe[i+1].path
            index1 = random.randint(0, num_vertices//2)
            index2 = random.randint(index1+1, num_vertices-1)
            son1 = clone_list(genes1)
            son2 = clone_list(genes2)
            self.fill_route(son1, genes2, index1, index2)
            self.fill_route(son2, genes1, index1, index2)
            next_gen.append(Traveler(son1))
            next_gen.append(Traveler(son2))
        return next_gen

    def modify(self, next_gen):
        """
        变异操作。

        参数:
            next_gen (list[Traveler]): 新一代的后代列表。

        返回:
            None
        """
        for individual in next_gen:
            if random.random() < mut_prob:  # 根据变异概率进行变异
                old_genes = clone_list(individual.path)
                index1 = random.randint(0, num_vertices - 2)
                index2 = random.randint(index1, num_vertices - 1)
                genes_mutate = old_genes[index1:index2]
                genes_mutate.reverse()  # 反转切片
                individual.path = old_genes[:index1] + genes_mutate + old_genes[index2:]
        self.tribe += next_gen  # 将新一代的个体添加到种群中

    def select(self):
        """
        使用锦标赛选择进行选择操作。

        返回:
            None
        """
        group_num = 10
        group_size = 10
        group_winner = pop_size // group_num
        winners = []
        for i in range(group_num):
            group = []
            for j in range(group_size):
                player = random.choice(self.tribe)  # 随机选择个体组成一个小组
                player = Traveler(player.path)
                group.append(player)
            group = GeneticOptimizer.rank(group)
            winners += group[:group_winner]  # 取每个小组的胜利者
        self.tribe = winners  # 更新种群

    @staticmethod
    def rank(group):
        """
        对个体进行排名（按适应度评分排序）。

        参数:
            group (list[Traveler]): 要排名的个体列表。

        返回:
            list[Traveler]: 排序后的个体列表。
        """
        for i in range(1, len(group)):
            for j in range(0, len(group) - i):
                if group[j].score > group[j + 1].score:
                    group[j], group[j + 1] = group[j + 1], group[j]
        return group

    def next_generation(self):
        """
        生成下一代种群。

        返回:
            None
        """
        offspring = self.mate()  # 交叉
        self.modify(offspring)  # 变异
        self.select()  # 选择
        for individual in self.tribe:
            if individual.score < self.champ.score:  # 更新最佳个体
                self.champ = individual
        print(f"当前最佳个体: {self.champ.path}, 适应度: {self.champ.score}")

    def optimize(self):
        """
        训练函数，运行遗传算法解决旅行商问题。

        返回:
            tuple: 包含每一代最佳路线和每一代最佳适应度评分的列表的元组。
        """
        self.tribe = [Traveler() for _ in range(pop_size)]  # 初始种群
        self.champ = self.tribe[0]
        for i in range(num_gens):  # 迭代
            print(f"第 {i+1} 代")
            self.next_generation()
            result = clone_list(self.champ.path)
            result.append(result[0])  # 连接最佳路径形成闭环
            self.journey_log.append(result)
            self.fitness_log.append(self.champ.score)
        return self.journey_log, self.fitness_log

city_distance_matrix = np.array([
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
    ga = GeneticOptimizer(city_distance_matrix)
    # 获取每一代的最佳路线和适应度评分
    journey_log, fitness_log = ga.optimize()
    # 输出最终结果
    result = journey_log[-1]
    shortest_distance = fitness_log[-1]
    print("最佳路线: {}".format(result))
    print("最短距离: {}".format(shortest_distance))

    # 绘制适应度曲线
    plt.rcParams['font.family'] = 'SimHei' 
    plt.figure()
    plt.plot(fitness_log)
    plt.title(u'适应度曲线')
    plt.show()
