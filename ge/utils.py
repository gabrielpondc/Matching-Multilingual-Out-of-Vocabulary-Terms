import itertools
import math
import random
import numpy as np

import pandas as pd
from joblib import Parallel, delayed

from .alias import alias_sample, create_alias_table
from .utils import partition_num


class RandomWalker:
    def __init__(self, G, p=1, q=1, use_rejection_sampling=False):
        """
        :param G:
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        :param use_rejection_sampling: Whether to use the rejection sampling strategy in node2vec.
        """
        self.G = G
        self.p = p
        self.q = q
        self.use_rejection_sampling = use_rejection_sampling

    def deepwalk_walk(self, walk_length, start_node):

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def node2vec_walk(self, walk_length, start_node):

        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0],
                                                      alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break

        return walk

    def node2vec_walk2(self, walk_length, start_node):
        """
        Reference:
        KnightKing: A Fast Distributed Graph Random Walk Engine
        http://madsys.cs.tsinghua.edu.cn/publications/SOSP19-yang.pdf
        """

        def rejection_sample(inv_p, inv_q, nbrs_num):
            upper_bound = max(1.0, max(inv_p, inv_q))
            lower_bound = min(1.0, min(inv_p, inv_q))
            shatter = 0
            second_upper_bound = max(1.0, inv_q)
            if (inv_p > second_upper_bound):
                shatter = second_upper_bound / nbrs_num
                upper_bound = second_upper_bound + shatter
            return upper_bound, lower_bound, shatter

        G = self.G
        alias_nodes = self.alias_nodes
        inv_p = 1.0 / self.p
        inv_q = 1.0 / self.q
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    upper_bound, lower_bound, shatter = rejection_sample(
                        inv_p, inv_q, len(cur_nbrs))
                    prev = walk[-2]
                    prev_nbrs = set(G.neighbors(prev))
                    while True:
                        prob = random.random() * upper_bound
                        if (prob + shatter >= upper_bound):
                            next_node = prev
                            break
                        next_node = cur_nbrs[alias_sample(
                            alias_nodes[cur][0], alias_nodes[cur][1])]
                        if (prob < lower_bound):
                            break
                        if (prob < inv_p and next_node == prev):
                            break
                        _prob = 1.0 if next_node in prev_nbrs else inv_q
                        if (prob < _prob):
                            break
                    walk.append(next_node)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):

        G = self.G

        nodes = list(G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, ):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                if self.p == 1 and self.q == 1:
                    walks.append(self.deepwalk_walk(
                        walk_length=walk_length, start_node=v))
                elif self.use_rejection_sampling:
                    walks.append(self.node2vec_walk2(
                        walk_length=walk_length, start_node=v))
                else:
                    walks.append(self.node2vec_walk(
                        walk_length=walk_length, start_node=v))
        return walks

    def get_alias_edge(self, t, v):
        """
        compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
        :param t:
        :param v:
        :return:
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)  # w_vx
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight / p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr].get('weight', 1.0)
                                  for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)

        if not self.use_rejection_sampling:
            alias_edges = {}

            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                if not G.is_directed():
                    alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
                self.alias_edges = alias_edges

        self.alias_nodes = alias_nodes
        return


class BiasedWalker:
    def __init__(self, idx2node, temp_path):

        self.idx2node = idx2node
        self.idx = list(range(len(self.idx2node)))
        self.temp_path = temp_path
        pass

    def simulate_walks(self, num_walks, walk_length, stay_prob=0.3, workers=1, verbose=0):

        layers_adj = pd.read_pickle(self.temp_path + 'layers_adj.pkl')
        layers_alias = pd.read_pickle(self.temp_path + 'layers_alias.pkl')
        layers_accept = pd.read_pickle(self.temp_path + 'layers_accept.pkl')
        gamma = pd.read_pickle(self.temp_path + 'gamma.pkl')

        nodes = self.idx  # list(self.g.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length, stay_prob, layers_adj, layers_accept, layers_alias,
                                          gamma) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))
        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, stay_prob, layers_adj, layers_accept, layers_alias, gamma):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self._exec_random_walk(layers_adj, layers_accept, layers_alias,
                                                    v, walk_length, gamma, stay_prob))
        return walks

    def _exec_random_walk(self, graphs, layers_accept, layers_alias, v, walk_length, gamma, stay_prob=0.3):
        initialLayer = 0
        layer = initialLayer

        path = []
        path.append(self.idx2node[v])

        while len(path) < walk_length:
            r = random.random()
            if (r < stay_prob):  # same layer
                v = chooseNeighbor(v, graphs, layers_alias,
                                   layers_accept, layer)
                path.append(self.idx2node[v])
            else:  # different layer
                r = random.random()
                try:
                    x = math.log(gamma[layer][v] + math.e)
                    p_moveup = (x / (x + 1))
                except:
                    print(layer, v)
                    raise ValueError()

                if (r > p_moveup):
                    if (layer > initialLayer):
                        layer = layer - 1
                else:
                    if ((layer + 1) in graphs and v in graphs[layer + 1]):
                        layer = layer + 1

        return path


def chooseNeighbor(v, graphs, layers_alias, layers_accept, layer):
    v_list = graphs[layer][v]

    idx = alias_sample(layers_accept[layer][v], layers_alias[layer][v])
    v = v_list[idx]

    return 

def create_alias_table(area_ratio):
    """
    :param area_ratio: sum(area_ratio)=1，概率向量
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)
    # 此时small，large记录的是 概率是否大于或小于平均概率 1/len(area_ratio)，大于1 则大于平均概率值
    # 假设此时large的个数多余small
    while small and large:
        small_idx, large_idx = small.pop(), large.pop()  # 从后往前，拿出最后一个大于平均值的索引，最后一个小于平均值的索引
        accept[small_idx] = area_ratio_[small_idx]  # 将最后一个小于平均值的位置上设置为该小于平均值的数值
        alias[small_idx] = large_idx  # 一个小于平均值的位置上设置为进行了放缩的一个大于平均值的索引
        area_ratio_[large_idx] = area_ratio_[large_idx] - (1 - area_ratio_[small_idx])  # 在area_ratio_中的最后一个大于平均值的数值变小了，变小的幅度是最后一个小于平均值
        if area_ratio_[large_idx] < 1.0:  # 对于拿出来的大于平均值，在进行判断
            small.append(large_idx)
        else:
            large.append(large_idx)

        # 一次循环之后，accept 记录的是在一个小于平均值的位置上记录值，在alias记录的是一个小于平均值的数对于哪个大于平均值数进行了放缩，记录了这个大于平均值的索引
        # 一次操作之后，small少了一个，large少了一个，然后large、small按照对应大小可能多一个，

    # 这个while，将每个大于平均值的概率进行了缩放，可能会变大可能会变小，当while结束之后
    # accpet除了0，每个位置上就是小于平均值的数值，对应索引就是该小于平均值的数值
    # 0, 0, 0.1, 0   0.1的索引2表示第2个数小于平均值，数值是0.1
    # alias除了0，其他的都是对于该位置索引所对应的小于平均值的数，对大于平均值的数进行放缩的那个大于平均值的数的索引
    # 0, 0，3， 0       3的索引是2，表示位置是2的小于平均值的数对位置是3的大于平均值的数进行了放缩，之前的放缩（假设位置1的数之前还对位置是5的数进行了放缩）的记录被覆盖了

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1  # 将现在放缩之后仍然有的每个大于平均值的，都设置为1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1  # 将每个放缩之后仍要有的小于平均值的，设置为1
    # accept 中的小于平均值的数，都对一个大于平均值的数进行了放缩，其他的都没有对其他进行过放缩
    # alias 记录的则是每个小于平均值的数对哪个大于平均值的数进行了放缩，其余的0 则是没有进行了放缩

    # accept 1,1,0.1,1
    # alias 0，0，3，0
    return accept, alias


def alias_sample(accept, alias):
    """
    在 alias 上进行随机取样
    :param accept: 向量，除了1，就是小于1的数，小于1的数意思是在之前的概率向量中该位置的小于1的概率对一个大于平均值概率进行了放缩，既该概率值使用过
    :param alias: 向量，除了1，其他的都是位置索引，和accept对应，如果accept中不是1，那么alias中对应位置就是个索引，且alias中该位置上的索引的大于平均值的数根据accept上的小于平均值的数进行了放缩
    :return: sample index
    返回值要么根据概率输出的随机索引，要么是 alias 的上索引
    所以返回值要么1，要么是个alias上的索引，要么是个随机索引
    """
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]


def preprocess_nxgraph(graph):
    """
    该函数将图结构中实体映射成索引
    :param graph: 输入的图结构
    :return: node2idx 字典 key是实体名称，value是索引，idx2node 按照字典中的索引中到字典中的词
    """
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def partition_dict(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in vertices.items():
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_list(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in enumerate(vertices):
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_num(num, workers):
    """
    按照批次获得数据
    :param num: 一个批次内传进来的数据量
    :param workers: 并行CPU数，
    :return: 返回一个长度为workers的list，每个元素是每个worker处理的数据量
    """
    if num % workers == 0:

        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]

def partition_dict(vertices,workers):
    batch_size = (len(vertices)-1)//workers +1 
    part_list = []
    part = []
    count = 0 
    for v,nbs in vertices.items():
        part.append((v,nbs))
        count +=1 
        if count%batch_size == 0:
            part_list.append(part)
            part=[]
    if len(part)>0:
        part_list.append(part)
    return part_list