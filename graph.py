"""
n个顶点的无向图最多含n(n-1)/2条边,有向图最多含n(n-1)条边
G(V,E)满足2*E=所有顶点的度之和
图可以用邻接矩阵或邻接表表示
邻接矩阵是一个对角线元素为0的n阶方阵,无向图邻接矩阵是对称矩阵
图结构循环引用,应该对其父级和兄弟级引用使用弱引用
"""

# to be continued

from collections import deque

import matplotlib.pyplot as plt
import networkx

UDG = 1  # 无向图
DG = 0  # 有向图


class Node:
    def __init__(self, data, edge=None):
        self.data = data
        self.edge = edge

    def __eq__(self, other):
        return self.data == other.data


class Edge:
    def __init__(self, vertex, weight, right=None):
        self.vertex = vertex
        self.weight = weight
        self.right = right


class Graph:  # 邻接表存储
    def __init__(self, kind=UDG):
        self.__kind = kind
        self.__vertices = {}
        self.__edge_num = 0
        self.G = networkx.Graph()

    def show(self):
        networkx.draw(self.G, with_labels=True)
        plt.show()

    def add(self, come, to, weight=0):
        self.G.add_edge(come, to, weight=weight)
        self.__edge_num += 1
        if come in self.__vertices:
            self.__vertices[come] = Edge(to, weight, self.__vertices[come])
        else:
            self.__vertices[come] = Edge(to, weight)

        if self.__kind == UDG:
            if to in self.__vertices:
                self.__vertices[to] = Edge(come, weight, self.__vertices[to])
            else:
                self.__vertices[to] = Edge(come, weight)

    def delete(self, come, to):
        self.G.remove_edge(come, to)
        if (come in self.__vertices) and (to in self.__vertices):
            edge = self.__vertices[come]
            if edge:
                self.__edge_num -= 1
                if edge.vertex == to:
                    self.__vertices[come] = edge.right
                else:
                    pre = edge
                    edge = edge.right
                    while edge:
                        if edge.vertex == to:
                            pre.right = edge.right
                            break
                        pre = edge
                        edge = edge.right

            if self.__kind == 'UDG':  # undirected graph
                edge = self.__vertices[to]
                if edge:
                    if edge.vertex == come:
                        self.__vertices[to] = edge.right
                    else:
                        pre = edge
                        edge = edge.right
                        while edge:
                            if edge.vertex == come:
                                pre.right = edge.right
                                break
                            pre = edge
                            edge = edge.right

    def DFS(self):  # 类似于树的先根遍历,有向图的时间复杂度是O(n+e),无向图是O(n+2e),其中e代表邻接点个数
        vertices = set()

        def _DFS(vertex):
            vertices.add(vertex)
            print(vertex, end=' ')
            edge = self.__vertices[vertex]
            while edge:
                vertex = edge.vertex
                if vertex not in vertices:
                    _DFS(vertex)
                edge = edge.right

        for vertex in self.__vertices:  # For non connected graphs
            if vertex not in vertices:
                _DFS(vertex)

    def DFS_stack(self):
        is_head = True
        pos = []
        vertices = set()
        for vertex in self.__vertices:
            if vertex not in vertices:  # 仅仅是为了过滤上一层for循环中的重复元素
                print(vertex)
                vertices.add(vertex)
                while True:
                    if is_head:
                        edge = self.__vertices[vertex]
                    elif pos:
                        edge = pos.pop()
                        is_head = True
                    else:
                        break
                    while edge:
                        vertex = edge.vertex
                        if vertex not in vertices:
                            print(vertex)
                            vertices.add(vertex)
                            pos.append(edge.right)
                            break
                        edge = edge.right
                    else:
                        is_head = False
        # 低效版           
        # stack=[]
        # tmp=[]
        # vertices={vertex for vertex in self.__vertices} # 保存未访问节点
        # for vertex in self.__vertices:
        #     stack.append(vertex)
        #     while stack:
        #         vertex=stack.pop()
        #         if vertex in vertices:  # 不能少,但执行find_path或单独执行_DFS则不需要加这个判断
        #             vertices.remove(vertex)
        #             print(vertex,end=' ')    
        #         edge=self.__vertices[vertex]
        #         while edge:
        #             vertex=edge.vertex
        #             if vertex in vertices:  # 不能少
        #                 tmp.append(vertex)
        #             edge=edge.right
        #         while tmp:
        #             stack.append(tmp.pop())     

    def BFS(self):
        vertices = set()
        queue = deque()
        for vertex in self.__vertices:
            if vertex not in vertices:
                vertices.add(vertex)
                print(vertex, end=' ')
                queue.append(self.__vertices[vertex])
            while queue:
                edge = queue.popleft()
                while edge:
                    vertex = edge.vertex
                    if vertex not in vertices:
                        vertices.add(vertex)
                        print(vertex, end=' ')
                        queue.append(self.__vertices[vertex])
                    edge = edge.right

    def find_path_stack(self, start, end):
        if start == end:
            yield [start]
        else:
            flag = True
            pos = []
            vertices = set()
            path = []
            path.append(start)
            vertices.add(start)
            while True:
                if flag:
                    edge = self.__vertices[start]
                elif pos:
                    edge = pos.pop()
                    flag = True
                else:
                    break
                while edge:
                    start = edge.vertex
                    if start not in vertices:
                        path.append(start)
                        vertices.add(start)
                        if start == end:
                            yield path[:]
                            vertices.remove(path.pop())
                        else:
                            pos.append(edge.right)
                            break
                    edge = edge.right
                else:
                    vertices.remove(path.pop())
                    flag = False

    def find_path(self, start, end):  # 回溯法,常与DFS配合使用
        """
        已知有向图graph如下,求给定两点间的所有路径(1代表可达)
        graph=[  # 有向图
            [0,1,1,0,0,1],
            [1,0,0,0,0,0],
            [0,1,0,1,0,0],
            [0,1,0,0,0,1],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
        ]
        """
        vertices = set()
        path = []
        result = []

        def _find_path(start, end):
            vertices.add(start)
            path.append(start)
            if start == end:
                print(path)
                result.append(path[:])  # 注意这里是深拷贝
            else:
                edge = self.__vertices[start]
                while edge:
                    vertex = edge.vertex
                    if vertex not in vertices:
                        _find_path(vertex, end)
                    edge = edge.right
            vertices.remove(start)
            path.pop()

        _find_path(start, end)
        return result

    def topSortByDFS(self):  # 拓扑排序,相当于后续遍历,只适用于有向无环图(不能判断是否有环)
        flag = [False] * self.length
        result = []

        def _traverse(i):
            flag[i] = True
            edge = self.vertices[i].edge
            while edge:
                vertex = edge.vertex
                if not flag[vertex]:
                    _traverse(vertex)
                edge = edge.right
            result.append(self.vertices[i].data)

        for i in range(self.length):
            if not flag[i]:
                _traverse(i)
        return result[::-1]

    # 时间复杂度是O(V^2),适用于稠密图
    def Prim(self, pos):  # 待斐波那契堆改进！！！
        MST = [Node(each.data) for each in self.vertices]
        Dist = [{'dist': float('inf'), 'parent': -1, 'pos': i} for i in range(self.length)]
        collected = [False] * self.length

        collected[pos] = True
        edge = self.vertices[pos].edge
        while edge:
            Dist[edge.vertex]['dist'] = edge.weight
            Dist[edge.vertex]['parent'] = pos
            edge = edge.right
        for i in range(1, self.length):
            vertex = {'dist': float('inf')}
            for each in Dist:
                if not collected[each['pos']] and each['dist'] < vertex['dist']:
                    vertex = each

            edge = Edge(vertex['pos'], vertex['dist'], MST[vertex['parent']].edge)
            MST[vertex['parent']].edge = edge
            collected[vertex['pos']] = True
            edge = self.vertices[vertex['pos']].edge
            while edge:
                if not collected[edge.vertex] and Dist[edge.vertex]['dist'] > edge.weight:
                    Dist[edge.vertex]['dist'] = edge.weight
                    Dist[edge.vertex]['parent'] = vertex['pos']
                edge = edge.right
        return MST

    # 时间复杂度是O(ElgV)或者O(ElgE),适用于稀疏图
    def Kruskal(self):
        from heapq import heappop, heappush
        class Distance:
            def __init__(self, dist, starts, ends):
                self.dist = dist
                self.starts = starts
                self.ends = ends

            def __lt__(self, other):
                return self.dist < other.dist

        MST = [Node(each.data) for each in self.vertices]
        vertices = UnionFindSet([-1] * self.length)  # 需要导入自定义的并查集类

        h = []
        for pos, each in enumerate(self.vertices):
            edge = each.edge
            while edge:
                heappush(h, Distance(edge.weight, pos, edge.vertex))
                edge = edge.right

        edgeNum = self.length
        while edgeNum > 1:
            if not h:
                print('图不连通,不存在最小生成树')
                break
            v = heappop(h)
            if vertices.union(v.starts, v.ends):
                edge = Edge(v.ends, v.dist, MST[v.starts].edge)
                MST[v.starts].edge = edge
                edgeNum -= 1

    # 无权图单源最短路问题可以看作是对图的广度遍历
    def Dijkstra(self, source):  # 要求所有边权重都是非负数
        from heapq import heappop, heappush
        class Distance:
            def __init__(self, dist, pos):
                self.dist = dist
                self.pos = pos

            def __lt__(self, other):
                return self.dist < other.dist

        Dist = [float('inf')] * self.length  # 正无穷大
        Path = [source] * self.length
        collected = [False] * self.length

        Dist[source] = 0
        h = []
        heappush(h, Distance(Dist[source], source))
        while h:
            pos = heappop(h).pos  # 弹出dist最小的顶点并求得其坐标
            if not collected[pos]:
                collected[pos] = True
                edge = self.vertices[pos].edge
                while edge:
                    edgeIndex = edge.vertex
                    if Dist[edgeIndex] > edge.weight + Dist[pos]:
                        Dist[edgeIndex] = edge.weight + Dist[pos]
                        Path[edgeIndex] = pos
                        heappush(h, Distance(Dist[edgeIndex], edgeIndex))  # 由于Dist的初始化为inf,so从source开始的所有联通点都有机会进入堆h
                    edge = edge.right
        for path, dist in zip(Path, Dist):
            print(f'parent:{path} dist:{dist}')


if __name__ == '__main__':
    '''
          A
         / \
        B---C---D
        | \ |
        E---F
    '''
    graph = Graph()
    for edge in [('A', 'C', 2), ('A', 'B', 1), ('B', 'C', 5), ('C', 'D', 4), ('B', 'F', 7), ('B', 'E', 2),
                 ('E', 'F', 3), ('F', 'C', 6)]:
        graph.add(*edge)
    graph.BFS()
    graph.find_path('A', 'F')
    graph.show()

'''
多源最短路算法时间复杂度是O(V^3),适用于邻接矩阵表示的稠密图,稀疏图则可以迭代调用Dijkstra函数V次即可
graph=[
    [0,2,6,4],
    [float('inf'),0,3,float('inf')],
    [7,float('inf'),0,1],
    [5,float('inf'),12,0],
]
'''


def floyd(graph):
    length = len(graph)
    path = [[-1] * length for j in range(length)]
    for k in range(length):  # k要写外面,里面的i,j是对称的,随便嵌套没所谓
        for i in range(length):
            for j in range(length):
                if graph[i][j] > graph[i][k] + graph[k][j]:  # 加=不影响graph结果,但会影响path导致路径出错
                    graph[i][j] = graph[i][k] + graph[k][j]
                    path[i][j] = k

    def __show(i, j):
        if path[i][j] == -1:
            print(f'{i}=>{j}', end=' ')
        else:
            __show(i, path[i][j])
            __show(path[i][j], j)
            # print(f'{path[i][j]}=>{j}',end=' ')  # error

    for i in range(length):
        for j in range(length):
            __show(i, j)
            print(f'shortest path is {graph[i][j]}')
