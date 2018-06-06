'''
n个顶点的无向图最多含n(n-1)/2条边,有向图最多含n(n-1)条边
G(V,E)满足2*E=所有顶点的度之和
图可以用邻接矩阵或邻接表表示
邻接矩阵是一个对角线元素为0的n阶方阵,无向图邻接矩阵是对称矩阵
'''

# to be continued

from collections import deque

UDG=1  #无向图
DG=0   #有向图

class Edge:
    def __init__(self,vertex,weight,right=None):
        self.vertex=vertex
        self.weight=weight
        self.right=right

class Graph:  #邻接表存储
    def __init__(self,kind=UDG):
        self.__kind=kind
        self.__vertices={}
        self.__edge_num=0

    def add(self,come,to,weight=0):
        self.__edge_num+=1
        if come in self.__vertices:
            self.__vertices[come]=Edge(to,weight,self.__vertices[come])
        else:
            self.__vertices[come]=Edge(to,weight)

        if self.__kind==UDG:
            if to in self.__vertices:
                self.__vertices[to]=Edge(come,weight,self.__vertices[to])
            else:
                self.__vertices[to]=Edge(come,weight)
    
    def delete(self,come,to):
        if (come in self.__vertices) and (to in self.__vertices):    
            edge=self.__vertices[come]
            if edge:
                self.__edge_num-=1
                if edge.vertex==to:
                    self.__vertices[come]=edge.right
                else:
                    pre=edge
                    edge=edge.right
                    while edge:
                        if edge.vertex==to:
                            pre.right=edge.right
                            break
                        pre=edge
                        edge=edge.right
        
            if self.__kind=='UDG': # undirected graph
                edge=self.__vertices[to]
                if edge:
                    if edge.vertex==come:
                        self.__vertices[to]=edge.right
                    else:
                        pre=edge
                        edge=edge.right
                        while edge:
                            if edge.vertex==come:
                                pre.right=edge.right
                                break
                            pre=edge
                            edge=edge.right
                        
    def _DFS(self,vertex,vertices=set()): # don't assign to vertices 
        if vertex not in vertices:
            vertices.add(vertex)
            print(vertex,end=' ')
            edge=self.__vertices[vertex]
            while edge:
                vertex=edge.vertex
                if vertex not in vertices:  # 非必需,加判断减少递归次数
                    self._DFS(vertex)
                edge=edge.right

    def DFS(self):    #类似于树的先根遍历,有向图的时间复杂度是O(n+e),无向图是O(n+2e),其中e代表邻接点个数   
        for vertex in self.__vertices:  # For non connected graphs
            self._DFS(vertex)
            
    def DFS_stack(self):  # 非递归
        stack=[]
        tmp=[]
        vertices={vertex for vertex in self.__vertices} # 保存未访问节点
        for vertex in self.__vertices:
            stack.append(vertex)
            while stack:
                vertex=stack.pop()
                if vertex in vertices:  # 不能少
                    vertices.remove(vertex)
                    print(vertex,end=' ')    
                edge=self.__vertices[vertex]
                while edge:
                    vertex=edge.vertex
                    if vertex in vertices:  # 不能少
                        tmp.append(vertex)
                    edge=edge.right
                while tmp:
                    stack.append(tmp.pop())
                    
    def BFS(self):
        vertices=set()
        queue=deque()
        for vertex in self.__vertices:
            if vertex not in vertices:
                vertices.add(vertex)
                print(vertex,end=' ')               
                queue.append(self.__vertices[vertex])
            while queue:
                edge=queue.popleft()
                while edge:
                    vertex=edge.vertex
                    if vertex not in vertices:
                        vertices.add(vertex)
                        print(vertex,end=' ')
                        queue.append(self.__vertices[vertex])
                    edge=edge.right
    
    def find_path(self,start,end):  # 回溯法,常与DFS配合使用
        vertices=set()
        path=[]
        result=[]
        def _find_path(start,end):
            vertices.add(start)
            path.append(start)
            if start==end:
                print(path)
                result.append(path[:])  # 注意这里是深拷贝
            else:
                edge=self.__vertices[start]
                while edge:
                    vertex=edge.vertex
                    if vertex not in vertices:
                        _find_path(vertex,end)
                    edge=edge.right
            vertices.remove(start)
            path.pop()
        _find_path(start,end)
        return result
        '''
        已知有向图graph如下,求给定两点间的所有路径(1代表可达)
        graph=[  # 有向图
            [0,1,1,0,0,1],
            [1,0,0,0,0,0],
            [0,1,0,1,0,0],
            [0,1,0,0,0,1],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
        ] 
        ''' 

if __name__=='__main__':
    '''
          A
         / \
        B---C---D
        | \ |
        E---F
    '''
    graph=Graph()
    for edge in [('A','C',2),('A','B',1),('B','C',5),('C','D',4),('B','F',7),('B','E',2),('E','F',3),('F','C',6)]:
        graph.add(*edge)
    graph.BFS()
    graph.find_path('A','F')
