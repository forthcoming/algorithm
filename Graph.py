from collections import deque

UDG=1  #无向图
DG=0   #有向图

class Edge:
    def __init__(self,vertex,weight,right=None):
        self.vertex=vertex
        self.weight=weight
        self.right=right

class Graph:  #邻接表存储
    def __init__(self,edges=[('A','C',2),('A','B',1),('B','C',5),('C','D',4),('B','F',7),('B','E',2),('E','F',3),('F','C',6)],kind=UDG):
        '''
              A
             / \
            B---C---D
            | \ |
            E---F
        '''
        self.__kind=kind
        self.__vertices={}
        self.__edge_num=0
        self.__vertex_num=0
        for edge in edges:
            self.add(edge)

    def add(self,edge):
        self.__edge_num+=1
        vertex_f=edge[0]
        vertex_t=edge[1]
        try:
            weight=edge[2]
        except IndexError:
            weight=0
        if vertex_f in self.__vertices:
            self.__vertices[vertex_f]=Edge(vertex_t,weight,self.__vertices[vertex_f])
        else:
            self.__vertex_num+=1
            self.__vertices[vertex_f]=Edge(vertex_t,weight)

        if self.__kind==UDG:
            if vertex_t in self.__vertices:
                self.__vertices[vertex_t]=Edge(vertex_f,weight,self.__vertices[vertex_t])
            else:
                self.__vertex_num+=1
                self.__vertices[vertex_t]=Edge(vertex_f,weight)

    def DFSTraverse(self):  #类似于树的先根遍历,有向图的时间复杂度是O(n+e),无向图是O(n+2e),其中e代表邻接点个数
        vertices=set()  # 保存已访问节点
        def _traverse(vertex):
            vertices.add(vertex)
            print(vertex,end=' ')
            edge=self.__vertices[vertex]
            while edge:
                vertex=edge.vertex
                if vertex not in vertices:
                    _traverse(vertex)
                edge=edge.right
        for vertex in self.__vertices:  #对于非连通图,尚需要从所有未被访问的顶点起调用_traverse
            if vertex not in vertices:
                _traverse(vertex)

    def BFSTraverse(self):
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
    
    def find_path(self,start,end):
        vertices=set()
        path=deque()
        result=[]
        def _find_path(start,end):
            vertices.add(start)
            path.append(start)
            if start==end:
                print(path)
                result.append([vertex for vertex in path])
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
    graph=Graph()
    graph.BFSTraverse()
    graph.find_path('A','F')
