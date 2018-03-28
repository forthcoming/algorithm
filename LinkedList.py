# 带头节点的单链表
class Node:
    def __init__(self, data,right=None):
        self.data = data
        self.right = right

    def __str__(self):
        return f'data:{self.data}'

class LinkedList:
    def __init__(self):
        self.__length=0
        self.__head = Node(None)

    def __len__(self):
        return self.__length

    def unshift(self, node):
        node.right = self.__head.right
        self.__head.right = node
        self.__length+=1

    def reverse(self):
        L = None
        cur=self.__head.right
        while cur:
            tmp = cur
            cur=cur.right   #注意他跟上一句的顺序
            tmp.right = L
            L = tmp
        self.__head.right = L

    def delete(self, node): #带头节点的删除操作更简单
        cur=self.__head
        while cur.right:
            if cur.right.data==node.data:
                cur.right=cur.right.right
                self.__length-=1
                return True
            cur=cur.right
        else:
            return False

    def insert(self, node, num):
        if 1 <= num <= self.__length:
            num -= 1
            cur = self.__head.right
            while num:
                cur = cur.right
                num -= 1
            node.right = cur.right
            cur.right = node
            self.__length+=1
            return True

    def traverse(self):
        cur = self.__head.right
        while cur:
            print(cur)
            cur = cur.right
        print()

    def reverseK(self,count): #分组反转链表
#         head=self.__head
#         while head.right:
#             tail=starts=head.right
#             if starts:
#                 ends=starts.right
#                 flag=count-1
#                 while ends and flag:
#                     cur=ends
#                     ends=ends.right
#                     cur.right=starts
#                     starts=cur
#                     flag-=1   
#                 tail.right=ends
#                 head.right=starts
#             head=tail

        start=self.__head
        end=cur=self.__head.right
        while cur:
            flag=count
            L=None
            while flag and cur:
                tmp=cur
                cur=cur.right
                tmp.right=L
                L=tmp
                flag-=1
            start.right=L
            start=end
            end=cur

    def init(self,node): #当且仅当链表为空时才有效
        if not self.__length:
            self.__head.right=node
            while node:
                node=node.right
                self.__length+=1
                
    def randinit(self,ranges=99,num=10):
        from random import randrange
        from functools import reduce
        def concat(node1,node2):
            node2.right=node1
            return node2
        self.__head.right=reduce(concat,[Node(randrange(0,ranges)) for i in range(num)])
        self.__length+=num

    @staticmethod
    def quickSort(starts,ends=None):
        if starts!=ends: #这里要特别注意,starts可能为空
            pre=starts
            cur=starts.right
            while cur!=ends:
                if cur.data<starts.data:
                    pre=pre.right
                    if pre!=cur:
                        pre.data,cur.data=cur.data,pre.data
                cur=cur.right
            pre.data,starts.data=starts.data,pre.data
            __class__.quickSort(starts,pre) #不包含pre
            __class__.quickSort(pre.right,ends)
                
    @staticmethod
    def filter(La,Lb,Lc):  #有序链表La,Lb,Lc,删除Lc中同时出现在三者之间的节点
        root=Lc
        pre=Lc.__head
        La=La.__head.right
        Lb=Lb.__head.right
        Lc=Lc.__head.right
        while La and Lb and Lc:
            if La.data<Lc.data:
                La=La.right
            elif Lb.data<Lc.data:
                Lb=Lb.right
            elif La.data==Lb.data==Lc.data:
                pre.right=Lc.right
                Lc=Lc.right
                root.__length-=1
            else:
                Lc=Lc.right
                pre=pre.right
                
        # while La and Lb and Lc:
        #     if La.data>Lb.data:
        #         Lb=Lb.right
        #     elif La.data<Lb.data:
        #         La=La.right
        #     else:
        #         if Lc.data<La.data:
        #             Lc=Lc.right
        #             pre=pre.right
        #         elif Lc.data==La.data:
        #             pre.right=Lc.right
        #             Lc=Lc.right
        #             root.__length-=1
        #         else:
        #             La=La.right
        #             Lb=Lb.right

if __name__=='__main__':
    La=LinkedList()
    La.init(Node(0,Node(1,Node(3,Node(5,Node(7,Node(9)))))))
    Lb=LinkedList()
    Lb.init(Node(1,Node(2,Node(3,Node(4,Node(5,Node(6)))))))
    Lc=LinkedList()
    Lc.init(Node(1,Node(3,Node(3,Node(5,Node(7,Node(9,Node(10,Node(11,Node(12))))))))))
    Lc.filter(La,Lb,Lc) 
    Lc.traverse()
    Lc.reverseK(2)
    Lc.traverse()
    
    Ld=LinkedList()
    Ld.randinit(9,11)
    Ld.quickSort(Ld._LinkedList__head.right)
    Ld.traverse()
