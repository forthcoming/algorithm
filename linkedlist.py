from functools import reduce
from random import randrange


# 带头节点的单链表
class Node:
    def __init__(self, data, right=None):
        self.data = data
        self.right = right

    def __str__(self):
        return f'data:{self.data}'


class LinkedList:
    def __init__(self):
        self.__length = 0
        self.__head = Node(None)

    def __len__(self):
        return self.__length

    def traverse(self):
        cur = self.__head.right
        while cur:
            print(cur, end="\t")
            cur = cur.right
        print()

    def reverse(self):  # 反转链表
        head = None
        cur = self.__head.right
        while cur:
            tmp = cur
            cur = cur.right  # 注意他跟上一句的顺序
            tmp.right = head
            head = tmp
        self.__head.right = head

    def recur_reverse(self):
        def _reverse(root):
            if root and root.right:
                rev_linked_list = _reverse(root.right)
                root.right.right = root
                root.right = None  # 不要忘记
                return rev_linked_list
            else:
                return root

        self.__head.right = _reverse(self.__head.right)

    def delete(self, node):  # 带头节点的删除操作更简单
        cur = self.__head
        while cur.right:
            if cur.right.data == node.data:
                cur.right = cur.right.right
                self.__length -= 1
                return True
            cur = cur.right
        else:
            return False

    def insert(self, node, pos=1):
        assert 0 <= pos <= self.__length
        cur = self.__head
        while pos:
            cur = cur.right
            pos -= 1
        node.right = cur.right
        cur.right = node
        self.__length += 1

    def reverse_k(self, count):  # 分组反转链表
        # def _reverse_k(root):
        #     tail = root
        #     head = None
        #     _count = count
        #     while root and _count:
        #         _ = root
        #         root = root.right
        #         _.right = head
        #         head = _
        #         _count -= 1
        #     if root:
        #         tail.right = _reverse_k(root)
        #     return head
        #
        # self.__head.right = _reverse_k(self.__head.right)

        start = self.__head
        cur = start.right
        while cur:
            times = count
            head = None
            end = cur
            while cur and times:
                tmp = cur
                cur = cur.right
                tmp.right = head
                head = tmp
                times -= 1
            start.right = head
            start = end

    def init(self, node):  # 当且仅当链表为空时才有效
        if not self.__length:
            self.__head.right = node
            while node:
                node = node.right
                self.__length += 1

    def rand_init(self, ranges=99, num=10):
        def _concat(node1, node2):
            node2.right = node1
            return node2

        self.__head.right = reduce(_concat, [Node(randrange(0, ranges)) for _ in range(num)])
        self.__length += num

    @staticmethod
    def filter(la, lb, lc):  # 有序链表la,lb,lc,删除lc中同时出现在三者之间的节点
        root = lc
        pre = lc.__head
        la = la.__head.right
        lb = lb.__head.right
        lc = lc.__head.right
        while la and lb and lc:
            if la.data < lc.data:
                la = la.right
            elif lb.data < lc.data:
                lb = lb.right
            elif la.data == lb.data == lc.data:  # 此时la.data,lb.data一定大于等于lc.data
                pre.right = lc.right
                lc = lc.right
                root.__length -= 1
            else:
                lc = lc.right
                pre = pre.right

        # while la and lb and lc:
        #     if la.data > lb.data:
        #         lb = lb.right
        #     elif la.data < lb.data:
        #         la = la.right
        #     else:
        #         if lc.data < la.data:
        #             lc = lc.right
        #             pre = pre.right
        #         elif lc.data == la.data:
        #             pre.right = lc.right
        #             lc = lc.right
        #             root.__length -= 1
        #         else:
        #             la = la.right
        #             lb = lb.right

    @staticmethod
    def merge(starts, mid, ends):  # 合并2个以starts,mid开头ends(None)结尾的有序单链表
        if starts.data > mid.data:
            '''保证头指针始终指向链表的头位置,避免遍历链表得到starts前驱结点'''
            L = Node(starts.data, starts.right)
            starts.data = mid.data
            mid = mid.right

            # starts.data,mid.data=mid.data,starts.data
            # L=mid
            # mid=mid.right
            # L.right=starts.right
        else:
            L = starts.right
        while L != ends and mid != ends:
            if L.data < mid.data:
                starts.right = L  # 这里别漏掉了
                starts = L
                L = L.right
            else:
                starts.right = mid
                starts = mid
                mid = mid.right
        starts.right = L if L else mid

    @staticmethod
    def merge_sort(starts, ends=None):
        if starts.right != ends:  # 至少包含了两个节点
            pre = Node(None, starts)
            mid = cur = starts
            # while cur.right and cur.right.right:  这种写法有问题,应为mid应该指向中间的后一个位置
            while cur != ends and cur.right != ends:  # 不要漏掉cur!=ends检验
                mid = mid.right
                pre = pre.right
                cur = cur.right.right
            pre.right = ends  # 这里是重点!
            __class__.merge_sort(starts, ends)  # 不包含ends
            __class__.merge_sort(mid, ends)
            __class__.merge(starts, mid, ends)


if __name__ == '__main__':
    La = LinkedList()
    La.init(Node(0, Node(1, Node(3, Node(5, Node(7, Node(9)))))))
    Lb = LinkedList()
    Lb.init(Node(1, Node(2, Node(3, Node(4, Node(5, Node(6)))))))
    Lc = LinkedList()
    Lc.init(Node(1, Node(3, Node(3, Node(5, Node(7, Node(9, Node(10, Node(11, Node(12))))))))))
    Lc.filter(La, Lb, Lc)
    Lc.traverse()
    Lc.reverse_k(2)
    Lc.traverse()

    Ld = LinkedList()
    Ld.init(Node(0, Node(11, Node(3, Node(5, Node(17, Node(9)))))))
    Ld.merge_sort(Ld._LinkedList__head.right)
    Ld.traverse()
