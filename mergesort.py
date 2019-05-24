class MergeSort:  
    # 归并排序(稳定排序,时间复杂度永远是nlogn,跟数组的数据无关)
    def __init__(self,li):
        self.li = li              # 待排序数组
        self.inversion_number = 0 # 逆序数,针对recursive_sort方法有效

    def merge(self,left,mid,right): # 包含[left,mid],[mid+1,right]边界
        result=[]
        p1=left
        p2=mid+1
        while p1<=mid and p2<=right:
            if self.li[p1] < self.li[p2]:
                result.append(self.li[p1])
                p1 += 1
            else:
                result.append(self.li[p2])
                p2 += 1
                self.inversion_number+=mid-p1+1
        if p1<=mid:
            p2=right-mid+p1
            self.li[p2:right+1]=self.li[p1:mid+1]
        self.li[left:p2]=result

    def recursive_sort(self,left,right):  #递归版归并排序,包含left,right边界
        if left<right:
            mid = (left+right)>>1
            self.recursive_sort(left,mid)
            self.recursive_sort(mid+1,right)
            self.merge(left,mid,right)

    def reverse(self,left,right): #[::-1] or list.reverse
        while left<right:
            self.li[left],self.li[right]=self.li[right],self.li[left]
            left+=1
            right-=1

    def inplace_merge(self,left,mid,right): # 包含[left,mid],[mid+1,right]边界,效率低于merge
        mid+=1
        while left<mid and mid<=right:
            p=mid
            while left<mid and self.li[left]<=self.li[mid]:
                left+=1
            while mid<=right and self.li[mid]<=self.li[left]:
                mid+=1
            self.reverse(left,p-1)    
            self.reverse(p,mid-1)    
            self.reverse(left,mid-1)  
            left+=mid-p

    def iter_sort(self):   #迭代版归并排序
        length=len(self.li)
        initmid=0
        while initmid<length-1:
            step=(initmid+1)<<1
            for mid in range(initmid,length,step):
                left=mid-(step>>1)+1
                right=mid+(step>>1)  # right=left+step-1            
                if right>=length:
                    right=length-1
                self.inplace_merge(left,mid,right)
            initmid=(initmid<<1)+1

if __name__ == '__main__':
    import random
    li = list(range(10))
    random.shuffle(li)
    print(li)  # [0, 8, 4, 5, 7, 6, 3, 2, 1, 9]
    res = MergeSort(li)
    res.recursive_sort(0,len(li)-1)
    print(li,res.inversion_number)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 23
