class Queue:
    def __init__(self,maxlen=1000,incrementsize=100):
        self.front=self.rear=0
        self.maxlen=maxlen
        self.incrementsize=incrementsize
        self.data=[None]*maxlen

    def length(self):
        return (self.rear-self.front+self.maxlen)%self.maxlen

    def pop(self):
        if self.front==self.rear:  #空队列
            return False,None
        else:
            value=self.data[self.front]
            self.front=(self.front+1)%self.maxlen
            return True,value

    def push(self,value):
        if (self.rear+1)%self.maxlen==self.front:
            self.increment()
        self.data[self.rear]=value
        self.rear=(self.rear+1)%self.maxlen

    def increment(self):  #比较耗时，当不知道队列可能达到的长度时推荐用链队列
        print('increment')
        new=[None]*(self.maxlen+self.incrementsize)
        for i in range(self.maxlen-1):
            new[i]=self.data[(self.front+i)%self.maxlen]
        self.data=new
        self.front=0
        self.rear=self.maxlen-1
        self.maxlen+=self.incrementsize

    def traverse(self):
        index=self.front
        while index!=self.rear:
            print(self.data[index])
            index=(index+1)%self.maxlen
