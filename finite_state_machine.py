"""
有限状态自动机FSM包括确定有限自动机deterministic finite automata(DFA)、非确定有限自动机non-deterministic finite automata(NFA)等
FSM状态转移图是一个有向图,接受状态用双圈表示,一个箭头指向表示开始状态顶点
正则文法跟FSM一一对应
DFA由M=(Q,∑,q0,F,δ)组成,Q是有限状态集合,∑是有限输入字符集合,q0是初始状态且属于Q,F是可接受状态且属于Q,δ是状态转移函数
NFA的M'中只有δ与DFA不同,区别是NFA在接收到字母后可以有多个转移函数,某个状态可以不处理接受到的字母,即存在dead state
DFA是一种特殊的NFA,任意一个NFA都可以找到一个与之等价的DFA,假设NFA当前状态为n个,与之等价的DFA状态最多为2^n个
NFA在状态A下接收字符1后可到状态A或者B,可接受状态为C,等价的DFA则是在状态A下接收字符1后可到状态AB,可接受状态为包含C的组合状态,转换后的DFA还可以进一步简化为状态最少的DFA
DFA接受语言的补集(以ab结尾的ab串补集是不以ab结尾的ab串),对应状态转移图中接受状态改为非接受状态,非接受状态改为接受状态,NFA接受语言的补集需要先转换成DFA
FSM接受语言的交集(起始状态分别为A,B,可接受状态分别为C,D),交集对应状态转移图的起始状态是AB,可接受状态为CD,然后根据原先的状态转移图构造交集状态转移图
"""


class DFA:
    def __init__(self):
        self.__states = set()  # Q
        self.__alphabet = set()  # ∑
        self.__transitions = {}  # δ
        self.__initial_state = None  # q0
        self.__final_states = set()  # F

    def add_state(self, states):
        self.__states |= set(states)

    def add_alphabet(self, alphabet):
        self.__alphabet |= set(alphabet)

    def set_initial_state(self, state):
        self.__initial_state = state

    def add_final_state(self, states):
        self.__final_states |= set(states)

    def add_transition(self, from_state, symbol, to_state):
        self.__transitions.setdefault(from_state, {})[symbol] = to_state

    def process(self, alphabet):
        current_state = self.__initial_state
        for symbol in alphabet:
            if symbol not in self.__alphabet:
                raise ValueError("Invalid symbol in input")
            if current_state not in self.__transitions or symbol not in self.__transitions[current_state]:
                return False
            current_state = self.__transitions[current_state][symbol]
        return current_state in self.__final_states


if __name__ == "__main__":
    dfa = DFA()
    """
    判断二进制串能否被3整除的状态转移表
        字母0    1
    状态        
     零     零   一
     一     二   零
     二     一   二
    """
    dfa.add_state(["零", "一", "二"])
    dfa.add_alphabet("01")
    dfa.set_initial_state("零")
    dfa.add_final_state(["零"])
    dfa.add_transition("零", "0", "零")
    dfa.add_transition("零", "1", "一")
    dfa.add_transition("一", "0", "二")
    dfa.add_transition("一", "1", "零")
    dfa.add_transition("二", "0", "一")
    dfa.add_transition("二", "1", "二")
    result = dfa.process("01111110")  # 126,或者用正则语言/^1((10*1)|(01*0))*10*$/判断

    print(result)
