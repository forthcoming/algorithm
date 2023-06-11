"""
有限状态自动机FSM包括确定有限自动机deterministic finite automata(DFA)、非确定有限自动机non-deterministic finite automata(NFA)等
DFA由(Q,∑,q0,F,δ)组成,Q是有限状态集合,∑是有限输入字符集合,q0是初始状态且属于Q,F是可接受状态且属于Q,δ是状态转移函数
FSM状态转移图是一个有向图,接受状态用双圈表示,一个箭头指向表示开始状态顶点
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
        if from_state in self.__transitions:
            self.__transitions[from_state][symbol] = to_state
        else:
            self.__transitions[from_state] = {symbol: to_state}

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
    result = dfa.process("1111")

    print(result)
