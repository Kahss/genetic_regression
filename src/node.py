from enum import IntEnum
from typing import Optional, Union
import random 
import numpy as np

class NODE_TYPE(IntEnum):
    DUMMY = 0
    CONSTANT = 1
    ADD = 2
    SUBTRACT = 3
    MULTIPLY = 4
    DIVIDE = 5
    INPUT = 6

class TWO_ARGUMENTS_NODE(IntEnum):
    ADD = NODE_TYPE.ADD.value
    SUBTRACT = NODE_TYPE.SUBTRACT.value
    MULTIPLY = NODE_TYPE.MULTIPLY.value
    DIVIDE = NODE_TYPE.DIVIDE.value


type_to_func: dict[IntEnum, callable] = {
    NODE_TYPE.DUMMY: lambda x: x,
    NODE_TYPE.ADD: np.add,
    NODE_TYPE.SUBTRACT: np.subtract,
    NODE_TYPE.MULTIPLY: np.multiply,
    NODE_TYPE.DIVIDE: np.divide
}

class Node:
    def __init__(self, arguments: Optional[list] = None, value: Optional[float] = None, node_type: IntEnum = NODE_TYPE.DUMMY) -> None:
        self.node_type: str = node_type
        self.value: Optional[float] = value
        self.set_arguments(arguments)
        self.parent: Optional[Node] = None

        if arguments is not None and node_type in [NODE_TYPE.CONSTANT, NODE_TYPE.INPUT]:
            raise Exception("arguments should be None for input or constant node")
    
    def set_arguments(self, arguments: Optional[list]):
        if arguments is not None:
            for arg in arguments:
                arg.parent = self
        self.arguments = arguments

    def compute(self, input: np.ndarray = None):
        if input is not None and len(input.shape) == 1:
            input = input.reshape(1, input.size)

        if self.node_type == NODE_TYPE.CONSTANT:
            return self.value
        
        if self.node_type == NODE_TYPE.INPUT:
            _, max_dim = input.shape
            if self.value >= max_dim:
                self.value %= max_dim
            return input[:, self.value]

        func: callable = type_to_func[self.node_type]
        arguments_results: list[float] = [arg.compute(input) for arg in self.arguments]
        return func(*arguments_results)
    
    def clone(self):
        return Node(
            node_type=self.node_type,
            arguments=[arg.clone() for arg in self.arguments] if self.arguments is not None else None,
            value=self.value
        )
    
    def __str__(self) -> str:
        if self.node_type == NODE_TYPE.CONSTANT:
            return str(self.value)
        elif self.node_type == NODE_TYPE.INPUT:
            return f'x{self.value}'
        else:
            args_str = [str(arg) for arg in self.arguments]
            if self.node_type == NODE_TYPE.MULTIPLY:
                return ' * '.join(args_str)
            elif self.node_type == NODE_TYPE.DIVIDE:
                return ' / '.join(args_str)
            elif self.node_type == NODE_TYPE.ADD:
                return f"({' + '.join(args_str)})"
            elif self.node_type == NODE_TYPE.SUBTRACT:
                return f"({' - '.join(args_str)})"

    
    def flatten(self):
        node_list = [self]
        if self.arguments is not None:
            for arg in self.arguments:
                node_list.extend(arg.flatten())
        return node_list

def generate_random_tree(input_dim: int):
    root: Node = Node(arguments=[Node(), Node()], node_type=random.choice(list(TWO_ARGUMENTS_NODE)))
    current_node: Node = root

    for i in range(input_dim - 1):
        current_node.arguments[0] = Node(value=i, node_type=NODE_TYPE.INPUT)
        if i == (input_dim - 2):
            current_node.arguments[1] = Node(value=i+1, node_type=NODE_TYPE.INPUT)
        else:
            next_node: Node = Node(arguments=[Node(), Node()], node_type=random.choice(list(TWO_ARGUMENTS_NODE)))
            current_node.arguments[1] = next_node
            current_node = next_node

    return root