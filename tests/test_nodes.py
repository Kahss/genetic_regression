import numpy as np
from src.node import Node, NODE_TYPE, generate_random_tree

def test_constant_node():
    node = Node(value=1, node_type=NODE_TYPE.CONSTANT)
    assert node.compute() == 1

def test_add_node():
    node1 = Node(value=1, node_type=NODE_TYPE.CONSTANT)
    node2 = Node(value=2, node_type=NODE_TYPE.CONSTANT)
    node_add = Node(arguments=[node1, node2], node_type=NODE_TYPE.ADD)
    assert node_add.compute() == 3

def test_add_node_with_1_dim_input():
    node1 = Node(value=0, node_type=NODE_TYPE.INPUT)
    node2 = Node(value=1, node_type=NODE_TYPE.INPUT)
    node_add = Node(arguments=[node1, node2], node_type=NODE_TYPE.ADD)
    assert node_add.compute(np.array([1, 2])) == np.array(3)

def test_add_node_with_2_dim_input():
    node1 = Node(value=0, node_type=NODE_TYPE.INPUT)
    node2 = Node(value=1, node_type=NODE_TYPE.INPUT)
    node_add = Node(arguments=[node1, node2], node_type=NODE_TYPE.ADD)
    results = node_add.compute(np.asarray([[1, 2], [3, 4]]))
    target = np.asarray([3, 7])
    assert (results == target).all()

def test_generate_random_individual():
    node = generate_random_tree(input_dim=3)
    assert isinstance(node, Node)

def test_flatten_tree():
    node1 = Node(value=1, node_type=NODE_TYPE.CONSTANT)
    node2 = Node(value=2, node_type=NODE_TYPE.CONSTANT)
    node_add = Node(arguments=[node1, node2], node_type=NODE_TYPE.ADD)
    node_list = node_add.flatten()
    assert node1 in node_list
    assert node2 in node_list
    assert node_add in node_list

def test_str_tree():
    node1 = Node(value=1, node_type=NODE_TYPE.CONSTANT)
    node2 = Node(value=2, node_type=NODE_TYPE.CONSTANT)
    node_add = Node(arguments=[node1, node2], node_type=NODE_TYPE.ADD)
    assert str(node_add) == "(1 + 2)"