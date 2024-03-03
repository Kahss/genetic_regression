import numpy as np
import pytest

import src.engine as engine
from src.node import NODE_TYPE, Node

@pytest.fixture
def add_tree():
    node0 = Node(value=0, node_type=NODE_TYPE.INPUT)
    node1 = Node(value=1, node_type=NODE_TYPE.INPUT)
    node_to_evaluate = Node(arguments=[node0, node1], node_type=NODE_TYPE.ADD)
    return node_to_evaluate

@pytest.fixture
def multiply_tree():
    node0 = Node(value=0, node_type=NODE_TYPE.INPUT)
    node1 = Node(value=1, node_type=NODE_TYPE.INPUT)
    node_to_evaluate = Node(arguments=[node0, node1], node_type=NODE_TYPE.MULTIPLY)
    return node_to_evaluate

def test_evaluate_node(add_tree: Node):
    results = engine.evaluate(
        node=add_tree,
        X=np.asarray([[1, 2], [3, 4]]),
        y=np.asarray([4, 8])
    )
    assert results == 2

def test_rank_nodes():
    X=np.asarray([[1, 2], [3, 4]])
    y=np.asarray([3, 7])

    nodes = []
    node0 = Node(value=0, node_type=NODE_TYPE.INPUT)
    node1 = Node(value=1, node_type=NODE_TYPE.INPUT)

    for i in range(3):
        node_constant = Node(value=(i+1)%3, node_type=NODE_TYPE.CONSTANT)
        node_branch = Node(arguments=[node1, node_constant], node_type=NODE_TYPE.ADD)
        nodes.append(Node(arguments=[node0, node_branch], node_type=NODE_TYPE.ADD))

    ranked_nodes = engine.rank_nodes(nodes, X=X, y=y)

    for i in range(3):
        assert ranked_nodes[i].arguments[1].arguments[1].value == i

def test_crawl_mutate_node(add_tree: Node):
    node_to_crawl = add_tree.clone()
    mutant = engine.crawl(node_to_crawl, p_mutation=1)
    assert mutant.node_type != NODE_TYPE.ADD
    assert mutant.arguments[0].arguments != 0
    assert mutant.arguments[1].arguments != 1

def test_crossover_two_trees(add_tree: Node, multiply_tree: Node):
    for _ in range(20):
        child = engine.crossover([add_tree, multiply_tree])
        assert isinstance(child, Node)