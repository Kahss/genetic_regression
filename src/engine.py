import random

import numpy as np

import config
from src.node import NODE_TYPE, TWO_ARGUMENTS_NODE, Node


def rank_nodes(nodes: list[Node], X: np.ndarray, y: np.ndarray, order="ascending"):
    ## Evaluate individuals
    results = [evaluate(indiv, X=X, y=y) for indiv in nodes]

    ## Select bests
    indices = np.argsort(results)

    if order == "descending":
        indices = np.flip(indices)

    return [nodes[i] for i in indices]

def evaluate(node: Node, X: np.ndarray, y: np.ndarray):
    output = node.compute(X) 
    return np.sum(np.square(output - y))

def crossover(nodes: list[Node]):
    return breed(*random.sample(nodes, 2))

def breed(node_root: Node, node_branch: Node):
    node_root = node_root.clone()
    node_branch = node_branch.clone()
    list_node_root = node_root.flatten()
    list_node_branch = node_branch.flatten()
    base_node = random.choice(list_node_root)
    branch_node = random.choice(list_node_branch)
    branch_node.parent = base_node.parent

    if base_node.parent is None:
        return branch_node

    for i, child in enumerate(base_node.parent.arguments):
        if child == base_node:
            base_node.parent.arguments[i] = branch_node
            break
    
    return node_root


def mutate(nodes: list[Node], p_mutation=config.P_MUTATE):
    selected_node: Node = random.choice(nodes).clone()
    return crawl(selected_node, p_mutation)

def crawl(node: Node, p_mutation: float):
    returned_node: Node = node
    if random.random() < p_mutation:
        if node.node_type in iter(TWO_ARGUMENTS_NODE):
            original_type = node.node_type
            while node.node_type == original_type:
                node.node_type = random.choice(list(TWO_ARGUMENTS_NODE))
        elif node.node_type == NODE_TYPE.CONSTANT:
            node.value += 2*random.random() - 1
        elif node.node_type == NODE_TYPE.INPUT:
            if random.random() < (1 - config.P_MUTATE_NEW_FACTOR):
                node.value += random.choice([-1, 1])
            else:
                node_constant = Node(value=10*random.random(), node_type=NODE_TYPE.CONSTANT)
                node_parent = node.parent
                node_factor = Node(arguments=[node, node_constant], node_type=random.choice((NODE_TYPE.ADD, NODE_TYPE.MULTIPLY)))
                node_factor.parent = node_parent
                returned_node = node_factor
                if node_parent is not None:
                    for i, child in enumerate(node_parent.arguments):
                        if child == node:
                            node_parent.arguments[i] = node_factor


    if node.arguments is not None:
        node.set_arguments([crawl(arg, p_mutation=p_mutation) for arg in node.arguments])
        
    return returned_node