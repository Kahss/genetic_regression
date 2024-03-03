import numpy as np
from tqdm import tqdm

import src.engine as engine
from src.node import generate_random_tree

N_INDIV = 1000
N_EPOCHS = 100

N_INPUT_DIM = 2
N_SAMPLES = 100
NOISE_FACTOR = 1/20

P_BEST = .5
P_BREED = .25
P_MUTATE = .25

# Source data
X = np.random.rand(N_SAMPLES, N_INPUT_DIM)
X0 = X[:, 0]
X1 = X[:, 1]
y = X0 + 2*X1 + X0 * X1 + NOISE_FACTOR * np.random.rand(N_SAMPLES)

# Generate initial population
population = [generate_random_tree(X.shape[1]) for _ in range(N_INDIV)]
bests = []

# Loop
for _ in tqdm(range(N_EPOCHS)):
    ## Champions
    champions = engine.rank_nodes(nodes=population, X=X, y=y)[:int(N_INDIV*P_BEST)]

    ## Crossover
    children = [engine.crossover(champions) for _ in range(int(N_INDIV*P_MUTATE))]

    ## Mutate
    mutants = [engine.mutate(champions) for _ in range(int(N_INDIV*P_MUTATE))]

    ## Save best
    bests.append(champions[0])

    ## New population
    population = [*champions, *children, *mutants]

substr = [f"{i} : {str(bests[i])}" for i in range(len(bests))]
print("Sub Champions")
print("\n".join(substr))
print(f"Best solution : {str(bests[-1])}")