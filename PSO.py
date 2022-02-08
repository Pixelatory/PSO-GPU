import math
import time
import torch
from tqdm import tqdm


# Optimization Test Functions
def rastrigin(pop):
    dim = pop.size(dim=-1)
    return (10 * dim) + \
           torch.sum(torch.pow(pop, 2) - (10 * torch.cos(2 * torch.pi * pop)), dim=1).reshape(-1, 1)


def ackley(pop, a=20, b=0.2, c=2 * math.pi):
    dim = pop.size(dim=-1)
    first = -a * torch.exp(-b * torch.sqrt(1 / dim * torch.sum(torch.pow(pop, 2), dim=1)))
    second = torch.exp(1 / dim * torch.sum(torch.cos(c * pop), dim=1))
    return (first - second + a + math.exp(1)).reshape(-1, 1)


def PSO(f, max_iter, n, dim, init_min_x, init_max_x, device):
    # hyper parameters
    w = 0.729  # inertia
    c1 = 1.49445  # cognitive (particle)
    c2 = 1.49445  # social (swarm)

    # All the data containers
    positions = (init_max_x - init_min_x) * torch.rand(size=(n, dim), dtype=torch.float, device=device) + init_min_x
    velocities = torch.zeros(size=(n, dim), dtype=torch.float, device=device)
    personal_best_pos = torch.rand(size=(n, dim), dtype=torch.float, device=device)
    personal_bests = float('inf') * torch.ones(size=(n, 1), dtype=torch.float, device=device)
    best_pos = torch.zeros(size=(1, dim), dtype=torch.float, device=device)
    best_val = torch.tensor(float('inf'), dtype=torch.float, device=device)

    iter = 0
    pbar = tqdm(total=max_iter)
    while iter < max_iter:
        # First calculate fitnesses
        fitnesses = f(positions)

        # Update personal best
        personal_best_pos = torch.where(fitnesses < personal_bests, positions, personal_best_pos)
        personal_bests = torch.minimum(fitnesses, personal_bests)

        # Update global best
        bestSol = torch.min(personal_bests, 0)
        best_pos = personal_best_pos[bestSol[1]].detach().clone()
        best_val = bestSol[0]

        # print(best_pos, best_val)

        # Velocity equation
        r1 = torch.rand(size=(n, dim), dtype=torch.float, device=device)  # random coeff 1 vector
        r2 = torch.rand(size=(n, dim), dtype=torch.float, device=device)  # random coeff 2 vector
        inertia = velocities * w
        cognitive = r1 * c1 * (personal_best_pos - positions)
        social = r2 * c2 * (best_pos - positions)

        # velocity update and constraint
        velocities = inertia + cognitive + social

        positions += velocities  # position update

        iter += 1
        pbar.update(1)

    return best_val, best_pos


def execPSO(f, iter, n, dim, init_min, init_max, device):
    startTime = time.time()
    PSO(f, iter, n, dim, init_min, init_max, device)
    endTime = time.time()
    print(endTime - startTime)

print("##### Rastrigin #####")
print("10000 iter; 100 particles; 100 dim")
execPSO(rastrigin, 10000, 100, 100, -5.12, 5.12, 'cuda')
execPSO(rastrigin, 10000, 100, 100, -5.12, 5.12, 'cpu')

print("10000 iter; 500 particles; 100 dim")
execPSO(rastrigin, 10000, 500, 100, -5.12, 5.12, 'cuda')
execPSO(rastrigin, 10000, 500, 100, -5.12, 5.12, 'cpu')

print("10000 iter; 100 particles; 500 dim")
execPSO(rastrigin, 10000, 100, 500, -5.12, 5.12, 'cuda')
execPSO(rastrigin, 10000, 100, 500, -5.12, 5.12, 'cpu')

print("10000 iter; 1000 particles; 100 dim")
execPSO(rastrigin, 10000, 1000, 100, -5.12, 5.12, 'cuda')
execPSO(rastrigin, 10000, 1000, 100, -5.12, 5.12, 'cpu')

print("10000 iter; 100 particles; 1000 dim")
execPSO(rastrigin, 10000, 100, 1000, -5.12, 5.12, 'cuda')
execPSO(rastrigin, 10000, 100, 1000, -5.12, 5.12, 'cpu')

print("10000 iter; 1000 particles; 1000 dim")
execPSO(rastrigin, 10000, 1000, 1000, -5.12, 5.12, 'cuda')
execPSO(rastrigin, 10000, 1000, 1000, -5.12, 5.12, 'cpu')

print("##### ACKLEY #####")
print("10000 iter; 100 particles; 100 dim")
execPSO(ackley, 10000, 100, 100, -5.12, 5.12, 'cuda')
execPSO(ackley, 10000, 100, 100, -5.12, 5.12, 'cpu')

print("10000 iter; 500 particles; 100 dim")
execPSO(ackley, 10000, 500, 100, -5.12, 5.12, 'cuda')
execPSO(ackley, 10000, 500, 100, -5.12, 5.12, 'cpu')

print("10000 iter; 100 particles; 500 dim")
execPSO(ackley, 10000, 100, 500, -5.12, 5.12, 'cuda')
execPSO(ackley, 10000, 100, 500, -5.12, 5.12, 'cpu')

print("10000 iter; 1000 particles; 100 dim")
execPSO(ackley, 10000, 1000, 100, -5.12, 5.12, 'cuda')
execPSO(ackley, 10000, 1000, 100, -5.12, 5.12, 'cpu')

print("10000 iter; 100 particles; 1000 dim")
execPSO(ackley, 10000, 100, 1000, -5.12, 5.12, 'cuda')
execPSO(ackley, 10000, 100, 1000, -5.12, 5.12, 'cpu')

print("10000 iter; 1000 particles; 1000 dim")
execPSO(ackley, 10000, 1000, 1000, -5.12, 5.12, 'cuda')
execPSO(ackley, 10000, 1000, 1000, -5.12, 5.12, 'cpu')
