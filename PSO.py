import math
import time
import torch
from tqdm import tqdm


# Optimization Test Functions
def rastrigin(pop):
    if pop.dim() == 1:
        pop = pop.unsqueeze(0)
    dim = pop.size(dim=-1)
    return (10 * dim) + \
        torch.sum(torch.pow(pop, 2) - (10 * torch.cos(2 * torch.pi * pop)), dim=1).reshape(-1, 1)


def ackley(pop, a=20, b=0.2, c=2 * math.pi):
    if pop.dim() == 1:
        pop = pop.unsqueeze(0)
    dim = pop.size(dim=-1)
    first = -a * torch.exp(-b * torch.sqrt(1 / dim * torch.sum(torch.pow(pop, 2), dim=1)))
    second = torch.exp(1 / dim * torch.sum(torch.cos(c * pop), dim=1))
    return (first - second + a + math.exp(1)).reshape(-1, 1)


def PSO(f, max_iter, n, dim, w, c1, c2, init_min_x, init_max_x, device):
    # All the data containers
    positions = (init_max_x - init_min_x) * torch.rand(size=(n, dim), dtype=torch.float, device=device) + init_min_x
    velocities = torch.zeros(size=(n, dim), dtype=torch.float, device=device)
    personal_best_pos = torch.rand(size=(n, dim), dtype=torch.float, device=device)
    personal_bests = float('inf') * torch.ones(size=(n, 1), dtype=torch.float, device=device)
    best_pos = torch.zeros(size=(1, dim), dtype=torch.float, device=device)
    best_val = torch.tensor(float('inf'), dtype=torch.float, device=device)

    curr_iter = 0
    pbar = tqdm(total=max_iter)
    while curr_iter < max_iter:
        # First calculate fitnesses
        fitnesses = f(positions)

        # Update personal best
        personal_best_pos = torch.where(fitnesses < personal_bests, positions, personal_best_pos)
        personal_bests = torch.minimum(fitnesses, personal_bests)

        # Update global best
        bestSol = torch.min(personal_bests, dim=0)
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

        curr_iter += 1
        pbar.update(1)

    return best_val.item(), best_pos.tolist()[0]


def CPSO(f, max_iter, n, dim, dim_split, w, c1, c2, init_min_x, init_max_x, device):
    """
    Cooperative Particle Swarm Optimization.

    Assumes minimization problem.

    Source:
    Van den Bergh, Frans, and Andries P. Engelbrecht.
    "A cooperative approach to particle swarm optimization."
    IEEE transactions on evolutionary computation 8.3 (2004): 225-239.

    :param f: optimization function
    :param max_iter: maximum iteration limit
    :param n: number of particles in each sub-swarm
    :param dim: dimension of optimization problem
    :param dim_split: number of sub-swarms
    :param w: inertia coefficient
    :param c1: cognitive coefficient
    :param c2: social coefficient
    :param init_min_x: minimum initial position for swarm(s) generation
    :param init_max_x: maximum initial position for swarm(s) generation
    :param device: 'cpu' or 'cuda' for gpu
    :return: (best fitness, best position)
    """
    # Create the data containers
    decision_variables = torch.tensor_split(torch.randperm(dim), dim_split)
    context_vector = (init_max_x - init_min_x) * torch.rand(size=(dim,), dtype=torch.float, device=device) + init_min_x
    subswarm_positions = [
        (init_max_x - init_min_x) * torch.rand(size=(n, tensor.shape[0]), dtype=torch.float, device=device) + init_min_x
        for tensor in decision_variables]
    subswarm_velocities = [torch.zeros(size=(n, tensor.shape[0]), dtype=torch.float, device=device) for tensor in
                           decision_variables]
    personal_best_pos = [torch.zeros(size=(n, tensor.shape[0]), dtype=torch.float, device=device) for tensor in
                         decision_variables]
    personal_bests = [torch.full(size=(n, 1), fill_value=float('inf'), dtype=torch.float, device=device) for _ in
                      decision_variables]
    g_best_pos = [torch.tensor(float('inf'), dtype=torch.float, device=device) for _ in range(len(subswarm_positions))]
    g_bests = [torch.tensor(float('inf'), dtype=torch.float, device=device) for _ in range(len(subswarm_positions))]

    curr_iter = 0
    pbar = tqdm(total=max_iter)
    while curr_iter < max_iter:
        for i in range(len(subswarm_positions)):
            # First calculate fitnesses (must use context vector)
            context_vector = context_vector.repeat(n, 1)
            context_vector[:, decision_variables[i]] = subswarm_positions[i]
            fitnesses = f(context_vector)

            # Update personal bests
            improved_pbests = (fitnesses < personal_bests[i]).flatten()
            personal_best_pos[i][improved_pbests] = subswarm_positions[i][improved_pbests].detach().clone()
            personal_bests[i] = torch.minimum(fitnesses, personal_bests[i])

            # Update global best
            best_pbest_idx = torch.argmin(personal_bests[i])
            if personal_bests[i][best_pbest_idx] < g_bests[i]:
                g_bests[i] = personal_bests[i][best_pbest_idx].detach().clone()
                g_best_pos[i] = personal_best_pos[i][best_pbest_idx].detach().clone()

            context_vector = context_vector[0]
            context_vector[decision_variables[i]] = g_best_pos[i]  # context vector is the global best of all subswarms

            # Velocity equation
            r1 = torch.rand(size=subswarm_velocities[i].shape, dtype=torch.float,
                            device=device)  # random coeff 1 vector
            r2 = torch.rand(size=subswarm_velocities[i].shape, dtype=torch.float,
                            device=device)  # random coeff 2 vector
            inertia = subswarm_velocities[i] * w
            cognitive = r1 * c1 * (personal_best_pos[i] - subswarm_positions[i])
            social = r2 * c2 * (g_best_pos[i] - subswarm_positions[i])

            # Velocity update
            subswarm_velocities[i] = inertia + cognitive + social

            # Position update
            subswarm_positions[i] += subswarm_velocities[i]

        curr_iter += 1
        pbar.update(1)

    return f(context_vector).item(), context_vector.tolist()


def execAlgo(algo, *args):
    """

    :param algo: algorithm to use for optimization
    :param args: arguments for algorithm (in proper order)
    :return: result of algorithm used as input
    """
    startTime = time.time()
    result = algo(*args)
    endTime = time.time()
    print(f"time: {endTime - startTime}")
    return result


# hyper parameters
w = 0.729  # inertia
c1 = 1.49445  # cognitive (particle)
c2 = 1.49445  # social (swarm)

print("##### Rastrigin #####")
print("10000 iter; 100 particles; 100 dim")
execAlgo(PSO, rastrigin, 10000, 100, 100, w, c1, c2, -5.12, 5.12, 'cuda')
execAlgo(PSO, rastrigin, 10000, 100, 100, w, c1, c2, -5.12, 5.12, 'cpu')

print("10000 iter; 500 particles; 100 dim")
execAlgo(PSO, rastrigin, 10000, 500, 100, w, c1, c2, -5.12, 5.12, 'cuda')
execAlgo(PSO, rastrigin, 10000, 500, 100, w, c1, c2, -5.12, 5.12, 'cpu')

print("10000 iter; 100 particles; 500 dim")
execAlgo(PSO, rastrigin, 10000, 100, 500, w, c1, c2, -5.12, 5.12, 'cuda')
execAlgo(PSO, rastrigin, 10000, 100, 500, w, c1, c2, -5.12, 5.12, 'cpu')

print("10000 iter; 1000 particles; 100 dim")
execAlgo(PSO, rastrigin, 10000, 1000, 100, w, c1, c2, -5.12, 5.12, 'cuda')
execAlgo(PSO, rastrigin, 10000, 1000, 100, w, c1, c2, -5.12, 5.12, 'cpu')

print("10000 iter; 100 particles; 1000 dim")
execAlgo(PSO, rastrigin, 10000, 100, 1000, w, c1, c2, -5.12, 5.12, 'cuda')
execAlgo(PSO, rastrigin, 10000, 100, 1000, w, c1, c2, -5.12, 5.12, 'cpu')

print("10000 iter; 1000 particles; 1000 dim")
execAlgo(PSO, rastrigin, 10000, 1000, 1000, w, c1, c2, -5.12, 5.12, 'cuda')
execAlgo(PSO, rastrigin, 10000, 1000, 1000, w, c1, c2, -5.12, 5.12, 'cpu')

print("##### ACKLEY #####")
print("10000 iter; 100 particles; 100 dim")
execAlgo(PSO, ackley, 10000, 100, 100, w, c1, c2, -5.12, 5.12, 'cuda')
execAlgo(PSO, ackley, 10000, 100, 100, w, c1, c2, -5.12, 5.12, 'cpu')

print("10000 iter; 500 particles; 100 dim")
execAlgo(PSO, ackley, 10000, 500, 100, w, c1, c2, -5.12, 5.12, 'cuda')
execAlgo(PSO, ackley, 10000, 500, 100, w, c1, c2, -5.12, 5.12, 'cpu')

print("10000 iter; 100 particles; 500 dim")
execAlgo(PSO, ackley, 10000, 100, 500, w, c1, c2, -5.12, 5.12, 'cuda')
execAlgo(PSO, ackley, 10000, 100, 500, w, c1, c2, -5.12, 5.12, 'cpu')

print("10000 iter; 1000 particles; 100 dim")
execAlgo(PSO, ackley, 10000, 1000, 100, w, c1, c2, -5.12, 5.12, 'cuda')
execAlgo(PSO, ackley, 10000, 1000, 100, w, c1, c2, -5.12, 5.12, 'cpu')

print("10000 iter; 100 particles; 1000 dim")
execAlgo(PSO, ackley, 10000, 100, 1000, w, c1, c2, -5.12, 5.12, 'cuda')
execAlgo(PSO, ackley, 10000, 100, 1000, w, c1, c2, -5.12, 5.12, 'cpu')

print("10000 iter; 1000 particles; 1000 dim")
execAlgo(PSO, ackley, 10000, 1000, 1000, w, c1, c2, -5.12, 5.12, 'cuda')
execAlgo(PSO, ackley, 10000, 1000, 1000, w, c1, c2, -5.12, 5.12, 'cpu')
