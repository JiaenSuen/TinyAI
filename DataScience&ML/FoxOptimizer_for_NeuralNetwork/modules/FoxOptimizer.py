import numpy as np
from modules.base import *
class FoxOptimizer:

    def __init__(self,
                 pop_size=10,
                 iterations=20,
                 alpha=0.5,
                 beta=0.3):

        self.pop_size = pop_size
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta


    def optimize(self, objective_fn, bounds):

        dim = len(bounds)

        # --- initialization ---
        population = np.random.rand(self.pop_size, dim)

        for d in range(dim):
            low, high = bounds[d]
            population[:, d] = low + population[:, d] * (high - low)

        fitness = np.array([objective_fn(p) for p in population])

        best_idx = np.argmax(fitness)
        best = population[best_idx].copy()
        best_score = fitness[best_idx]

        # --- optimization loop ---
        for it in range(self.iterations):

            alpha = self.alpha * (1 - it/self.iterations)

            for i in range(self.pop_size):

                r1 = np.random.rand(dim)
                r2 = np.random.randn(dim)

                rand_idx = np.random.randint(self.pop_size)
                x_rand = population[rand_idx]

                # --- improved FOA move ---
                new = (
                    population[i]
                    + r1 * (best - population[i])
                    + alpha * r2
                    + self.beta * (x_rand - population[i])
                )

                # clip bounds
                for d in range(dim):
                    low, high = bounds[d]
                    new[d] = np.clip(new[d], low, high)

                score = objective_fn(new)

                # greedy selection
                if score > fitness[i]:
                    population[i] = new
                    fitness[i] = score

                    if score > best_score:
                        best_score = score
                        best = new.copy()

            print(f"{RED}FOA iter {it} best = {best_score:.4f}{RESET}")

        return best
