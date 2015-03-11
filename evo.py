import numpy as np
import random

def main():
    function = lambda x: (x**2).sum()
    population_size = 100
    param_size = 10
    population = np.array(np.random.uniform(-100, 100, [population_size, param_size]), dtype=np.int)
    population = sorted(population, key=function)
    population = population[:20]

    for gen in range(1000):
        for i in range(80):
            a = random.randint(0, 20-1)
            b = random.randint(0, 20-1)
            cross_point = random.randint(2, param_size-2)
            old_a, old_b = np.copy(population[a]), np.copy(population[b])
            population[a][:cross_point] = old_b[:cross_point]
            population[b][cross_point:] = old_b[cross_point:]

    print map(function, population)






if __name__ == '__main__':
    main()