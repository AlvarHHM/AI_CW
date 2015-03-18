import numpy as np
import random


def main():
    function = lambda x: (np.array(x)**2).sum().tolist()
    population_size = 1000
    parent_size = 100
    param_size = 5
    population = np.array(np.random.uniform(-100.0, 100.0, [population_size, param_size]), dtype=np.float).tolist()

    for gen in range(1000):
        population = sorted(population, key=function)
        population = population[:parent_size]
        print map(function, population)
        for i in range(population_size-parent_size):
            a = random.randint(0, 20-1)
            b = random.randint(0, 20-1)
            cross_point = random.randint(2, param_size-2)
            new_being = population[a][:cross_point] + population[b][cross_point:]
            new_being = map(lambda x: x*(random.random()*2) if (random.randint(0, 9) == 1) else x, new_being)
            population.append(new_being)

    print map(function, population)


if __name__ == '__main__':
    main()