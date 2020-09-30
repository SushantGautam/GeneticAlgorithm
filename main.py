import json
import statistics
from random import seed, random, sample, randrange, randint

seed(1)
input = None

import numpy as np


class Chromosome:
    def __init__(self, BOUNDS):
        NVAR = len(BOUNDS)
        self.gene = []
        for i in range(NVAR):
            lower, upper = BOUNDS[i]
            self.gene.append(lower + (upper - lower) * random())

    def mutate(self, BOUNDS):
        m = randrange(len(BOUNDS))
        lower, upper = BOUNDS[m]
        self.gene[m] = lower + (upper - lower) * random()
        (lower + (upper - lower) * random())

    def __repr__(self):
        return repr(self.gene)


class Population:
    def __init__(self, BOUNDS, POPSIZE, fitness=0, chromosomes=None, input=None):
        self.bounds = BOUNDS
        self.fitness = fitness
        self.popsize = POPSIZE
        self.input = input.copy()
        self.best = None

        if chromosomes:
            self.chromosomes = chromosomes
        else:
            self.chromosomes = []
            # no chromosome given so generate
            for i in range(self.popsize):
                self.chromosomes.append(Chromosome(BOUNDS))

    # return as array for visulization
    def __repr__(self):
        return repr(np.array([e for e in self.chromosomes]))

    def keep_the_best(self):
        self.best = {"bestFitnessVal": max(self.fitness),
                     "bestFitChromo": self.chromosomes[self.fitness.index(max(self.fitness))]
                     }

    def Evaluate(self, ):
        def fitness_function(chromosome):
            gene = chromosome.gene
            return gene[0] * gene[1] + gene[1] * gene[0] + gene[2]

        self.fitness = [fitness_function(chromosome) for chromosome in self.chromosomes]

    def Selection(self):
        totalFitness = sum(self.fitness)
        rfitness = np.array(self.fitness) / totalFitness
        cfitness = np.cumsum(rfitness)
        for i in range(self.popsize):
            randomNumber = random()
            justGreaterThan = next(i for i, v in enumerate(cfitness) if v >= randomNumber)
            oldChromosomes = self.chromosomes.copy()
            self.chromosomes[i] = oldChromosomes[justGreaterThan]

        return

    def Crossover(self):
        # find number of population to go for crossover
        XOVER_num = int(self.input['PXOVER'] * self.popsize)
        XOVER_Popn = sample(range(0, self.popsize), XOVER_num)  # select population for crossover
        # improvement can be done with more than two candidate for crossover
        old = self.chromosomes.copy()
        for x in XOVER_Popn:
            while True:
                y = randint(0, XOVER_num)
                if x != y: break
            # select  crossover  point
            b = randrange(len(self.bounds))
            # swap
            self.chromosomes[x].gene[b], self.chromosomes[y].gene[b] = old[y].gene[b], old[x].gene[b]
        return self

    def Mutate(self):
        MUTATION_num = int(self.input['PMUTATION'] * self.popsize)
        MUTATION_Popn = sample(range(0, self.popsize), MUTATION_num)
        for x in MUTATION_Popn:
            self.chromosomes[x].mutate(self.bounds)

        return self

    def Report(self):
        stdev = statistics.stdev(self.fitness)
        print(self.best['bestFitnessVal'], stdev, self.fitness)
        return self

    def Elitist(self):
        best = {"bestFitnessVal": max(self.fitness),
                "bestFitChromo": self.chromosomes[self.fitness.index(max(self.fitness))]
                }
        old = self.best

        if best['bestFitnessVal'] <= old['bestFitnessVal']:
            worst = self.fitness.index(min(self.fitness))
            self.chromosomes[worst] = old['bestFitChromo']
            self.best = old
        else:
            self.keep_the_best()


def main(name):
    # global input
    generation = 0

    with open('input.json') as json_file:
        input = json.load(json_file)
        BOUNDS = input['BOUNDS']
        POPSIZE = input['POPSIZE']

        population = Population(BOUNDS, POPSIZE, input=input)
        population.Evaluate()
        population.keep_the_best()

        while (generation < input['MAXGENS']):
            print("Generation", generation)
            generation += 1
            population.Selection()
            population.Crossover()
            population.Mutate()
            population.Report()
            population.Evaluate()
            population.Elitist()


if __name__ == '__main__':
    main('PyCharm')
