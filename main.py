import json
import statistics
from copy import deepcopy
from random import random, sample, randrange, randint, seed

import numpy as np

from input import fitness_function


def GeneticAlgorithm():
    # read input params from file
    with open('input.json') as json_file:
        input = json.load(json_file)
    # Initialize Population from given parameters
    population = Population(input=input)
    # Also Evaluate fitness and store the best parameter
    population.Evaluate()
    population.keep_the_best()
    print('\nOperation Running.')
    # loop until MAXGENS
    generation = 0  # counter
    while (generation < input['MAXGENS']):
        print("\n## Generation:", generation)
        population.Selection()
        population.Crossover()
        population.Mutate()
        population.Report(generation)
        population.Evaluate()
        population.Elitist()
        generation += 1  # counter update

    print('\nOperation Completed.')
    print(np.array(statLog), )  # print log
    print('The best solution is ', population.best['bestFitChromo'])


statLog, input = [], None
seed(2)  # random seed value


class Chromosome:
    def __init__(self, BOUNDS):
        self.gene = []  # init gene variable
        NVAR = len(BOUNDS)
        for i in range(NVAR):
            lower, upper = BOUNDS[i]
            self.gene.append(lower + (upper - lower) * random())  # random gene generation within boundary

    def mutate(self, BOUNDS):
        m = randrange(0, len(BOUNDS))
        lower, upper = BOUNDS[m]
        self.gene[m] = lower + (upper - lower) * random()  # gene mutation to random bound value
        print("Mutate ", m, " now ", self.gene)

    def __repr__(self):  # to display the object
        return repr(self.gene)


class Population:
    def __init__(self, input=None, ):
        self.input = input.copy()
        self.bounds, self.popsize = input['BOUNDS'], input['POPSIZE']
        self.chromosomes = [Chromosome(self.bounds) for i in range(self.popsize)]  # random generate chromosome
        self.fitness, self.best = None, None

    def keep_the_best(self):
        self.best = {"bestFitnessVal": max(self.fitness),
                     "bestFitChromo": self.chromosomes[self.fitness.index(max(self.fitness))]
                     }

    def Evaluate(self, ):
        self.fitness = [fitness_function(chromosome.gene) for chromosome in self.chromosomes]

    def Selection(self):
        totalFitness = sum(self.fitness)
        rfitness = np.array(self.fitness) / totalFitness
        cfitness = np.cumsum(rfitness)
        oldChromosomes = deepcopy(self.chromosomes)  # save old state of population chromosomes
        for i in range(self.popsize):
            justGreaterThan = next(
                i for i, v in enumerate(cfitness) if v >= random())  # just greater than a random num between 0 and 1
            self.chromosomes[i] = deepcopy(oldChromosomes[justGreaterThan])  # selected
            print('selected ', justGreaterThan, ' at ', i)

    def Crossover(self):
        # find number of population to go for crossover
        XOVER_num = int(self.input['PXOVER'] * self.popsize)
        XOVER_Popn = sample(range(0, self.popsize), XOVER_num)  # select population for crossover
        # improvement can be done with more than two candidate for crossover
        print("XOVER_Popn: ", XOVER_Popn, " Len: ", len(XOVER_Popn))
        old = deepcopy([(x.gene) for x in self.chromosomes])  # save old state of population chromosomes
        for x in XOVER_Popn:
            while True:
                y = randint(0, XOVER_num)
                if x != y: break
            # select  crossover  point
            b = randrange(0, len(self.bounds))
            self.chromosomes[x].gene[b] = old[y][b]  # crossover
            print('Cross ', x, "-", y, "@", b)

    def Mutate(self):
        MUTATION_num = int(self.input['PMUTATION'] * self.popsize)
        MUTATION_Popn = sample(range(0, self.popsize), MUTATION_num)
        print("Mutating ", MUTATION_Popn, ' len ', MUTATION_num)
        for x in MUTATION_Popn:
            self.chromosomes[x].mutate(self.bounds)

    def Elitist(self):
        old = deepcopy(self.best)
        best = {"bestFitnessVal": max(self.fitness),
                "bestFitChromo": self.chromosomes[self.fitness.index(max(self.fitness))]
                }

        if best['bestFitnessVal'] < old['bestFitnessVal']:
            worst = self.fitness.index(min(self.fitness))
            self.chromosomes[worst] = old['bestFitChromo']  # perform elitism
            self.fitness[worst] = old['bestFitnessVal']  # also update fitness value
            print('Elitism Performed for ', old['bestFitChromo'], ' FitnessVal: ', old['bestFitnessVal'])
        else:
            print('Elitism Not Performed')
        self.keep_the_best()  # select and store the best of all

    def Report(self, generation):
        stdev = statistics.stdev(self.fitness)
        # print("BestVal", self.best['bestFitnessVal'], " stdev ", stdev, " fit ", self.best)
        statLog.append({"Generation": generation,
                        "BestFit": self.best['bestFitnessVal'],
                        "std": stdev,
                        "chromosome": self.best['bestFitChromo']
                        })

    @property  # just for better display
    def array(self):
        return np.array([e for e in self.chromosomes]).reshape((self.popsize, -1))


if __name__ == '__main__':
    GeneticAlgorithm()
