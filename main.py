import json
import statistics
from copy import deepcopy
from random import seed, random, sample, randrange, randint

import numpy as np
import pandas as pd

from input import fitness_function


def GeneticAlgorithm(name):
    generation = 0
    with open('input.json') as json_file:
        input= json.load(json_file)

    population = Population(input=input)

    population.Evaluate()
    population.keep_the_best()
    print('\nOperation Running.')

    while (generation < input['MAXGENS']):
        print("\n## Generation:", generation)
        population.Selection()
        population.Crossover()
        population.Mutate()
        population.Report()
        population.Evaluate()
        population.Elitist()
        generation += 1

    df = pd.DataFrame(statLog)

    print('\nOperation Completed.')
    print(df)
    print('The best solution is ', population.best['bestFitChromo'])


statLog = []
seed(2)
input = None


class Chromosome:
    def __init__(self, BOUNDS):
        NVAR = len(BOUNDS)
        self.gene = []
        for i in range(NVAR):
            lower, upper = BOUNDS[i]
            self.gene.append(lower + (upper - lower) * random())

    def mutate(self, BOUNDS):
        m = randrange(0, len(BOUNDS))
        lower, upper = BOUNDS[m]
        self.gene[m] = lower + (upper - lower) * random()
        print("Mutate ", m, " now ", self.gene)

    def __repr__(self):
        return repr(self.gene)


class Population:
    def __init__(self, input=None, fitness=0, chromosomes=None, ):
        self.bounds = input['BOUNDS']
        self.fitness = fitness
        self.popsize = input['POPSIZE']
        self.input = input.copy()
        self.best = None

        if chromosomes:
            self.chromosomes = chromosomes
        else:
            self.chromosomes = []
            # no chromosome given so generate
            for i in range(self.popsize):
                self.chromosomes.append(Chromosome(self.bounds))

    # return as array for visulization
    def __repr__(self):
        return repr(np.array([e for e in self.chromosomes]))

    @property
    def array(self):
        return np.array([e for e in self.chromosomes]).reshape((self.popsize, -1))

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
        oldChromosomes = deepcopy(self.chromosomes)
        for i in range(self.popsize):
            randomNumber = random()
            justGreaterThan = next(i for i, v in enumerate(cfitness) if v >= randomNumber)
            self.chromosomes[i] = deepcopy(oldChromosomes[justGreaterThan])
            print('selected ', justGreaterThan, ' at ', i)

    def Crossover(self):
        # find number of population to go for crossover
        XOVER_num = int(self.input['PXOVER'] * self.popsize)
        XOVER_Popn = sample(range(0, self.popsize), XOVER_num)  # select population for crossover
        # improvement can be done with more than two candidate for crossover
        print("XOVER_Popn: ", XOVER_Popn, " Len: ", len(XOVER_Popn))
        old = deepcopy([(x.gene) for x in self.chromosomes])
        for x in XOVER_Popn:
            while True:
                y = randint(0, XOVER_num)
                if x != y: break
            # select  crossover  point
            b = randrange(0, len(self.bounds))
            xm, my = x, y
            xc, cx = self.chromosomes[x].gene[b], old[y][b]
            self.chromosomes[x].gene[b] = old[y][b]
            print('Cross ', x, "-", y, "@", b)

    def Mutate(self):
        MUTATION_num = int(self.input['PMUTATION'] * self.popsize)
        MUTATION_Popn = sample(range(0, self.popsize), MUTATION_num)
        print("Mutating ", MUTATION_Popn, ' len ', MUTATION_num)
        for x in MUTATION_Popn:
            self.chromosomes[x].mutate(self.bounds)

    def Report(self):
        stdev = statistics.stdev(self.fitness)
        # print("BestVal", self.best['bestFitnessVal'], " stdev ", stdev, " fit ", self.best)
        statLog.append({"bestVal": self.best['bestFitnessVal'],
                        " stdev ": stdev,
                        " fit ": self.best['bestFitChromo']
                        })

    def Elitist(self):
        old = deepcopy(self.best)
        best = {"bestFitnessVal": max(self.fitness),
                "bestFitChromo": self.chromosomes[self.fitness.index(max(self.fitness))]
                }

        if best['bestFitnessVal'] < old['bestFitnessVal']:
            worst = self.fitness.index(min(self.fitness))
            self.chromosomes[worst] = old['bestFitChromo']
            self.fitness[worst] = old['bestFitnessVal']
            self.keep_the_best()
            print('Elitism Performed for ', old['bestFitChromo'], ' FitnessVal: ', old['bestFitnessVal'])
            pass

        else:
            self.keep_the_best()
            print('Elitism Not Performed')
            pass


if __name__ == '__main__':
    GeneticAlgorithm('GA')
