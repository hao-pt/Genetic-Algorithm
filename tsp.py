import numpy as np
import math
import random
from operator import itemgetter, attrgetter
from bit_manager import Number, NumberArray

# Set seed
random.seed(174)

class City:
    """
    City is a specific location (x, y) as gene
    """
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def distance(self, other):
        """
        Compute distance between 2 cities
        """
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __str__(self):
        return "(%.2f, %.2f)"%(self.x, self.y)

class Map:
    """Map stores list of cities (locations)"""
    def __init__(self):
        self.map = [] # list of cites

    def insert(self, city: City):
        """Insert new city into map (list)"""
        self.map.append(city)

    def get(self):
        return self.map

    def size(self):
        return len(self.map)

class Tour:
    """Tour is a individual (aka chromosome) which is one satisfied solution"""
    def __init__(self, city_map: Map, tour=[]):
        self.map = city_map
        self.tour = tour
        if self.size() == 0:
            self.createNewTour()
        self.fitness = self.compute_fitness()

    def createNewTour(self):
        """Create new tour by sampling randomly map without replacement"""
        self.tour = random.sample(self.map.get(), self.map.size())
        return self.tour

    def compute_tour_cost(self):
        """Compute the cost of tour by sum over distances"""
        cost = 0.0
        n = len(self.tour) # Number of cities
        for i in range(n):
            if i + 1 < n:
                cost += self.tour[i].distance(self.tour[i + 1])
            else:
                cost += self.tour[i].distance(self.tour[0])

        return cost

    def compute_fitness(self):
        """Compute total distance from city i to n and n to i"""
        
        # Inverse fittness because we want total distance as small as posible
        self.fitness = math.e / self.compute_tour_cost()
        return self.fitness

    def get_fitness(self):
        return self.fitness

    def get_cost(self):
        return math.e / self.fitness

    def size(self):
        return len(self.tour)

    def __str__(self):
        """Return the representation of tour"""
        string = ""
        for city in self.tour:
            string = string + str(city) + " -> "
        string += str(self.tour[0])
        return string
            
class Population:
    """Population is a collection of posible tours (individuals)"""
    def __init__(self, population_size, city_map, initial=False):
        self.population = []
        self.population_size = population_size
        self.map = city_map

        if initial:
            self.initPopulation()
        
    def initPopulation(self):
        """Initialize first population"""
        for i in range(self.population_size):
            new_tour = Tour(self.map)
            self.population.append(new_tour)

        self.sortPopulation()
        return self.population

    def samplePopulation(self, k):
        """Sample population with replacement"""
        sampledPopulation = Population(self.population_size, self.map)
        sampledPopulation.population = random.choices(self.population, k=k)
        sampledPopulation.sortPopulation()
        return sampledPopulation

    def get_fittess_tour(self):
        """Get fittest tour with max fitness"""
        # max_index = 0
        # for i in range(1, self.population_size):
        #     if self.population[i].fitness > self.population[max_index]:
        #         max_index = i
        # return self.population[max_index]
        return self.population[0]

    def sortPopulation(self):
        """Sort population base on tour fitness in descending order"""
        self.population = sorted(self.population, key=attrgetter('fitness'), reverse=True)

    def insert(self, new_tour):
        """Insert new tour"""
        self.population.append(new_tour)
        self.sortPopulation()

    def getPopulation(self):
        """Get population list"""
        return self.population

    def size(self):
        return self.population_size

    def __str__(self):
        string = ""
        for i, tour in enumerate(self.population):
            string = string + str(i) + " | " + str(tour)
            string +=  " | " + str(tour.get_cost())
            string += "\n"
        return string

class Evoluation:
    """Evoluation class implements genetic functions that are useful for algorithm.

        Evolve next generation includes:
            Selection
            Cross-over
            Mutate
    """
    def __init__(self, city_map, population_size=50, mutation_rate=0.015, elitism_size=10, pop=None, tournament_size=5):
        self.map = city_map # Construct map
        self.mutation_rate = mutation_rate # low probability to mutate
        if pop != None:
            self.population = pop
            self.population_size = pop.size()
        else:
            self.population = Population(population_size, city_map)
            self.population_size = population_size
        
        self.elitism_size = elitism_size # Number of best individuals in next generation
        self.tournament_size = tournament_size

    def setPop(self, pop):
        """Set new population and update population size"""
        self.population = pop
        self.population_size = pop.size()

    def getPop(self):
        return self.population

    def selection(self):
        """Perform selection by sampling randomly the current population with replacement
        and pick the fittest tour"""
        sampledPopulation = self.population.samplePopulation(self.tournament_size)
        return sampledPopulation.get_fittess_tour()

    def cross_over(self, parent1, parent2):
        """Cross-over (breeding) between parents"""
        startGene = int(random.random() * parent1.size())
        endGene = int(random.random() * parent1.size())

        if startGene > endGene:
            startGene, endGene = endGene, startGene

        child = [None for i in range(parent1.size())]

        # print(startGene, endGene)
        for i in range(startGene, endGene):
            child[i] = parent1.tour[i]
        
        parent_idx = 0
        for i in range(0, parent1.size()):
            # Fill only empty positions by genes of parent2 in order
            if child[i] == None:
                while parent_idx < parent2.size():
                    if parent2.tour[parent_idx] not in child:
                        child[i] = parent2.tour[parent_idx]
                        parent_idx += 1
                        break
                    parent_idx += 1

        return Tour(self.map, child)
    
    def mutatation(self, indiv):
        """Mutation function swap the two random genes of particular individual
        so the new individual never has missing or duplicate gene that satisfies the constraints of TSP"""
        for i in range(indiv.size()):
            if random.random() < self.mutation_rate:
                j = int(random.random() * indiv.size())

                # Swap 2 genes (cities)
                indiv.tour[i], indiv.tour[j] = indiv.tour[j], indiv.tour[i]

        # Update fitness
        indiv.compute_fitness()
        return indiv

    def evolve_generation(self):
        """
        Evolve next generation:
            Selection
            Cross-over
            Mutate

        Return:
            new_population
        """
        # Init new population
        new_population = Population(self.population_size, self.map)

        # Get current population
        current_population = self.population.getPopulation()

        # print(self.elitism_size)

        # Keep first best (elitism_size) of tour
        for i in range(0, self.elitism_size):
            new_population.insert(current_population[i])
        
        for i in range(self.elitism_size, self.population_size):
            # Randomly select parents
            father = self.selection()
            # print(father)
            
            mother = self.selection()
            
            if mother == father:
                mother = self.selection()
            # print(mother)

            # Breeding (cross-over)
            child = self.cross_over(father, mother)
            # print(child)

            # Mutatation respects to low probability (mutation_rate)
            child = self.mutatation(child)
            # print(child)

            # Add child into new_population
            new_population.insert(child)

        return new_population

class GA:
    """
    Genetic algorithm function wraps up all the processes to give the final results

        Processes:
            Initialize population
            Compute fittness values of population
            Evolve next generation
                Selection
                Cross-over
                Mutate
            Loop until no_generations has reached
    """
    def __init__(self, city_map=Map(), population_size=50, no_generations=100, mutation_rate=0.015, elitism_rate=0.2, tournament_size=5):
        """
        Init function

        Args:
            no_generations (int): the number of generations are considered (stopping condition)
            population_size (int): the size of population
            city_map (Map object): stores the list of cities (genes)
            mutation_rate: the low probability for mutation
            elitism_rate: the rate of best individuals to keep in next generation
            tournament_size: a set of individuals sample randomly from population to select the one with high fitness as parent

        """
        self.map = city_map
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism_size = int(self.population_size * elitism_rate) # Number of best individuals in next generation
        self.no_generations = no_generations
        self.tournament_size = tournament_size
        # Init evol object
        self.evol = Evoluation(self.map, population_size=self.population_size, mutation_rate=self.mutation_rate, elitism_size=self.elitism_size, tournament_size=self.tournament_size)

    def insert_city(self, city_point):
        """
        Insert new city (point) into map
        """
        city = City(city_point)
        self.map.insert(city)


    def genetic_algorithm(self):
        """
        Genetic algorithm function wrap up all the processes to give the final results

        Processes:
            Initialize population
            Compute fittness values of population
            Evolve next generation
                Selection
                Cross-over
                Mutate
            Loop until no_generations has reached

        Args:
            no_generations (int): the number of generations are considered (stopping condition)
            population_size (int): the size of population
            city_map (Map object): stores the list of cities (genes)
            mutation_rate: the low probability for mutation
            elitism_size: Number of best individuals in next generation
        """

        # Init first population
        pop = Population(self.population_size, self.map, initial=True)
        # print(pop)
        print("Initial tour:", pop.get_fittess_tour())
        print("Initial tour cost:", pop.get_fittess_tour().get_cost())
        print("Initial tour fitness:", pop.get_fittess_tour().get_fitness())

        # Set first pop
        self.evol.setPop(pop)
        
        for i in range(self.no_generations):
            # Evolve next generation
            pop = self.evol.evolve_generation()
            # print(pop)

            # Set new pop
            self.evol.setPop(pop)

        return self.evol.getPop().get_fittess_tour()
    



def main():
    city_map = Map()
    # Create and add our cities
    city = City(60, 200)
    city_map.insert(city)
    city2 = City(180, 200)
    city_map.insert(city2)
    city3 = City(80, 180)
    city_map.insert(city3)
    city4 = City(140, 180)
    city_map.insert(city4)
    city5 = City(20, 160)
    city_map.insert(city5)
    city6 = City(100, 160)
    city_map.insert(city6)
    city7 = City(200, 160)
    city_map.insert(city7)
    city8 = City(140, 140)
    city_map.insert(city8)
    city9 = City(40, 120)
    city_map.insert(city9)
    city10 = City(100, 120)
    city_map.insert(city10)
    city11 = City(180, 100)
    city_map.insert(city11)
    city12 = City(60, 80)
    city_map.insert(city12)
    city13 = City(120, 80)
    city_map.insert(city13)
    city14 = City(180, 60)
    city_map.insert(city14)
    city15 = City(20, 40)
    city_map.insert(city15)
    city16 = City(100, 40)
    city_map.insert(city16)
    city17 = City(200, 40)
    city_map.insert(city17)
    city18 = City(20, 20)
    city_map.insert(city18)
    city19 = City(60, 20)
    city_map.insert(city19)
    city20 = City(160, 20)
    city_map.insert(city20)


    ga = GA(city_map, population_size=50, no_generations=200, mutation_rate=0.15, elitism_rate=0.15, tournament_size=10)


    fittess_tour = ga.genetic_algorithm()
    print("Fittess tour:", fittess_tour)
    print("Final cost:", fittess_tour.get_cost())
    print("Final fitness:", fittess_tour.get_fitness())

    
if __name__ == "__main__":
    main()

