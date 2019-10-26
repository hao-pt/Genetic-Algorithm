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
        self.list_cities = [] # list of cites

    def insert(self, city: City):
        """Insert new city into map (list)"""
        self.list_cities.append(city)

    def get(self) -> list:
        return self.list_cities

    def get_city(self, i) -> City:
        return self.list_cities[i]

    def size(self):
        return len(self.list_cities)

class Tour:
    """Tour is a individual (aka chromosome) which is one satisfied solution"""
    def __init__(self, tour_size: int, tour_ids: "list/NumberArray"=None, list_cities: list=[]):
        self.tour_size = tour_size # The number of cities in tour
        self.no_bits_per_index = math.ceil(math.log2(self.tour_size)) + 1 # include 1 sign bit
        self.tour_ids = NumberArray(self.no_bits_per_index, self.tour_size) # NumberArray of city indices
        if tour_ids == None:
            self.createNewTour()
        elif isinstance(tour_ids, list):
            self.tour_ids[:] = tour_ids
        elif isinstance(tour_ids, NumberArray):
            self.tour_ids = tour_ids

        self.fitness = self.compute_fitness(list_cities)

    def createNewTour(self):
        """Create new tour by sampling randomly map without replacement"""
        n = self.tour_size
        indices = list(range(n))
        self.tour_ids[:] = random.sample(indices, n)
        return self.tour_ids

    def compute_tour_cost(self, list_cities: list) -> float:
        """Compute the cost of tour by sum over distances"""
        cost = 0.0
        n = self.tour_size # Number of cities
        for i in range(n):
            if i + 1 < n:
                cost += list_cities[self.tour_ids[i]].distance(list_cities[self.tour_ids[i + 1]])
            else:
                cost += list_cities[self.tour_ids[i]].distance(list_cities[self.tour_ids[0]])

        return cost

    def compute_fitness(self, list_cities: list) -> float:
        """Compute total distance from city i to n and n to i"""
        
        # Inverse fittness because we want total distance as small as posible
        self.fitness = math.e / self.compute_tour_cost(list_cities)
        return self.fitness

    def get_tour(self, list_cities):
        """Get tour (list of City)"""
        return [list_cities[i] for i in self.tour_ids]

    def get_fitness(self) -> float:
        """Get fitness value that equals e / cost"""
        return self.fitness

    def get_cost(self) -> float:
        """Get cost for this tour"""
        return math.e / self.fitness

    def size(self):
        return len(self.tour_ids)

    def __str__(self):
        """Return the representation of tour"""
        string = ""
        for i in self.tour_ids:
            string = string + str(i) + " -> "
        string += str(self.tour_ids[0])
        return string
            
class Population:
    """Population is a collection of posible tours (individuals)"""
    def __init__(self, population_size, city_map, initial=False):
        self.population = [] # List of NumberArray indices
        self.population_size = population_size
        self.map = city_map

        if initial:
            self.initPopulation()
        
    def initPopulation(self) -> list:
        """Initialize first population"""
        n = self.map.size() # Number of cities
        for i in range(self.population_size):
            new_tour = Tour(tour_size=n, list_cities=self.map.get())
            self.population.append(new_tour)

        self.sortPopulation()
        return self.population

    def samplePopulation(self, k) -> "Population":
        """Sample population with replacement"""
        sampledPopulation = Population(self.population_size, self.map)
        sampledPopulation.population = random.choices(self.population, k=k)
        sampledPopulation.sortPopulation()
        return sampledPopulation

    def get_fittess_tour(self) -> NumberArray:
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

    def insert(self, new_tour: Tour):
        """Insert new tour"""
        self.population.append(new_tour)

    def getPopulation(self) -> list:
        """Get population list"""
        return self.population

    def size(self) -> int:
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
    def __init__(self, city_map: Map, population_size=50, mutation_rate=0.015, elitism_size=10, pop: Population=None, tournament_size=5):
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

    def setPop(self, pop: Population):
        """Set new population and update population size"""
        self.population = pop
        self.population_size = pop.size()

    def getPop(self) -> Population:
        return self.population

    def selection(self) -> Tour:
        """Perform selection by sampling randomly the current population with replacement
        and pick the fittest tour"""
        sampledPopulation = self.population.samplePopulation(self.tournament_size)
        return sampledPopulation.get_fittess_tour()

    def cross_over(self, father: Tour, mother: Tour) -> Tour:
        """Cross-over (breeding) between parents"""
        startGene = int(random.random() * father.size())
        endGene = int(random.random() * father.size())

        # Swap 2 position if start > end
        if startGene > endGene:
            startGene, endGene = endGene, startGene

        # Init child as list
        child = [None for i in range(father.size())]

        # print(startGene, endGene)
        for i in range(startGene, endGene):
            child[i] = father.tour_ids[i]
        
        mother_idx = 0
        for i in range(0, father.size()):
            # Fill only empty positions by genes of mother in order
            if child[i] == None:
                while mother_idx < mother.size():
                    if mother.tour_ids[mother_idx] not in child:
                        child[i] = mother.tour_ids[mother_idx]
                        mother_idx += 1
                        break
                    mother_idx += 1

        return Tour(self.map.size(), child, self.map.get())
    
    def mutatation(self, indiv: Tour) -> Tour:
        """Mutation function swap the two random genes of particular individual
        so the new individual never has missing or duplicate gene that satisfies the constraints of TSP"""
        n = indiv.size()
        for i in range(n):
            if random.random() < self.mutation_rate:
                j = int(random.random() * n)

                # Swap 2 genes (cities)
                indiv.tour_ids[i], indiv.tour_ids[j] = indiv.tour_ids[j], indiv.tour_ids[i]

        # Update fitness
        indiv.compute_fitness(self.map.get())
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

        # Sort population (list of tour) base on fitness
        new_population.sortPopulation()
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
    def __init__(self, city_map=Map(), population_size=50, no_generations=100,\
                 mutation_rate=0.015, elitism_rate=0.2, tournament_size=5, print_cost_per_gen=5):
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
        self.print_cost_per_gen = print_cost_per_gen

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
        print("{:*^120}".format("Initial fittes tour"))        
        print("Tour:", pop.get_fittess_tour())
        print("Cost:", pop.get_fittess_tour().get_cost())
        # print("Initial tour fitness:", pop.get_fittess_tour().get_fitness())

        # Set first pop
        self.evol.setPop(pop)
        
        for i in range(self.no_generations):
            # Evolve next generation
            pop = self.evol.evolve_generation()
            
            if i % self.print_cost_per_gen == 0:
                print("{:-^50}Generation {}{:-^50}".format("", i, ""))
                print("Tour:", pop.get_fittess_tour())
                print("Tour cost: {}".format(pop.get_fittess_tour().get_cost()))

            # Set new pop
            self.evol.setPop(pop)

        return self.evol.getPop().get_fittess_tour()
    



def main():
    city_map = Map()
    # Create and add our cities
    # city = City(60, 200)
    # city_map.insert(city)
    # city2 = City(180, 200)
    # city_map.insert(city2)
    # city3 = City(80, 180)
    # city_map.insert(city3)
    # city4 = City(140, 180)
    # city_map.insert(city4)
    # city5 = City(20, 160)
    # city_map.insert(city5)
    # city6 = City(100, 160)
    # city_map.insert(city6)
    # city7 = City(200, 160)
    # city_map.insert(city7)
    # city8 = City(140, 140)
    # city_map.insert(city8)
    # city9 = City(40, 120)
    # city_map.insert(city9)
    # city10 = City(100, 120)
    # city_map.insert(city10)
    # city11 = City(180, 100)
    # city_map.insert(city11)
    # city12 = City(60, 80)
    # city_map.insert(city12)
    # city13 = City(120, 80)
    # city_map.insert(city13)
    # city14 = City(180, 60)
    # city_map.insert(city14)
    # city15 = City(20, 40)
    # city_map.insert(city15)
    # city16 = City(100, 40)
    # city_map.insert(city16)
    # city17 = City(200, 40)
    # city_map.insert(city17)
    # city18 = City(20, 20)
    # city_map.insert(city18)
    # city19 = City(60, 20)
    # city_map.insert(city19)
    # city20 = City(160, 20)
    # city_map.insert(city20)

    for i in range(100):
        x, y = random.random() * 500, random.random() * 500
        city_map.insert(City(x, y))

    ga = GA(city_map, population_size=70, no_generations=150, mutation_rate=0.15,\
            elitism_rate=0.15, tournament_size=15, print_cost_per_gen=5)


    fittess_tour = ga.genetic_algorithm()

    print("{:*^120}".format("Final fittess tour"))        
    print("Tour:", fittess_tour)
    print("Cost:", fittess_tour.get_cost())
    # print("Final fitness:", fittess_tour.get_fitness())

    
if __name__ == "__main__":
    main()

