import numpy as np
import math
import random
from operator import itemgetter, attrgetter
from bit_manager import Number, NumberArray
import argparse
from matplotlib import pyplot as plt
import json_tricks

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
        self.map_obstacle = [] # list of NumberArray (1 if obstacle and 0 if non-obstacle)

    def insert(self, city: City):
        """Insert new city into map (list)"""
        self.list_cities.append(city)

    def get(self) -> list:
        return self.list_cities

    def get_map(self) -> list:
        """
        Return map_obstacle (list of NumberArray)
        """
        return self.map_obstacle

    def get_map_numpy(self) -> np.array:
        n = self.size()
        tmp = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            # print(len(tmp[i,:]))
            tmp[i,:] = self.map_obstacle[i].get_list()
        return tmp

    def get_city(self, i) -> City:
        return self.list_cities[i]

    def size(self):
        return len(self.list_cities)

    def init_list_cities(self, n: int, boundary: tuple):
        """Initialize list_cities with random location which values are range from boundary tuple (max_x, max_y)
        
        Args:
            n: number of cities
            boundary: a tuple of range value for x and y (max_x, max_y)
        """
        max_x, max_y = boundary
        for i in range(n):
            x, y = random.random() * max_x, random.random() * max_y
            self.list_cities.append(City(x, y))

    def init_map(self, obstacle_rate=0.9):
        """
        Initial map_obstacle (mask matrix) contains 0/1 values

        Args:
            obstacle_rate: the probability of obstacle
        """
        n = self.size()

        map_obstacles = [] # np.zeros((n, n)) # 1: obstacle, 0: non-obstacle
    
        for i in range(n):
            # We only need 2 bit to encode 1/0 for each element of NumberArray
            row = NumberArray(2, n)
            for j in range(n):
                if i == j:
                    # map_obstacles[i][j] = 0
                    row[j] = 0
                elif i > j:
                    # map_obstacles[i][j] = map_obstacles[j][i]
                    row[j] = map_obstacles[j][i]
                else:
                    # map_obstacles[i][j] = 1 if random.random() > 0.9 else 0
                    row[j] = 1 if random.random() > obstacle_rate else 0
            map_obstacles.append(row)

        self.map_obstacle = map_obstacles

    def set_map(self, map_obstacle: np.array):
        n = self.size()

        tmp = [] # np.zeros((n, n)) # 1: obstacle, 0: non-obstacle
    
        for i in range(n):
            # We only need 2 bit to encode 1/0 for each element of NumberArray
            row = NumberArray(2, n)
            for j in range(n):
                row[j] = int(map_obstacle[i][j])
            tmp.append(row)

        self.map_obstacle = tmp


    def is_feasible_tour(self, tour_ids) -> bool:
        for i in range(1, len(tour_ids)):
            # If exist obstacle, return False
            if self.map_obstacle[tour_ids[i-1]][tour_ids[i]] == 1:
                return False
        return True

    def is_movable(self, i, j) -> bool:
        if self.map_obstacle[i][j] == 1:
            return False
        return True

    def get_movable_city_ids(self, i) -> list:
        """
        Get movable city indices of i-th city
        """
        list_movable_ids = []
        for j in range(self.size()):
            if i == j:
                continue
            if self.map_obstacle[i][j] == 0:
                list_movable_ids.append(j)

        return list_movable_ids

    def align_tour_ids(self, tour_ids: list) -> list:
        """
        Align the tour_ids to satisfy the constraint
        """
        n = self.size()
        
        # levels is list of number of movable cities for each city
        levels = np.zeros(len(tour_ids), dtype=np.uint8)
        for i in tour_ids:
            levels[i] = len(self.get_movable_city_ids(i))
        # print(levels)

        for i in range(1, n):
            # Find all posible city indices that (tour_ids[i-1])-th city are linked to
            list_ids = self.get_movable_city_ids(tour_ids[i-1])
            
            # Check if (tour_ids[i-1])-th city and (tour_ids[i])-th city are not linked
            if self.is_movable(tour_ids[i - 1], tour_ids[i]) == False:
                # print(tour_ids[i - 1], tour_ids[i])
                # print(list_ids)
                sub_list_ids = []
                # Loop over posible city indices to filter out connectable city at current time
                for j in list_ids:
                    # Make sure j is not in previous part of tour_ids
                    # Because i just want to change only the after part of tour_ids which is counted from i position
                    if j not in tour_ids[0:i]: 
                        sub_list_ids.append(j)
                # print(sub_list_ids)

                # If sub_list_ids is empty, we drop this tour
                if len(sub_list_ids) == 0:
                    return None

                # Find argmin of levels
                j_min_in_slice = np.argmin(levels[sub_list_ids])
                j_min = sub_list_ids[j_min_in_slice]
                # print(j_min)    
                
                idx_of_j = tour_ids.index(j_min) # find position of j_min value in tour_ids
                # swap two postions
                tour_ids[i], tour_ids[idx_of_j] = j_min, tour_ids[i]

                # print(tour_ids)
            
            # Set level of (tour_ids[i-1])-th city is equal 0 when it satisfied the constraint
            levels[tour_ids[i-1]] = 0
            # Update levels of cities has connect with (tour_ids[i-1])-th city:
            for j in list_ids:
                # Avoid levels of city is already satisfied
                if levels[j] > 0:
                    levels[j] -= 1
            # print(levels)

        return tour_ids

    def plot_map(self):
        """Plot map_obstacle"""
        n = self.size()
        tmp = np.zeros((n, n))
        for i in range(n):
            # print(len(tmp[i,:]))
            tmp[i,:] = self.map_obstacle[i].get_list()
        
        # plotting map_obstacle 
        plt.figure()
        plt.title("Map obstacle (n = " + str(n) + " cities)") 
        plt.imshow(1-tmp, cmap='gray')

    def write_json(self, filename):
        data = {}
        data["map_obstacle"] = self.get_map_numpy()
        data["list_cities"] = self.get()
        print(type(data["map_obstacle"]))
        print(type(data["list_cities"]))

        # print(json_tricks.dumps(data, indent=4))

        with open(filename, "w") as json_file:
            json_tricks.dump(data, fp=json_file, indent=4)

    def read_json(self, filename):
        with open(filename, "r") as json_file:
            data = json_tricks.load(fp=json_file)
        self.list_cities = data["list_cities"]
        self.set_map(data["map_obstacle"])
            



class Tour:
    """Tour is a individual (aka chromosome) which is one satisfied solution"""
    def __init__(self, tour_size: int, tour_ids: "list/NumberArray"=None, city_map: Map=None):
        self.tour_size = tour_size # The number of cities in tour
        self.no_bits_per_index = math.ceil(math.log2(self.tour_size)) + 1 # include 1 sign bit
        self.tour_ids = NumberArray(self.no_bits_per_index, self.tour_size) # NumberArray of city indices
        if tour_ids == None:
            self.createNewTour(city_map)
        elif isinstance(tour_ids, list):
            self.tour_ids[:] = tour_ids
        elif isinstance(tour_ids, NumberArray):
            self.tour_ids = tour_ids

        self.fitness = self.compute_fitness(city_map.get())

    def createNewTour(self, city_map: Map):
        """Create new tour by sampling randomly map without replacement"""
        n = self.tour_size
        indices = list(range(n))
        flag = True
        while True:
            if flag:
                sampled_indices = random.sample(indices, n)
            if city_map.is_feasible_tour(sampled_indices): # If correct
                self.tour_ids[:] = sampled_indices
                break
            else: # If sampled_indices has unlinked city
                sampled_indices = city_map.align_tour_ids(sampled_indices)
                # If align fail, we pass it
                if sampled_indices == None:
                    continue
                
        # for i in range(n):
        #     index = random.randint(0, n-1)
        #     if i == 0:
        #         s
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

    def get_tour_ids(self):
        return [id for id in self.tour_ids]

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
            new_tour = Tour(tour_size=n, city_map=self.map)
            self.population.append(new_tour)

        self.sortPopulation()
        return self.population

    def samplePopulation(self, k) -> "Population":
        """Sample population with replacement"""
        sampledPopulation = Population(self.population_size, self.map)
        sampledPopulation.population = random.choices(self.population, k=k)
        sampledPopulation.sortPopulation()
        return sampledPopulation

    def get_fittess_tour(self) -> Tour:
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

            
        return Tour(self.map.size(), child, self.map)
    
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

            ## This stategies are slower
            # # Apply alignment if child is not valid
            # child_tour_ids = child.get_tour_ids()
            # if self.map.is_feasible_tour(child_tour_ids) == False:
            #     child_tour_ids = self.map.align_tour_ids(child_tour_ids)

            #     # If child_tour_ids is valid, add it into new_polulatio
            #     if child_tour_ids != None and self.map.is_feasible_tour(child_tour_ids):
            #         child = Tour(self.map.size(), child_tour_ids, self.map)
            #         new_population.insert(child)
            #     else:
            #         # Else, we select one individual with best score from sub-sample of self.polulation
            #         new_population.insert(self.selection())
            # else:
            #     # Add child into new_population
            #     new_population.insert(child)

            child_tour_ids = child.get_tour_ids()
            # Check if child is not valid
            if self.map.is_feasible_tour(child_tour_ids) == False:
                # we select one individual with best score from sub-sample of self.polulation
                new_population.insert(self.selection())
            else:
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


    def genetic_algorithm(self) -> Tour:
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

    def plot_result(self, fittess_tour):
        plt.figure()
        # Get tour_ids
        tour_ids = fittess_tour.get_tour_ids()
        n = len(tour_ids)
        # plotting result_map
        plt.title("Result map (n = " + str(n) + " cities)")

        list_cities = self.map.get()
        # xs, ys store cities' location
        xs = []
        ys = []
        # Loop over
        for i in tour_ids:
            x, y = list_cities[i].x, list_cities[i].y
            xs.append(x)
            ys.append(y)
            plt.text(x, y, str(i), fontsize=12)

        # Append start point as end location
        xs.append(list_cities[tour_ids[0]].x)
        ys.append(list_cities[tour_ids[0]].y)

        # Plot middle points and also start point
        plt.scatter(xs, ys, c='c')
        plt.scatter(xs[0], ys[0], c="r", s=200, label="Start point") # Plot Start point
        plt.plot(xs, ys) # Plot line
        plt.legend()
        plt.show()
    

def ParseArgs():
    parser = argparse.ArgumentParser(description="Travelling Salesman Problem commandline")
    parser.add_argument('-f', '--input_file', type=str,\
         help="Filename of inputs (Location and Map)")
    parser.add_argument("-psize", "--pop_size", type=int, default=50, help="Population size")
    parser.add_argument("-no_gens", "--no_generations", type=int, default=100,\
         help="Number of generations to process")
    parser.add_argument("-mu_rate", "--mutation_rate", type=float, default=0.15,\
         help="Probability of mutation")
    parser.add_argument("-elit_rate", "--elitism_rate", type=float, default=0.15,\
         help="The rate of number of best individuals to keep in next generation")
    parser.add_argument("-print", "--print_cost_per_gen", type=int, default=5,\
         help="Print cost per number of generations")
    parser.add_argument("-ts", "--tournament_size", type=int, default=15,\
         help="Tournament size to select one individual from sub-sample of particular population")
    parser.add_argument("-seed", "--seed", type=int, default=174,\
         help="Random seed")
    parser.add_argument("-plot_res", "--plot_result", action="store_true",\
         help="Plot result map")
    parser.add_argument("-plot_map", "--plot_map_obstacle", action="store_true",\
         help="Plot map obstacle")
    
    args = parser.parse_args()
    return args

def main(args):

    filename = args.input_file
    # Set seed
    random.seed(args.seed)
    # Init map
    city_map = Map()

    # -------------------------- Gennerate data -------------------------------------
    # # Write file json the data
    # n = 100
    # # Init city location randomly
    # city_map.init_list_cities(n=n, boundary=(500, 500))
    # # Init map_obstacle
    # city_map.init_map(obstacle_rate=0.97)
    # # city_map.plot_map()
    # # plt.show()
    # # city_map.set_map(np.zeros((n, n), dtype=np.uint8))
    # # city_map.write_json("input3.json")

    # Read file json of input data
    city_map.read_json(filename)
    if args.plot_map_obstacle:
        city_map.plot_map()

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
        
    ga = GA(city_map, population_size=args.pop_size, no_generations=args.no_generations, mutation_rate=args.mutation_rate,\
            elitism_rate=args.elitism_rate, tournament_size=args.tournament_size, print_cost_per_gen=args.print_cost_per_gen)

    fittess_tour = ga.genetic_algorithm()

    print("{:*^120}".format("Final fittess tour"))        
    print("Tour:", fittess_tour)
    print("Cost:", fittess_tour.get_cost())
    # print("Final fitness:", fittess_tour.get_fitness())

    if args.plot_result:
        ga.plot_result(fittess_tour)

    plt.show()

    
if __name__ == "__main__":
    main(ParseArgs())

