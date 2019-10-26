import math
import random
from bit_manager import Number, NumberArray

random.seed(174)

class Population:
    """Population is a collection of posible solutions (individuals)"""
    def __init__(self, equation: 'lambda', population_size: int, boundary: tuple, \
                 no_bits_per_item: int, offset, initial=False):
        self.population_size = population_size
        self.low_bound, self.high_bound = boundary
        self.equation = equation # lambda expression
        self.no_bits_per_item = no_bits_per_item # Number of bits represent the binary encode of each solution (Include extra sign bit)
        self.population = NumberArray(self.no_bits_per_item, self.population_size)
        self.offset = offset
        if initial:
            self.initPopulation()
        
    def initPopulation(self) -> "Population":
        """Initialize first population"""
        for i in range(self.population_size):
            k = random.randint(self.low_bound, self.high_bound)
            self.population[i] = k# Add new individual

        # Sort
        self.sortPopulation()
        return self.population

    def compute_weights(self) -> list:
        """Compute weights (fitness) of each individuals"""
        weights = []
        for num in self.population:
            weights.append(self.high_bound - self.equation(num+self.offset))
        return weights

    def pick_indiv_randomly(self) -> int:
        """Pick a individual randomly based on weight set"""
        return random.choices(self.population, weights=self.compute_weights(), k=1)[0]

    def get_fittess_indiv(self) -> int:
        """Get fittest individual with min fitness"""
        return self.population[-1]
        
    def sortPopulation(self):
        """Sort population base on fitness in descending order"""
        tmp = [num + self.offset for num in self.population]
        sorted_list = sorted(tmp, key=self.equation, reverse=True)
        self.population[:] = [num - self.offset for num in sorted_list]
        

    def getPopulation(self) -> NumberArray:
        """Get population NumberArray"""
        return self.population

    def __setitem__(self, i, value):
        self.population[i] = value

    def __getitem__(self, i):
        return self.population[i]

    def size(self) -> int:
        return len(self.population)

    def __str__(self):
        string = ""
        for i, indiv in enumerate(self.population):
            string += "Index " + str(i) + " | x = " + str(indiv)
            string +=  " | fitness = " + str(self.equation(indiv+self.offset))
            string += "\n"

        return string

class Evolution:
    """Evoluation class implements genetic functions that are useful for algorithm.

        Evolve next generation includes:
            Selection
            Cross-over
            Mutate
    """
    def __init__(self, equation: "lambda", boundary: tuple, offset,\
                 population_size=4, mutation_rate=0.1, mating_rate=0.5, elitism_size=10,\
                 pop: Population=None, no_bits_per_item=8):
        self.equation = equation
        self.mutation_rate = mutation_rate # low probability to mutate
        self.mating_rate = mating_rate # Probability for mating (crossover)        
        self.population = pop
        self.population_size = population_size
        self.elitism_size = elitism_size # Number of best individuals in next generation
        self.low_bound, self.high_bound = boundary
        self.no_bits_per_item = no_bits_per_item
        self.offset = offset

    def setPop(self, pop):
        """Set new population and update population size"""
        self.population = pop
        self.population_size = pop.size()

    def getPop(self):
        return self.population

    def selection(self) -> Number:
        """Perform selection by choose randomly a individual from weighted probabilities of current pop"""
        indiv = self.population.pick_indiv_randomly()
        return Number(indiv, no_bits=self.no_bits_per_item)

    def cross_over(self, father: Number, mother: Number) -> Number:
        """Cross-over (breeding) between parents"""
        child = Number(data=0, no_bits=self.population.no_bits_per_item)
        
        # Randomly set genes of child belong to both parents based on mating_rate
        for i in range(len(child)):
            child[i] = father[i] if random.random() < self.mating_rate else mother[i]
            
        return child
    
    def mutatation(self, indiv: Number) -> Number:
        """Mutation function by flip bits based on mutating_rate"""
        flip_bit = lambda x: 0 if x == 1 else 1

        start = 0 if self.low_bound >= 0 else 1
        # Traverse through bits (skip the sign bit)
        for i in range(start, len(indiv)):
            if random.random() < self.mutation_rate:
                indiv[i] = flip_bit(indiv[i])
        
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
        new_population = Population(equation=self.equation, population_size=self.population_size, \
                                    boundary=(self.low_bound, self.high_bound), no_bits_per_item=self.no_bits_per_item, offset=self.offset)

        # Get current population
        current_population = self.population

        # Keep first best (elitism_size) of tour
        for i in range(self.population_size - 1, self.population_size - self.elitism_size - 1, -1):
            new_population[i] = current_population[i]
        
        # print(current_population)
        # print(new_population)

        for i in range(self.elitism_size, self.population_size):
            # Randomly select parents
            father = self.selection()
            mother = self.selection()
            
            # print(father.to_bin(), mother.to_bin())

            # Breeding (cross-over)
            child = self.cross_over(father, mother)
            # print(child.to_bin())

            # Mutatation respects to low probability (mutation_rate)
            child = self.mutatation(child)
            # print(child.to_bin())

            if self.low_bound <= child.to_int() <= self.high_bound:
                # Add child into new_population
                new_population[i] = child
            else:
                # Check if new_population is not enough population_size
                # We pick randomly
                new_population[i] = self.population.pick_indiv_randomly()
        
        # print(new_population)
        new_population.sortPopulation()
        # print(new_population)
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
    def __init__(self, equation, boundary: tuple, population_size=4,\
                 no_generations=100, mutation_rate=0.1, mating_rate=0.5,\
                 elitism_rate=0.2, print_solution_each_gen=5):
        """
        Init function

        Args:
            no_generations (int): the number of generations are considered (stopping condition)
            population_size (int): the size of population
            mutation_rate: the low probability for mutation
            mating_rate: the probability for mating (crossover)
            elitism_rate: the rate of best individuals to keep in next generation
            equation: lambda equation
            print_solution_each_gen: print solution each 5 generations
            boundary: the boundary (doman) of x
        """
        self.equation = equation
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mating_rate = mating_rate
        self.elitism_size = int(self.population_size * elitism_rate) # Number of best individuals in next generation
        self.no_generations = no_generations
        self.low_bound, self.high_bound = boundary
        self.offset = 0 # Translate the low_bound to zero
        if self.low_bound > 0:
            self.offset = self.low_bound
        elif self.low_bound < 0:
            self.offset = self.high_bound
        self.low_bound = self.low_bound - self.offset
        self.high_bound = self.high_bound - self.offset
        self.domain = self.high_bound - self.low_bound + 1


        self.print_solution_each_gen = print_solution_each_gen
        self.no_bits_per_item = math.ceil(math.log2(self.domain)) + 1 # Number of bits represent the binary encode of each solution (Include extra sign bit)
        # Init evol object
        self.evol = Evolution(equation=self.equation, boundary=(self.low_bound, self.high_bound),\
                               population_size=self.population_size, mutation_rate=self.mutation_rate,\
                               mating_rate=self.mating_rate, elitism_size=self.elitism_size, pop=None, no_bits_per_item=self.no_bits_per_item, offset=self.offset)

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
        """
        
        # Init first population
        pop = Population(self.equation, population_size=self.population_size, boundary=(self.low_bound, self.high_bound),\
                         no_bits_per_item=self.no_bits_per_item, offset=self.offset, initial=True)
        # print(pop)
        x = pop.get_fittess_indiv()
        print("{:*^120}".format("Initial fittess x"))
        print("x = {0}, fitness = {1}".format((x+self.offset), self.equation(x+self.offset)))

        # Set first pop
        self.evol.setPop(pop)
        
        for i in range(self.no_generations):
            # Evolve next generation
            pop = self.evol.evolve_generation()

            # Set new pop
            self.evol.setPop(pop)

            # Print solution
            if i % self.print_solution_each_gen == 0:
                x = pop.get_fittess_indiv()
                print("{:-^50}Generation {}{:-^50}".format("", i, ""))
                print("x = {}".format(x+self.offset), end="")
                print(", fitness = {0}".format(self.equation(x+self.offset)))

        return self.evol.getPop().get_fittess_indiv() + self.offset

def main():
    # equation = lambda x: x**2 - 10*x + 5
    equation = lambda x: x**2 + 4*x + 4
    boundary = (-10, 10)
    ga = GA(equation, boundary, population_size=5,\
       no_generations=8, mutation_rate=0.1, mating_rate=0.5,\
       elitism_rate=0.2, print_solution_each_gen=5)

    x = ga.genetic_algorithm()
    print("{:*^120}".format("Final fittes x"))        
    print("x = {}, fitness = {}".format(x, equation(x)))

if __name__ == "__main__":
    main()