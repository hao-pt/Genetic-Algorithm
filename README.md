# Genetic-Algorithm

## Install requirements
```
pip install -r requirements.txt
```

## Manual instruction
### Travelling Salesman Problem
- Get help
```
python tsp.py -h
```
- Result on commandline
```Console
usage: tsp.py [-h] [-f INPUT_FILE] [-psize POP_SIZE] [-no_gens NO_GENERATIONS]
              [-mu_rate MUTATION_RATE] [-elit_rate ELITISM_RATE]
              [-print PRINT_COST_PER_GEN] [-ts TOURNAMENT_SIZE] [-seed SEED]
              [-plot_res] [-plot_map] [-genn_data] [-bound BOUNDARY BOUNDARY]
              [-fo OUTPUT_FILE] [-n NO_CITIES] [-ob_rate OBSTACLE_RATE]

Travelling Salesman Problem commandline

optional arguments:
  -h, --help            show this help message and exit
  -f INPUT_FILE, --input_file INPUT_FILE
                        Filename of json file inputs (Location and Map)
  -psize POP_SIZE, --pop_size POP_SIZE
                        Population size
  -no_gens NO_GENERATIONS, --no_generations NO_GENERATIONS
                        Number of generations to process
  -mu_rate MUTATION_RATE, --mutation_rate MUTATION_RATE
                        Probability of mutation
  -elit_rate ELITISM_RATE, --elitism_rate ELITISM_RATE
                        The rate of number of best individuals to keep in next
                        generation
  -print PRINT_COST_PER_GEN, --print_cost_per_gen PRINT_COST_PER_GEN
                        Print cost per number of generations
  -ts TOURNAMENT_SIZE, --tournament_size TOURNAMENT_SIZE
                        Tournament size to select one individual from sub-
                        sample of particular population
  -seed SEED, --seed SEED
                        Random seed
  -plot_res, --plot_result
                        Plot result map
  -plot_map, --plot_map_obstacle
                        Plot map obstacle
  -genn_data, --gennerate_data
                        Run gennerate dataset
  -bound BOUNDARY BOUNDARY, --boundary BOUNDARY BOUNDARY
                        Boundary (max_x, max_y) of x and y
  -fo OUTPUT_FILE, --output_file OUTPUT_FILE
                        Filename of json file to write out
  -n NO_CITIES, --no_cities NO_CITIES
                        Number of cities to gennerate
  -ob_rate OBSTACLE_RATE, --obstacle_rate OBSTACLE_RATE
                        Probability of obstacle to gennerate
```

- Gennerate data
```
python tsp.py -fo input3.json  -genn_data -bound -500 500 -n 80\
              -ob_rate 0.98 -seed 3000 -plot_map
```
For example: we gennerate our data with random coordinates x and y which are ranged [-500.0, 500.0] for 80 cities. We use obstacle rate (0.98) to gennerate `map_obstacle` which contains only 0/1 values. Where 0: no obstacle and otherwise. I set seed equal 3000 for consistent random each run and turn `plot_map` flag on to plot `map_obstacle`.

- Run GA algorithm
```
python tsp.py -f input1.json  -psize 70 -no_gens 150 -mu_rate 0.15\
              -elit_rate 0.15 -ts 15 -plot_res -plot_map
```

Arguments:
- Input file `-f`: input1.json
- Population size `-psize`: 70
- Number of generation to run `-no_gens`: 150
- Mutation rate `-mu_rate`: 0.15
- The rate of elitism that we want to keep in next generation `-elit_rate`: 0.15
- Tournament size is used to select one fittess individual in sub-sample of population `-ts`: 15
- Plot the path result by turn `-plot_res` flag on.
- Plot `map_obstacle` by toggle `-plot_map`.


### Qudratic problem
- Get help
```
python quadratic_eq.py -h
```
- Result on commandline
```Console
usage: quadratic_eq.py [-h] [-coeff COEFFICIENTS COEFFICIENTS COEFFICIENTS]
                       [-bound BOUNDARY BOUNDARY] [-psize POP_SIZE]
                       [-no_gens NO_GENERATIONS] [-mu_rate MUTATION_RATE]
                       [-ma_rate MATING_RATE] [-elit_rate ELITISM_RATE]
                       [-print PRINT_SOLUTION_PER_GEN] [-seed SEED]

Quadratic equation commandline

optional arguments:
  -h, --help            show this help message and exit
  -coeff COEFFICIENTS COEFFICIENTS COEFFICIENTS, --coefficients COEFFICIENTS COEFFICIENTS COEFFICIENTS
                        Coefficients of the equation (a: the quadratic
                        coefficient, b: the linear coefficient and c: the
                        constant)
  -bound BOUNDARY BOUNDARY, --boundary BOUNDARY BOUNDARY
                        Boundary (domain) of problem
  -psize POP_SIZE, --pop_size POP_SIZE
                        Population size
  -no_gens NO_GENERATIONS, --no_generations NO_GENERATIONS
                        Number of generations to process
  -mu_rate MUTATION_RATE, --mutation_rate MUTATION_RATE
                        Probability of mutation
  -ma_rate MATING_RATE, --mating_rate MATING_RATE
                        Probability of mating (crossover)
  -elit_rate ELITISM_RATE, --elitism_rate ELITISM_RATE
                        The rate of number of best individuals to keep in next
                        generation
  -print PRINT_SOLUTION_PER_GEN, --print_solution_per_gen PRINT_SOLUTION_PER_GEN
                        Print solution per number of generations
  -seed SEED, --seed SEED
                        Random seed
```
- Run GA algorithm

  - Formulation of quadratic equation: `ax^2 + bx + c = 0`
  - Notice: my program just solve for interger "variable" `x`. Therefore, the boundary of `x` must be in integer boundary [-int, int].

```
python quadratic_eq.py -coeff 1 4 4 -bound -10 10 -psize 5 -no_gens 8\
                       -mu_rate 0.1 -ma_rate 0.5 -elit_rate 0.2 -print 5
```


- For example: our quadratic equation is x**2 + 4*x + 4 where x is in [-10, 10].

  - Arguments:
    - `-coeff`: Coefficents for quadratic equation such as a, b and c
    - `-bound`: Boundary of x is list of integer ([int, int])
    - Population size `-psize` is 5
    - Number of gennerations to run `-no_gens` is 8
    - Mutation rate `-mu_rate` is 0.1
    - Mating rate `-ma_rate` is 0.5
    - The rate of elitism that we want to keep in next generation `-elit_rate` is 0.2
    - `-print`: Print results per 5 generations
