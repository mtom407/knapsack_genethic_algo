import numpy as np
import pandas as pd
import sys

def initialize_problem(N_passengers):
    
    #np.random.seed(2137)
    # wygeneruj wektor aI | amount vector generation
    a_vec = np.random.randint(low = 3, high = 8, size = N_passengers, dtype = 'uint16')

    # wygeneruj wektor vI | value vector generation
    v_vec = np.random.randint(low = 60, high = 100, size = N_passengers, dtype = 'uint16')
    
    # stworz dataframe | dataframe creation
    df = pd.DataFrame({'id_passenger': list(range(1, N_passengers+1)), 'amount': a_vec, 'value': v_vec})
    
#     print("Złodziej ma {} gramów złota.".format(X))
#     print("W pociągu jest {} pasażerów.".format(N))
#     print("Wektor wartości a: {}".format(a_vec))
#     print("Wektor wartości v: {}".format(v_vec))
    
    return df

    # ============================================= FUNKCJE | GENERAL FUNCTIONS  ================================================ #

# losowa początkowa populacja | initial random population
def generate_initial_population(pSize, N):
    
    population_array = np.random.randint(low = 0, high = 2, size = (pSize, N))
    
    return population_array

# funkcja do obliczenia dopasowania populacji | function for population fitness computation 
def fitness(population, values, weights, thief_amount):
    
    
    if len(population.shape) > 1:
        population_size, indi_length  = population.shape
    else:
        population_size = len(population)
    
    knapsack_weight_vector = np.matmul(population, weights)
    
    # binary vector to get rid of solutions that violate the constraints
    constraint_violation = knapsack_weight_vector < thief_amount
    # penalty
    #constraint_violation[constraint_violation==0] = 0.00001

    fitness_vector = np.matmul(population, values)
    fitness_vector = fitness_vector*constraint_violation
      
    return fitness_vector 

# funkcja krzyżująca | crossover function
def crossover(parent_a, parent_b, crossover_rate):
    
    # making sure that the parent vectors are of the same length
    assert len(parent_a) == len(parent_b), "Passed individuals are not of the same size"
    
    parent_len = len(parent_a)
    
    # check if the crossover is to be done and do it if so
    do_crossover = np.random.rand() <= crossover_rate
    if do_crossover:
        
        crossover_point = np.random.randint(low = 1, high = parent_len)
        # shuffle the parent vectors around the generated point
        child_a = np.hstack((parent_a[:crossover_point], parent_b[crossover_point:]))
        child_b = np.hstack((parent_b[:crossover_point], parent_a[crossover_point:]))

        return child_a, child_b
    else:
        return parent_a, parent_b

# funkcja mutująca | mutation function
def mutate(individual, p):
    
    # make sure the probability is correct
    assert p > 0 and p < 1, "Mutation probability parameter must be in (0, 1) range"
    
    indi_len = len(individual)
    
    # probability and mutation position vectors
    p_vector = np.random.rand(indi_len)
    mutation_idx = np.where((p_vector <= p) == 1)[0]
    
    # mutation implemented as negation on values under drawn indices
    for idx in mutation_idx:
        individual[idx] = not(individual[idx]) 
        
    return individual


def evolve(selected_population, mutation_rate, crossover_rate):
    
    # REKOMBINACJA | CROSSOVER
    # ile razy | how many crossovers?
    num_crossovers = len(selected_population)
    
    # gdzie zapisać | saving here
    individual_length = selected_population.shape[1]
    evolved_population = np.zeros((2*num_crossovers, individual_length))
    
    # zapełnij populację | populate 
    for cross_id in range(0, num_crossovers, 2):
        
        # oblicz miejsce na rodziców | compute parent spots
        parent_spot = cross_id + selected_population.shape[0] 
        
        # CROSSOVER
        
        # wybierz losowo rodziców | randomly choose the parents
        first_parent, second_parent = selected_population[np.random.randint(0, num_crossovers)], selected_population[np.random.randint(0, num_crossovers)]
        # stwórz dzieci | create children
        first_child, second_child = crossover(first_parent, second_parent, crossover_rate)
        
        # MUTACJA | MUTATION
        
        first_child = mutate(first_child, mutation_rate)
        second_child = mutate(second_child, mutation_rate)
        
        # ZAPIS | SAVING
        
        # zapisz dzieci | saving the children
        evolved_population[cross_id], evolved_population[cross_id+1] = first_child, second_child
        # zapisz rodziców | saving the parents
        evolved_population[parent_spot], evolved_population[parent_spot+1] = first_parent, second_parent
    
    
    return evolved_population


# BINARNY ALGORYTM GENETYCZNY - GŁÓWNA PĘTLA | BINARY GENETHIC ALGORITHM - MAIN LOOP

def knAGsack(df, capacity, pSize, max_iterations_no_change, selection_algorithm, crossover_rate, mutation_rate):

    # making sure population size is an even number
    assert (pSize//2)%2 == 0, "Population doesn't meet implementation requirements"
    
    # liczba iteracji | no of iterations
    iterations = 0

    # values & weights
    values = df['value']
    weights = df['amount']

    # populacja poczatkowa | initial population
    population = generate_initial_population(pSize,len(df))

    # listy do zbierania wyników | lists for saving
    fitness_history = []
    best_individuals = []
    min_vec = []
    max_vec = []
    mean_vec = []
    best_yet_vec = []

    remaining_time = max_iterations_no_change
    while(remaining_time):

        
        #print("Generation: {}".format(iterations))
        #sys.stdout.write("Generation: {}".format(iterations))
        remaining_time -= 1

        # kalkulacja dopasowania populacji | calculatiing the fitness of the actual population
        population_fitness = fitness(population, values, weights, capacity)
        fitness_sum = np.sum(population_fitness)


        # monitoruj najlepszy dotychczas | monitoring the best set of solutions yet
        this_gen_best = np.max(population_fitness)
        if iterations == 0:
            best_yet = this_gen_best
        else:

            # zapis do list ze statystykami | saing the stats
            max_vec.append(this_gen_best)
            #try:
            min_vec.append(np.min(population_fitness[population_fitness != 0]))
            #except:
                #print("The entire population violated the boundaries.")
                #min_vec.append(0)
            mean_vec.append(np.mean(population_fitness[population_fitness != 0]))

            if this_gen_best >= best_yet:
                best_yet = this_gen_best
                #best_yet_vec = best_yet
                best_individual = population[population_fitness==best_yet]
                best_individuals.append(best_individual)
                remaining_time = max_iterations_no_change

        best_yet_vec.append(best_yet)
        iterations += 1

        # zebranie wyników z iteracji | saving history
        fitness_history.append(best_yet)

        # sortowanie populacji według dopasowania | sorting by fitness scores
        fit_series = pd.Series(data = population_fitness)
        sorted_fitness = fit_series.sort_values(ascending = False)
        sorted_indices = sorted_fitness.keys()
        sorted_population = population[sorted_indices]

        # selekcja | select
        selected_population = selection_algorithm(sorted_population, sorted_fitness)

        # ewoluuj | evolve
        population = evolve(selected_population, mutation_rate, crossover_rate)

        #print("Best fitness yet: {} || Best fitness this gen: {}".format(best_yet, this_gen_best))
    
    sold_gold = np.matmul(best_individuals[-1], weights)
    remaining_gold = capacity - sold_gold
    profit = np.matmul(best_individuals[-1], values)
    return (profit, remaining_gold, iterations, (max_vec, mean_vec, min_vec, best_yet_vec))