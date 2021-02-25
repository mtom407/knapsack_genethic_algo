import numpy as np

# =========================== FUNKCJE POMOCNICZE DO SELEKCJI | SELECETION HELPER FUNCTIONS ================================== #

# cumulative vector - could have been omitted but oh well
def aggregation(normalized_fitness):
    aggregated_fitness = np.zeros_like(normalized_fitness)
    for i in range(len(normalized_fitness)):
        aggregated_fitness[i] = np.sum(normalized_fitness[:i+1])
    return aggregated_fitness    
# =============================================================== #

# roulette wheel simulation 
def spin_the_wheel(aggregated_fitness):
    random_number = np.random.rand(1)
    for i, aggr in enumerate(aggregated_fitness):
        if i == 0 and random_number <= aggr:
            return i
        elif i > 0 and (aggregated_fitness[i-1] <= random_number and random_number < aggr):
            return i
        else:
            continue
# =============================================================== #

# ================================ FUNKCJE REALIZUJĄCE SELEKCJE | SELECTION FUNCTIONS ======================================= #
def roulette_selection(sorted_population, sorted_fitness):
    
    # obliczenie sumy dopasowań i normalizacja wektora dopasowań 
    #| computing the sum of fitness values and normalizing the fitness vector
    fitness_sum = np.sum(sorted_fitness)
    normalized_fitness = (sorted_fitness/fitness_sum)
    
    # wektor agregowanych wartosc dopasowań (reprezentacja koła ruletki)
    # aggregateg fitness vector (roulette wheel representation)
    aggregated_fitness = aggregation(normalized_fitness)
    
    # teraz wybierz osobników do nowej generacji
    # choosing individuals for the next generation
    next_gen_count = len(sorted_population)//2 
    next_gen_indices = np.zeros(next_gen_count, dtype = 'uint16')
    while(next_gen_count > 0):
        individual_pick = spin_the_wheel(aggregated_fitness)
        if sorted_fitness[individual_pick] == np.min(sorted_population):
            # ignore those that violate the constraints
            # pomijamy te, które nie spełniają ograniczeń
            continue
        else:
            next_gen_indices[next_gen_count-1] = individual_pick
            next_gen_count -= 1
    
    # stwórz nową generację | create the new generation
    next_generation = sorted_population[next_gen_indices]
    
    return next_generation

def rank_selection(sorted_population, sorted_fitness):
    
    # utworzenie wektora rang | rank vector
    ranks = np.arange(len(sorted_fitness), 0, -1)
    
    # obliczenie sumy rang i normalizacja wektora rang 
    # sum of all ranks and normalization
    ranks_sum = np.sum(ranks)
    normalized_ranks = (ranks/ranks_sum)
    
    # wektor agregowanych wartości rang
    # aggregated rank vector
    aggregated_ranks = aggregation(normalized_ranks)
    
    # teraz wybierz osobników do nowej generacji
    # now pick new individuals for the new gen
    next_gen_count = len(sorted_population)//2 
    next_gen_indices = np.zeros(next_gen_count, dtype = 'uint16')
    while(next_gen_count > 0):
        individual_pick = spin_the_wheel(aggregated_ranks)
        if sorted_fitness[individual_pick] == np.min(sorted_population):
            continue
        else:
            next_gen_indices[next_gen_count-1] = individual_pick
            next_gen_count -= 1

    next_generation = sorted_population[next_gen_indices]

    return next_generation

# funkcja selekcji turniejowej | tournament selection function
# k to wielkość grupy turniejowej, p to prawdopodobieństwo wybrania "najlepszego"
# k --> tournament group size | p --> probability of choosing the best fitted individual
def tournament_selection(sorted_population, sorted_fitness):
    k = 12
    sorted_without_zeros = sorted_fitness[sorted_fitness != 0]    
    next_gen_indices = []
    while(len(next_gen_indices)<len(sorted_population)//2):
            
        # wybieramy z jednakowym prawdopodobieństwem k elementów
        # uniformly choose k elements
        selected_to_tour = np.random.choice(sorted_without_zeros, k)
        
        # wybieramy "wygranego" turnieju i bierzemy jego indeks
        # choosing the "winner" and taking his index
        strongest = selected_to_tour.max()
        individual_index = sorted_fitness[sorted_fitness.values == strongest].index
        next_gen_indices.append(individual_index[0])
        
    next_gen_pop = sorted_population[next_gen_indices]
        
    return next_gen_pop

def elite_selection(sorted_population, sorted_fitness):
    
    # picking just the best
    selection_range = sorted_population.shape[0]//2
    next_gen_pop = sorted_population[:selection_range]
    return next_gen_pop
