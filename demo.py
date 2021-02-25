import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from selection_functions import elite_selection, roulette_selection, rank_selection, tournament_selection
import general_ga as ga

N_passengers = 100
knapsack_capacity = 500

df = ga.initialize_problem(N_passengers)

selection_ting = elite_selection
pSize_ting = 100
cross_rate_ting = 0.95
mutation_rate = 1/(pSize_ting*2)
       
ga_profit, ga_remaining_grams, iterations, record_tuple = ga.knAGsack(df, capacity = knapsack_capacity, pSize = pSize_ting, 
                                                          max_iterations_no_change = 500, selection_algorithm = selection_ting, 
                                                          crossover_rate = cross_rate_ting, mutation_rate = mutation_rate) #1/(pSize_ting*2))

max_vec = record_tuple[0]
mean_vec = record_tuple[1]
min_vec = record_tuple[2]
best_yet_vec = record_tuple[3]

# CHYTRY ALGORYTM

# generuj dane    
greedy_df = df.copy() #ga.initialize_problem(N_passengers)

# ile gramow ma złodziej?
greedy_remaining_grams = knapsack_capacity
greedy_profit = 0

# nowa kolumna ktora przyda sie do dobierania 
greedy_df['ratio'] = greedy_df['value']/greedy_df['amount']
greedy_df.sort_values(by = "ratio", ascending = False, inplace = True)

picked_items = []

for index, row in greedy_df.iterrows():
    
    if greedy_remaining_grams - row['amount'] >= 0:
        greedy_remaining_grams -= row['amount']
        greedy_profit += row['value']
        picked_items.append(index)
    else:
        continue

print("Genetically enhanced thief made: {} gold.".format(ga_profit))
print("Remaining weight - GA: {} grams.".format(ga_remaining_grams))

print("Greedy thief made: {} gold.".format(greedy_profit))
print("Remaining weight - greed: {} grams.".format(greedy_remaining_grams))

show_graph = False
if show_graph:
    plt.figure(figsize = (20,10))
    plt.plot(min_vec)
    plt.plot(mean_vec)
    plt.plot(max_vec)
    plt.plot(best_yet_vec)
    plt.grid()
    plt.xlabel('Liczba iteracji')
    plt.ylabel('Zysk ze sprzedaży')
    plt.legend(['min', 'mean', 'max', 'best_yet'])
    plt.show()