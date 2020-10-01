def fitness_function(gene):
    # Just Modify this for fitness function change. Index of 'gene' is same as the "BOUNDS" value in input.json file.
    # gene[0] refers to first set of [upper, lower bound] and so on.
    return gene[0] * gene[0] - gene[1] * gene[0] + gene[1] * gene[2]
