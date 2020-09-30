def fitness_function(gene):
    #Just Modify this for fitness function change.
    return gene[0] ** gene[0] + gene[1] * gene[0] + gene[1] * gene[2]
