def fitness_function(x):
    # Just Modify this for fitness function change. Index of 'x' is same as the "BOUNDS" value in input.json file.
    # x[0] refers to first set of [upper, lower bound] and so on.
    return \
        x[0] * x[0] - x[1] * x[0] + x[1] * x[2]
