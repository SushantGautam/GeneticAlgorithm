import matplotlib.pyplot as plt # for beautiful plots

def plotData(statLog):
    plt.plot([e['Generation'] for e in statLog], [e['BestFit'] for e in statLog], label="Fitness Value")
    plt.xlabel('Generations')
    plt.ylabel('Fitness Value')
    plt.title('Generations vs Fitness Value')
    plt.legend()
    plt.show()
    pass