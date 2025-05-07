import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# Define the problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_float", random.uniform, -5, 5)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the evaluation function


def evaluate(individual):
    x = individual[0]
    f1 = x**2
    f2 = (x - 2)**2
    return f1, f2


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=-5, up=5)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=-5, up=5, indpb=0.2)
toolbox.register("select", tools.selNSGA2)


def main():
    random.seed(42)

    # Create an initial population
    population = toolbox.population(n=100)

    # Define the statistics to be collected
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", lambda x: min(x))
    stats.register("max", lambda x: max(x))

    # Run the NSGA-II algorithm
    result, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=100,
                                                cxpb=0.9, mutpb=0.1, ngen=50, stats=stats,
                                                verbose=True)

    # Extract the Pareto front
    pareto_front = tools.sortNondominated(result, len(result), first_front_only=True)[0]

    # Plot the Pareto front
    plt.scatter([ind.fitness.values[0] for ind in pareto_front],
                [ind.fitness.values[1] for ind in pareto_front],
                c="red", label="Pareto Front")
    plt.xlabel("f1(x)")
    plt.ylabel("f2(x)")
    plt.title("Pareto Front")
    plt.legend()
    plt.show()


def main_and_save(path):
    random.seed(42)

    # Create an initial population
    population = toolbox.population(n=100)

    # Define the statistics to be collected
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", lambda x: min(x))
    stats.register("max", lambda x: max(x))

    # Run the NSGA-II algorithm
    result, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=100,
                                                cxpb=0.9, mutpb=0.1, ngen=50, stats=stats,
                                                verbose=True)

    # Extract the Pareto front
    pareto_front = tools.sortNondominated(result, len(result), first_front_only=True)[0]

    # Plot the Pareto front
    plt.scatter([ind.fitness.values[0] for ind in pareto_front],
                [ind.fitness.values[1] for ind in pareto_front],
                c="red", label="Pareto Front")
    plt.xlabel("f1(x)")
    plt.ylabel("f2(x)")
    plt.title("Pareto Front")
    plt.legend()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    main()
