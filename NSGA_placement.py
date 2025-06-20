import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from placement_metrics import calculate_access_delay, calculate_workload_balance

# ======================
# PROBLEM CONFIGURATION
# ======================
VECTOR_DIM = 4      # Dimension of input vector (n)
LOW_BOUND = 0       # Minimum value for vector components
UP_BOUND = 1        # Maximum value for vector components
POP_SIZE = 100      # Population size
NGEN = 10           # Number of generations
CXPB = 0.9          # Crossover probability
MUTPB = 0.1         # Mutation probability

# ======================
# DEAP FRAMEWORK SETUP
# ======================
# Create fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))  # Minimize both objectives
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator for each vector component
toolbox.register("attr_float", random.uniform, LOW_BOUND, UP_BOUND)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual,
                toolbox.attr_float, n=VECTOR_DIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ======================
# OBJECTIVE FUNCTION
# ======================
# before the evaluate function we need to load the data frame first
df = \
    pd.read_csv('./shanghai_dataset/block_counts.csv')

def evaluate(vector):
    # here is the vector of this shit
    # result = calculate_workload_balance(x.reshape(int(self.input_dim/2), 2), self.data)
    # there should be the size of this vector.
    # but that is given at the beginning of thi fucking file
    # and then what happened?

    #
    # reshape the input vector?
    # first you should know about how reshape work
    v = np.array(vector)
    v = v.reshape(-1,2)
    f1 = calculate_workload_balance(v, df)
    f2 = calculate_access_delay(v, df)
    return f1, f2

# ======================
# GENETIC OPERATORS
# ======================
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0,
                low=LOW_BOUND, up=UP_BOUND)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0,
                low=LOW_BOUND, up=UP_BOUND, indpb=1.0/VECTOR_DIM)
toolbox.register("select", tools.selNSGA2)

# ======================
# OPTIMIZATION FUNCTION
# ======================
def run_optimization():
    random.seed(42)

    # Create initial population
    population = toolbox.population(n=POP_SIZE)

    # Set up statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("avg", np.mean, axis=0)

    # Run NSGA-II
    result, logbook = algorithms.eaMuPlusLambda(
        population, toolbox,
        mu=POP_SIZE,
        lambda_=POP_SIZE,
        cxpb=CXPB,
        mutpb=MUTPB,
        ngen=NGEN,
        stats=stats,
        verbose=True
    )


    # Extract Pareto front
    pareto_front = tools.sortNondominated(result, len(result), first_front_only=True)[0]
    df = {
        "solutions": [ind[0] for ind in pareto_front],
        "obj1values": [ind.fitness.values[0] for ind in pareto_front],
        "obj2values": [ind.fitness.values[1] for ind in pareto_front]
    }
    df = pd.DataFrame(df)
    df.to_csv('NSGA_test1.csv', index=False)

    # ======================
    # VISUALIZATION
    # ======================
    plt.figure(figsize=(10, 6))
    plt.scatter(
        [ind.fitness.values[0] for ind in pareto_front],
        [ind.fitness.values[1] for ind in pareto_front],
        c="red", label="Pareto Front"
    )
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title(f"Pareto Front for {VECTOR_DIM}-D Vector Optimization")
    plt.legend()
    plt.grid(True)
    plt.savefig("fig/NSGA_placement.png")
    # don't show for now
    # plt.show()

    # ======================
    # RESULTS ANALYSIS
    # ======================
    print("\nTop Pareto solutions:")
    for i, ind in enumerate(pareto_front[:5]):
        print(f"Solution {i+1}:")
        print(f"  Vector: {[round(x, 4) for x in ind]}")
        print(f"  Objectives: {ind.fitness.values}\n")

    return pareto_front, logbook

# ======================
# EXECUTION
# ======================
if __name__ == "__main__":
    pareto_front, logbook = run_optimization()
    front = np.array(pareto_front)
    np.save('placement_front1.npy', front)
    np.savetxt('NSGA_front1.csv', front, delimiter=',')
