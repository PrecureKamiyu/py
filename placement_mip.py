from mip import Model, xsum, maximize, BINARY, minimize

# I is a list for convenience
number_of_placed_locations = 5
number_of_possible_locations = 20
I = range(len(number_of_possible_locations))

# the string is only a name for displaying
m = Model("no real meaning")

# declare 20 variable
# BINARY means that this is a binary variable
# ranging from \{0,1\}
x = [m.add_var(var_type=BINARY) for _ in I]

m.objective = minimize(xsum(x[i] for i in I))

# adding restriction
m += xsum(x[i] for i in I) <= number_of_placed_locations
