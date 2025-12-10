from docplex.mp.model import Model
import time
import sys
import csv
import os

TIMEOUT = 600

def graph_labeling(n,k,UB,edges):
    mdl = Model(name = "graph_labeling")

    x = {i: mdl.integer_var(lb = 1, ub = UB, name = f"x_{i}") for i in range(n)}

    z = mdl.integer_var(lb = n, ub = UB, name = "z")

    for i in range(n):
        for j in range(i + 1, n):
            mdl.add_constraint(x[i] != x[j])

    for (u,v) in edges:
        mdl.add_constraint(mdl.abs(x[u] - x[v]) >= k)

    for i in range(n):
        mdl.add_constraint(x[i] <= z)

    mdl.minimize(z)
    
    start = time.time()
    sol = mdl.solve(log_output= False, time_limit=TIMEOUT)
    end = time.time()

    if sol:
        span = sol[z]
        status = "Optimal" if sol.solve_status == "optimal" else "Feasible"
    else:
        span = mdl.objective_bound
        status = "TO"
    
    clauses = len(list(mdl.iter_constraints()))
    var_count = len(list(mdl.iter_variables()))
    print("Var: " + str(var_count))
    print("Clauses:" + str(clauses))
    print("Span is ", span)
    return UB, span, clauses, var_count, status, f"{end - start:.2f}"

def main():
    file_path = sys.argv[1]

    with open(file_path, "r") as f:
        n,m,k,UB = map(int, f.readline().split())
        edges = [tuple(map(int, f.readline().split())) for _ in range(m)]
    
    start = time.time()
    graph_labeling(n,k,UB, edges)
    print("time:", time.time() - start)

if __name__ == "__main__":
    main()

