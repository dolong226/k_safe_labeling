import time
from docplex.cp.model import CpoModel
import sys
import os

TIMEOUT = 600
from docplex.cp.config import context

def graph_labeling(n, k, UB, edges):
    mdl = CpoModel(name="graph_labeling")

    x = [mdl.integer_var(1, UB, name=f"x_{i}") for i in range(n)]

    mdl.add(mdl.all_diff(x))

    # k-safe constraints
    for (u, v) in edges:
        mdl.add(mdl.abs(x[u] - x[v]) >= k)

    # minimize
    z = mdl.max(x)
    mdl.minimize(z)

    start = time.time()
    sol = mdl.solve(TimeLimit=TIMEOUT, LogVerbosity="Terse")
    end = time.time()

    if sol.is_solution():
        span = sol.get_objective_value()
        status = sol.get_solve_status()
    else:
        bound = sol.get_objective_bound()
        span = bound if bound is not None else n
        status = "TO"

    stats = mdl.get_statistics()
    clauses = stats.get_number_of_constraints()
    var_count = stats.get_number_of_variables()
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

