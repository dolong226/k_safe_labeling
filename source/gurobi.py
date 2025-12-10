from typing import List, Tuple
from gurobipy import Model, GRB
import time
import sys

def graph_labeling(n, k, edges,UB, TIMEOUT=600):
    mdl = Model("graph_labeling")
    mdl.Params.TimeLimit = TIMEOUT

    labels = range(1, UB+1)
    y = mdl.addVars(range(n), labels, vtype=GRB.BINARY, name="y")

    # exactly one per vertex
    mdl.addConstrs((y.sum(v, '*') == 1 for v in range(n)), name="assign")

    # all differtent per label
    mdl.addConstrs((y.sum('*', l) <= 1 for l in labels), name="label_once")

    for (u, v) in edges:
        for l1 in labels:
            low = max(1, l1 - (k - 1))
            high = min(UB, l1 + (k - 1))
            for l2 in range(low, high + 1):
                mdl.addConstr(y[u, l1] + y[v, l2] <= 1)

    z = mdl.addVar(lb=n, ub=UB, vtype=GRB.INTEGER, name="z")
    for v in range(n):
        mdl.addConstr(z >= sum(l * y[v, l] for l in labels))

    mdl.setObjective(z, GRB.MINIMIZE)
    mdl.optimize()

    print(f"Variables: {mdl.NumVars}")
    print(f"Clauses: {len(mdl.getConstrs())}")

    if mdl.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        sol = [None]*n
        for v in range(n):
            for l in labels:
                if y[v, l].X > 0.5:
                    sol[v] = l
                    break
        print("labels:", sol)
        print("span =", z.X)
    else:
        print("No solution found")

def main():
    file_path = sys.argv[1]

    with open(file_path, "r") as f:
        n,m,k,UB = map(int, f.readline().split())
        edges = [tuple(map(int, f.readline().split())) for _ in range(m)]
    
    start = time.time()
    graph_labeling(n,k,edges, UB)
    print("time:", time.time() - start)

if __name__ == "__main__":
    main()