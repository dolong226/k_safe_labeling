from typing import List, Tuple
import time
import sys
from multiprocessing import Process, Manager
from pysat.solvers import Glucose42
from pysat.formula import CNF


# n: vertex
# edges
# k
# UB: upper bound

TIMEOUT = 600

def var_id(v: int,l: int,UB: int) -> int:
    return v * UB + l

def control_var_id(s: int, n:int, UB:int) -> int:
    return n * UB + s


# def find_highest_degree_vertex(n: int, edges: List[Tuple[int,int]]):
#     best_v = 0
#     degree_count = [0] * n
#     for (u,v) in edges:
#         degree_count[u] += 1
#         degree_count[v] += 1

#     max_degree = degree_count[1]
#     best_vertex = 0
#     for v in range(1, n ):
#         if degree_count[v] > max_degree:
#             max_degree = degree_count[v]
#             best_vertex = v
#         elif degree_count[v] == max_degree:
#             if v < best_vertex:
#                 best_vertex = v

#     return best_vertex
 
def build_base_cnf(n: int, edges: List[Tuple[int, int]], k:int, UB:int) -> CNF:
    cnf = CNF()

    # # symmetry breaking
    # v0 = find_highest_degree_vertex(n, edges)
    # max_allowed = UB//2
    # for label in range(max_allowed + 1, UB + 1):
    #     cnf.append([-var_id(v0, label, UB)])

    # exactly one per vertex:
    for v in range(n):
        # at least one
        cnf.append([var_id(v,i,UB) for i in range(1,UB+1)])
        # at most one pairwise
        for a in range (1,UB + 1):
            for b in range (a+1, UB + 1):
                cnf.append([-var_id(v,a,UB), -var_id(v,b,UB)])

    # all different per label
    for l in range(1, UB+1):
        for u in range(n):
            for v in range(u+1, n):
                cnf.append([-var_id(u,l,UB), -var_id(v,l,UB)])
    
    # k-safe pairwise
    for (u,v) in edges:
        if u > v: u,v = v,u
        for l1 in range(1, UB + 1):
            low = max (1, l1 - (k - 1))
            high = min (UB, l1 + (k - 1))
            for l2 in range(low, high + 1):
                cnf.append([-var_id(u,l1,UB), - var_id(v,l2, UB)])
                
    # incremental sat
    for S in range (1, UB + 1):
        Cs = control_var_id(S,n,UB)
        for v in range(n):
            for l in range(S+1, UB + 1):
                cnf.append([-var_id(v,l,UB), Cs])
    return cnf
  
def solve_inc(n: int, edges:List[Tuple[int,int]], k:int, UB:int, shared, solver_cls = Glucose42):
    cnf = build_base_cnf(n,edges, k, UB)
    solver = solver_cls(bootstrap_with=cnf.clauses)

    print("Variables: ", solver.nof_vars())
    print("Clauses: ", solver.nof_clauses())
    
    best_S, best_model = None, None
    high = UB
    while high >= n:
        C = control_var_id(high, n, UB)
        sat = solver.solve(assumptions=[-C])

        if sat: 
            best_S = high
            best_model = solver.get_model()
            shared["best_S"] = best_S
            shared["best_model"] = best_model
            high = best_S - 1
        else: 
            break

    # decode labels
    labels = [-1] * n
    modelset = set(l for l in best_model if l > 0)
    for v in range(n):
        for l in range(1,UB + 1):
            if var_id(v,l,UB) in modelset:
                labels[v] = l
                break

    shared["labels"] = labels
    print ("Optimal UB =", best_S)
    for v in range(n):
        print(f"Vertex {v} -> label {labels[v]}")
    
    return best_S, labels

def main():
    file_path = sys.argv[1]
    with open(file_path, "r") as f:
        n,m,k,UB = map(int, f.readline().split())
        edges = [tuple(map(int, f.readline().split())) for _ in range(m)]

    start = time.perf_counter()

    with Manager() as manager:
        shared = manager.dict()
        p = Process(target=solve_inc, args= (n,edges,k,UB, shared, Glucose42))
        # Process(target=solve_inc, args= (n,edges,k,UB, shared, Cadical195))

        p.start()
        p.join(timeout = TIMEOUT)

        if p.is_alive():
            print(f"TIMEOUT")
            p.terminate()
            p.join()
        
            if "best_S" in shared and shared["best_S"] is not None:
                print("Last UB before timeout =", shared["best_S"])
            else:
                print("No solution was found before timeoutz")
        end = time.perf_counter()
        print(f"Time: {end - start:.2f} s")

if __name__ == "__main__":
    main()


