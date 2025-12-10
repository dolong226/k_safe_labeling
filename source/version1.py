import time
import sys
from typing import List, Tuple
from multiprocessing import Process, Manager
from pysat.formula import CNF
from pysat.solvers import Glucose42, Cadical195

TIMEOUT = 600

def X_id(i: int, j: int, UB: int) -> int:
    return i * UB + j

def control_var_id(S: int, n: int, UB: int) -> int:
    return n * UB + S

def find_highest_degree_vertex(n: int, edges: List[Tuple[int,int]]):
    degree_count = [0] * n
    for (u,v) in edges:
        degree_count[u] += 1
        degree_count[v] += 1

    max_degree = degree_count[0]
    best_vertex = 0
    for v in range(1, n):
        if degree_count[v] > max_degree:
            max_degree = degree_count[v]
            best_vertex = v

    print(best_vertex)
    return best_vertex

    print(count)

def build_base_cnf(n: int, edges:List[Tuple[int,int]], k:int, UB: int) -> CNF:
    cnf = CNF()

    # symmetry breaking
    v0 = find_highest_degree_vertex(n, edges)
    max_allowed = UB//2
    for j in range(max_allowed + 1, UB + 1):
        cnf.append([-X_id(v0, j, UB), X_id(v0, j-1, UB)])

    # X_i,UB = 1
    for i in range(n):
        cnf.append([X_id(i,UB,UB)])

    # X_i,j => X_i,j+1
    for i in range (n):
        for j in range (1, UB):
            cnf.append([-X_id(i,j,UB), X_id(i,j+1,UB)])

    # -X_i1,1 V - X_i2,1
    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            cnf.append([-X_id(i1, 1, UB), -X_id(i2, 1, UB)])

    # -(K_i1,j ∧ K_i2,j)
    # -X_i1,j V X_i1,j-1 V -X_i2,j V X_i2,j-1
    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            for j in range(2, UB + 1):
                cnf.append([-X_id(i1,j,UB), X_id(i1,j-1,UB), -X_id(i2,j,UB), X_id(i2,j-1,UB)])

    # K_u,val => X_v,val-k V -X_v,val+k-1
    # -X_u,val V X_u,val-1 V X_v,val-k V -X_v,val+k-1
    directed_edges = edges + [(v,u) for u,v in edges]
    for u,v in directed_edges:
        for val in range(1,UB + 1): 
            condition_lits = []
            if val - k >= 1:
                condition_lits.append(X_id(v,val-k,UB))
            if val + k - 1 <= UB:
                condition_lits.append(-X_id(v,val + k - 1, UB))
            
            if not condition_lits:
                if val == 1:
                    clause = [-X_id(u,1,UB)]
                else:
                    clause = [-X_id(u,val,UB), X_id(u, val-1, UB)]
            else:
                if val == 1:
                    clause = [-X_id(u,1,UB)] + condition_lits
                else:
                    clause = [-X_id(u,val,UB), X_id(u,val-1,UB)] + condition_lits
            
            if clause:
                cnf.append(clause)
    
    for S in range(1, UB + 1):
        Cs = control_var_id(S,n,UB)
        for i in range(n):
            cnf.append([Cs, X_id(i,S,UB)])

    return cnf

def solve_inc(n: int, edges: List[Tuple[int, int]], k: int, UB: int, shared, solver_cls = Glucose42):
    cnf = build_base_cnf(n, edges, k,UB)
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
    if best_S is None:
        print("UNSAT")
        return None, None
    
    # decode labels
    labels = [-1] * n
    modelset = set(l for l in best_model if l > 0)
    for v in range(n):
        for l in range(1,UB + 1):
            if X_id(v,l,UB) in modelset:
                labels[v] = l
                break
    shared["labels"] = labels
    print("Optimal UB =", best_S)
    for v in range(n):
        print(f"Vertex {v} -> label {labels[v]}")

    return best_S, labels

def main():
    file_path = sys.argv[1]
    with open(file_path, "r") as f:
        n,m,k,UB = map(int, f.readline().split())
        edges = [tuple(map(int, f.readline().split())) for _ in range(m)]
    # UB = k * (n - 1) + 1

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
                print("No solution was found before timeout")
        end = time.perf_counter()
        print(f"Time: {end - start:.2f} s")

if __name__ == "__main__":
    main()