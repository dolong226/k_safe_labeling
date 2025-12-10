from pysat.formula import CNF
from typing import List, Tuple
from pysat.solvers import Glucose42, Cadical195
import time
from multiprocessing import Process, Manager
import sys

TIMEOUT = 600

def X_id(i: int, j: int, UB: int) -> int:
    return i * UB + j

def control_var_id(S: int, n: int, UB: int) -> int:
    return n * UB + S

def build_base_cnf(n: int, edges:List[Tuple[int,int]], k:int, UB: int, root_fix:bool = True) -> CNF:
    cnf = CNF()


    # X_i,1 = True (label >= 1)
    for i in range(n):
        cnf.append([ X_id(i, 1, UB) ])

    # X_{j+1} => X_j   (for j = 1..n-1)
    for i in range(n):
        for j in range(1, UB):           # j = 1..n-1
            cnf.append([ -X_id(i, j+1, UB), X_id(i, j, UB) ])

    # at most one vertex per 
    # K_ij <-> X_ij ∧ ¬X_i,j+1  (K_in <-> X_in)
    for j in range(1, UB+1):             # j = 1..n
        for i1 in range(n):
            for i2 in range(i1+1, n):
                if j == UB:
                    # K_i, n <-> X_i,n  => at most one: ¬X_i1,n ∨ ¬X_i2,n
                    cnf.append([ -X_id(i1, UB, UB), -X_id(i2, UB, UB) ])
                else:
                    # ¬(X_i1,j ∧ ¬X_i1,j+1) ∨ ¬(X_i2,j ∧ ¬X_i2,j+1)
                    # -> (~X_i1,j ∨ X_i1,j+1 ∨ ~X_i2,j ∨ X_i2,j+1)
                    cnf.append([
                        -X_id(i1, j, UB), X_id(i1, j+1, UB),
                        -X_id(i2, j, UB), X_id(i2, j+1, UB)])

    # k-safe constraints
    directed_edges = edges + [(v,u) for (u,v) in edges]
    for (u,v) in directed_edges:
        for val in range(1, UB+1):
            cond = []
            # v <= val-k  or  v >= val+k
            low_index = val - k + 1          # if v <= val-k  <=>  ¬X_v, low_index
            if 1 <= low_index <= UB:
                cond.append( -X_id(v, low_index, UB) )   # ¬X_v, val-k+1

            high_index = val + k             # if v >= val+k  <=> X_v, high_index
            if 1 <= high_index <= UB:
                cond.append( X_id(v, high_index, UB) )

            # build clause
            if val == UB:
                # K_u,UB == X_u,UB  => clause is ¬X_u,UB ∨ or(cond)
                if cond:
                    clause = [ -X_id(u, UB, UB) ] + cond
                else:
                    clause = [ -X_id(u, UB, UB) ]

            else:
                # K_u,val == X_u,val ∧ ¬X_u,val+1
                if cond:
                    clause = [ -X_id(u, val, UB), X_id(u, val+1, UB) ] + cond
                else:
                    clause = [ -X_id(u, val, UB), X_id(u, val+1, UB) ]

            cnf.append(clause)

    # incremental
    for S in range(1, UB):
        Cs = control_var_id(S,n,UB)
        for i in range(n):
            cnf.append([Cs, -X_id(i,S+1,UB)])

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
        for l in range(UB, 0, -1):
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