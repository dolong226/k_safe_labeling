import argparse
import csv
import os
import sys
import time
from datetime import datetime
from multiprocessing import Process, Manager
from typing import List, Tuple, Optional

from pysat.card import CardEnc, IDPool
from pysat.formula import CNF
from pysat.solvers import Glucose42, Cadical195

TIMEOUT = 600


def X_id(i: int, j: int, UB: int) -> int:
    return i * UB + j


def K_id(i: int, j: int, n: int, UB: int) -> int:
    # K[i,j]: vertex i has label = j
    return n * UB + i * UB + j


def control_var_id(S: int, n: int, UB: int) -> int:
    # Cs: span > S
    return 2 * n * UB + S


def symmetry_breaking(n: int, edges: List[Tuple[int, int]]) -> int:
    fmax = 0
    sym = 0
    arr = [0] * n
    for (u, v) in edges:
        arr[u] += 1
        arr[v] += 1
        if arr[u] > fmax:
            fmax = arr[u]
            sym = u
        if arr[v] > fmax:
            fmax = arr[v]
            sym = v
    return sym


# VALIDATION
def validate_solution(n: int, edges: List[Tuple[int, int]], k: int, labels: List[int]) -> Tuple[
    bool, List[str]]:
    if labels is None or len(labels) != n:
        return False, ["Labels missing or incomplete"]

    errors = []
    for u, v in edges:
        dist = abs(labels[u] - labels[v])
        if dist < k:
            errors.append(
                f"Edge ({u},{v}): distance {dist} < k={k} (labels: {labels[u]}, {labels[v]})")

    return (len(errors) == 0, errors)


# LOG
def write_log_file(log_path: str, instance_name: str, n: int, m: int, k: int, UB: int,
                   edges: List[Tuple[int, int]], labels: Optional[List[int]], span: Optional[int],
                   status: str, elapsed_time: float, num_vars: int, num_clauses: int,
                   solver_name: str,
                   sym_breaking: bool = False, sb_vertex: int = -1):
    with open(log_path, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("INSTANCE LOG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Instance: {instance_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Solver: {solver_name}\n")
        f.write(f"Timeout: {TIMEOUT}s\n")
        if sym_breaking:
            f.write(f"Symmetry Breaking: True (vertex {sb_vertex})\n\n")
        else:
            f.write(f"Symmetry Breaking: NO\n\n")

        # Result
        f.write("-" * 80 + "\n")
        f.write("RESULT SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Status: {status}\n")
        f.write(f"Optimal Span: {span}\n")
        f.write(f"Total time (CNF + Solve): {elapsed_time:.2f}s\n")

        # validation
        is_valid, errors = validate_solution(n, edges, k, labels)
        f.write(f"Validation: {'PASSED' if is_valid else 'FAILED'}\n")
        if not is_valid:
            f.write(f"Validation Errors: {len(errors)}\n")

        # Instance Parameters
        f.write("-" * 80 + "\n")
        f.write("INSTANCE PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of vertices (n): {n}\n")
        f.write(f"Number of edges (m): {m}\n")
        f.write(f"Distance parameter (k): {k}\n")
        f.write(f"Upper bound (UB): {UB}\n\n")
        f.write("Edge list:\n")
        for i, (u, v) in enumerate(edges):
            f.write(f"({u},{v})")
            if (i + 1) % 10 == 0:
                f.write("\n")
            elif i < len(edges) - 1:
                f.write(", ")
        f.write("\n\n")

        # SAT Solver Statistics
        f.write("-" * 80 + "\n")
        f.write("SAT SOLVER STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of variables: {num_vars}\n")
        f.write(f"Number of clauses: {num_clauses}\n\n")

        # Solution
        if labels is not None:
            f.write("-" * 80 + "\n")
            f.write("SOLUTION - VERTEX LABELING\n")
            f.write("-" * 80 + "\n")
            for v in range(n):
                f.write(f"Vertex {v} -> Label {labels[v]}\n")
            f.write("\n")

            # Validation Details
            f.write("-" * 80 + "\n")
            f.write("VALIDATION DETAILS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Checking edge constraints (|label[u] - label[v]| >= k={k}):\n")
            is_valid, errors = validate_solution(n, edges, k, labels)

            if is_valid:
                for u, v in edges:
                    dist = abs(labels[u] - labels[v])
                    f.write(f"Edge ({u},{v}): |{labels[u]}-{labels[v]}| = {dist} >= {k}\n")
                f.write(f"\nAll {m} edges satisfy the distance constraint.\n")
            else:
                for error in errors:
                    f.write(f"✗ {error}\n")

        # Footer
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF LOG\n")
        f.write("=" * 80 + "\n")


def solve_single_instance(instance_path: str, log_dir: str, solver_cls,
                          use_sym_breaking: bool = True) -> dict:
    instance_name = os.path.basename(instance_path)
    print(f"\n{'=' * 80}")
    print(f"Processing: {instance_name}")
    print(f"{'=' * 80}")

    # Read instance
    with open(instance_path, "r") as f:
        n, m, k, UB = map(int, f.readline().split())
        edges = [tuple(map(int, f.readline().split())) for _ in range(m)]

    print(f"n={n}, m={m}, k={k}, UB={UB}")

    if use_sym_breaking:
        print(f"Symmetry Breaking: ENABLED")

    # Log file path
    log_filename = instance_name.replace('.txt', '_log.txt')
    log_path = os.path.join(log_dir, log_filename)

    start_time = time.perf_counter()
    # Solve instance
    with Manager() as manager:
        shared = manager.dict()
        p = Process(target=solve_inc, args=(n, edges, k, UB, shared, solver_cls, use_sym_breaking))

        p.start()
        p.join(timeout=TIMEOUT)

        status = "SAT"
        best_S = None
        labels = None
        sb_vertex = -1

        if p.is_alive():
            print(f"TIMEOUT after {TIMEOUT}s")
            p.terminate()
            p.join()
            status = "TIMEOUT"

            if "best_S" in shared and shared["best_S"] is not None:
                best_S = shared["best_S"]
                labels = list(shared.get("labels", []))
                print(f"Best span before timeout: {best_S}")
            else:
                print("No solution found before timeout")
        else:
            if "best_S" in shared and shared["best_S"] is not None:
                best_S = shared["best_S"]
                labels = list(shared.get("labels", []))
                sb_vertex = shared.get("sb_vertex", -1)
                print(f"Optimal span: {best_S}")
            else:
                print("UNSAT")
                status = "UNSAT"

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        num_vars = shared.get("num_vars", 0)
        num_clauses = shared.get("num_clauses", 0)

        print(f"Time: {elapsed_time:.2f}s")

        # Write log file
        write_log_file(log_path, instance_name, n, m, k, UB, edges, labels,
                       best_S, status, elapsed_time, num_vars, num_clauses, solver_cls.__name__,
                       use_sym_breaking, sb_vertex)

    return {
        "instance": instance_name,
        "n": n,
        "m": m,
        "k": k,
        "UB": UB,
        "span": best_S if best_S is not None else "N/A",
        "time": f"{elapsed_time:.2f}",
        "variables": num_vars,
        "clauses": num_clauses,
        "solver": solver_cls.__name__,
        "status": status,
        "sym_breaking": "YES" if use_sym_breaking else "NO"
    }


def build_base_cnf(n: int, edges: List[Tuple[int, int]], k: int, UB: int) -> CNF:
    cnf = CNF()

    # Xij = 1
    for i in range(n):
        cnf.append([X_id(i, 1, UB)])

    # Xi,j -> Xi,j-1
    for i in range(n):
        for j in range(2, UB + 1):
            cnf.append([-X_id(i, j, UB), X_id(i, j - 1, UB)])

    # 3: K[i,j] <=> X[i,j] ^ -X[i,j+1]
    # TH đặc biệt: K[i,UB] <=> X[i,UB]
    for i in range(n):
        cnf.append([-K_id(i, UB, n, UB), X_id(i, UB, UB)])
        cnf.append([K_id(i, UB, n, UB), -X_id(i, UB, UB)])

        # TH chung: j < UB
        for j in range(1, UB):
            cnf.append([-K_id(i, j, n, UB), X_id(i, j, UB)])
            cnf.append([-K_id(i, j, n, UB), -X_id(i, j + 1, UB)])
            cnf.append([-X_id(i, j, UB), X_id(i, j + 1, UB), K_id(i, j, n, UB)])

    # Không có 2 đỉnh cùng nhãn (pairwise)
    # for i1 in range(n):
    #     for i2 in range(i1 + 1, n):
    #         for j in range(1, UB + 1):
    #            cnf.append([-K_id(i1, j, n, UB), -K_id(i2, j, n, UB)])
    vpool = IDPool(start_from=2 * n * UB + UB + 1)
    for j in range(1, UB + 1):
        k_vars_for_label_j = [K_id(i, j, n, UB) for i in range(n)]

        clauses = CardEnc.atmost(lits=k_vars_for_label_j, bound=1,
                                 vpool=vpool, encoding=1)
        for clause in clauses.clauses:
            cnf.append(clause)

    # distance constraints

    directed_edges = edges
    for u, v in directed_edges:
        for val in range(1, UB + 1):
            has_lower = val - k + 1 >= 1
            has_upper = val + k <= UB

            if has_lower and has_upper:
                # ¬K[u,val] ∨ ¬X[v,val-k+1] ∨ X[v,val+k]
                cnf.append([-K_id(u, val, n, UB),
                            -X_id(v, val - k + 1, UB),
                            X_id(v, val + k, UB)])

            elif has_lower:
                # ¬K[u,val] ∨ ¬X[v,val-k+1]
                cnf.append([-K_id(u, val, n, UB),
                            -X_id(v, val - k + 1, UB)])

            elif has_upper:
                # ¬K[u,val] ∨ X[v,val+k]
                cnf.append([-K_id(u, val, n, UB),
                            X_id(v, val + k, UB)])

            else:
                # ¬K[u,val] (buộc K[u,val] = FALSE)
                cnf.append([-K_id(u, val, n, UB)])

    # 6 
    # for S in range(1, UB):
    #     Cs = control_var_id(S,n,UB)
    #     for i in range(n):
    #         cnf.append([Cs, -X_id(i,S+1,UB)])

    return cnf


def solve_inc(n, edges, k, UB, shared, solver_cls=Glucose42, use_sym_breaking=True):
    cnf = build_base_cnf(n, edges, k, UB)
    solver = solver_cls(bootstrap_with=cnf.clauses)

    print("Variables: ", solver.nof_vars())
    print("Clauses: ", solver.nof_clauses())

    shared["num_vars"] = solver.nof_vars()
    shared["num_clauses"] = solver.nof_clauses()

    best_S, best_model = None, None
    high = UB

    sb_vertex = -1
    sb_assumption = None
    if use_sym_breaking:
        sb_vertex = symmetry_breaking(n, edges)
        print(f"Symmetry Breaking Vertex: {sb_vertex} (highest degree)")

    while high >= n:
        # C = control_var_id(high, n, UB)

        # if high < UB:
        #     bounds = [-X_id(i,high + 1, UB) for i in range(n)]
        # else:
        #     bounds = []
        #
        #
        # assumptions = [-C] + bounds
        #
        # if use_sym_breaking and sb_vertex >= 0:
        #     sb_lit = X_id(sb_vertex, (high + 1) // 2, UB)
        #     assumptions.append(sb_lit)
        #
        # sat = solver.solve(assumptions=assumptions)

        if high != UB:
            for i in range(n):
                solver.add_clause([-X_id(i, high + 1, UB)])

        solver.add_clause([-X_id(sb_vertex, (high + 1) // 2, UB)])

        sat = solver.solve()

        if sat:

            best_S = high
            best_model = solver.get_model()

            # decode label
            labels = [-1] * n
            modelset = set(l for l in best_model if l > 0)
            for v in range(n):
                for l in range(UB, 0, -1):
                    if X_id(v, l, UB) in modelset:
                        labels[v] = l
                        break

                        # new ub
            actual_span = max(labels)
            best_S = actual_span

            shared["labels"] = labels
            shared["best_S"] = best_S
            shared["best_model"] = best_model
            shared["sb_vertex"] = sb_vertex

            high = best_S - 1
        else:
            break

    if best_S is None:
        print("UNSAT")
        return None, None

    print("Optimal span =", best_S)
    for v in range(n):
        print(f"Vertex {v} -> label {labels[v]}")

    return best_S, labels


def main():
    parser = argparse.ArgumentParser(
        description='L(k,1)-Labeling SAT Solver with Symmetry Breaking - Batch Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=None)

    parser.add_argument('--solver', type=str, default='glucose',
                        choices=['glucose', 'cadical'],
                        help='SAT solver to use (default: glucose)')
    parser.add_argument('--instances', type=str, default='instances',
                        help='Directory containing instance files (default: instances)')
    parser.add_argument('--logs', type=str, default='logs',
                        help='Directory to save log files (default: logs)')
    parser.add_argument('--output', type=str, default='results.csv',
                        help='Output CSV file name (default: results.csv)')
    parser.add_argument('--no-sym-breaking', action='store_true',
                        help='Disable symmetry breaking (default: enabled)')

    args = parser.parse_args()

    # Select solver
    # solver_cls = Glucose42 if args.solver == 'glucose' else Cadical195
    solver_cls = Cadical195
    print(f"Using solver: {solver_cls.__name__}")
    print(f"Symmetry Breaking: {'DISABLED' if args.no_sym_breaking else 'ENABLED'}")

    # Create logs directory if not exists
    os.makedirs(args.logs, exist_ok=True)

    # Get instance files (instance1.txt to instance16.txt)
    instance_files = []
    for i in range(1, 17):
        instance_path = os.path.join(args.instances, f"instance{i}.txt")
        if os.path.exists(instance_path):
            instance_files.append(instance_path)
        else:
            print(f"Warning: {instance_path} not found, skipping...")

    if not instance_files:
        print(f"Error: No instance files found in {args.instances}/")
        return

    print(f"Found {len(instance_files)} instances to process\n")

    # Process all instances
    results = []
    for instance_path in instance_files:
        try:
            result = solve_single_instance(instance_path, args.logs, solver_cls,
                                           use_sym_breaking=not args.no_sym_breaking)
            results.append(result)
        except Exception as e:
            print(f"Error in {instance_path}: {e}")
            results.append({
                "instance": os.path.basename(instance_path),
                "n": "N/A",
                "m": "N/A",
                "k": "N/A",
                "UB": "N/A",
                "span": "N/A",
                "time": "N/A",
                "variables": "N/A",
                "clauses": "N/A",
                "solver": solver_cls.__name__,
                "status": "ERROR",
                "sym_breaking": "N/A"
            })

    # Write CSV summary
    csv_path = args.output
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['instance', 'n', 'm', 'k', 'UB', 'span', 'time',
                      'variables', 'clauses', 'solver', 'status', 'sym_breaking']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"\n{'=' * 80}")
    print(f"All instances processed!")
    print(f"Logs saved to: {args.logs}/")
    print(f"CSV summary saved to: {csv_path}")
    print(f"{'=' * 80}")


def main1():
    file_path = sys.argv[1]
    with open(file_path, "r") as f:
        n, m, k, UB = map(int, f.readline().split())
        edges = [tuple(map(int, f.readline().split())) for _ in range(m)]

    print(f"Instance: n={n}, m={m}, k={k}, UB={UB}")
    print(f"Edges: {edges}\n")

    start = time.perf_counter()

    with Manager() as manager:
        shared = manager.dict()
        p = Process(target=solve_inc, args=(n, edges, k, UB, shared, Glucose42, True))

        p.start()
        p.join(timeout=TIMEOUT)

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
