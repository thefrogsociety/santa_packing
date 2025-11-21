import pandas as pd
import numpy as np
import os
import time
import multiprocessing
from optimizer import PackingSolver

# --- CONFIGURATION ---
SUBMISSION_FILE = 'submission_reverse.csv'
SOURCE_FILE = 'submission.csv' # Use this to load existing good anchors if available
TARGET_N = 200
SHRINK_RATE = 0.99

def generate_csv_rows(solver, n_trees):
    rows = []
    for i in range(n_trees):
        row_id = f"{n_trees:03d}_{i}"
        x, y, deg = f"s{solver.state[i,0]:.15f}", f"s{solver.state[i,1]:.15f}", f"s{solver.state[i,2]:.15f}"
        rows.append([row_id, x, y, deg])
    return rows

def parse_state_from_rows(group):
    group = group.sort_values('id')
    xs = group['x'].astype(str).str.replace('s', '').astype(float).values
    ys = group['y'].astype(str).str.replace('s', '').astype(float).values
    degs = group['deg'].astype(str).str.replace('s', '').astype(float).values
    return np.column_stack((xs, ys, degs))

def get_box_size(group):
    xs = group['x'].astype(str).str.replace('s', '').astype(float)
    ys = group['y'].astype(str).str.replace('s', '').astype(float)
    return max(xs.max(), ys.max())

def get_dynamic_iterations(n, is_anchor=False):
    """Scale effort based on complexity."""
    base_multiplier = 400 if is_anchor else 200
    iters = int(n * base_multiplier)
    
    if is_anchor:
        return max(10000, min(60000, iters))
    else:
        return max(3000, min(30000, iters))

def solve_batch_task(args):
    start_n, end_n, existing_data = args
    results = []
    
    previous_solver_state = None
    previous_box_size = None
    
    print(f"  [Worker] Starting Batch N={start_n} down to {end_n}")
    
    for n in range(start_n, end_n - 1, -1):
        iters = get_dynamic_iterations(n, is_anchor=(n == start_n))
        
        # --- STEP 1: INITIALIZATION ---
        if n == start_n:
            # ANCHOR PUZZLE: Try to load from file, otherwise generate
            best_solver = None
            
            # A. Try loading from existing data (Warm Anchor)
            if n in existing_data:
                print(f"    [N={n}] Loading Anchor from file (Size {existing_data[n]['size']:.4f})...")
                loaded_state = existing_data[n]['state']
                loaded_size = existing_data[n]['size']
                
                solver = PackingSolver(n, loaded_size, compression_weight=100.0, 
                                     prior_state=loaded_state, prior_box_size=loaded_size)
                # Repair phase: Fix CSV rounding errors
                solver.solve(iterations=5000)
                
                if solver.check_valid():
                    best_solver = solver
                else:
                    # Rounding error too big? Relax 1% and squeeze back
                    print(f"    [N={n}] File anchor invalid. Relaxing 1%...")
                    relaxed_size = loaded_size * 1.01
                    solver = PackingSolver(n, relaxed_size, compression_weight=100.0, 
                                         prior_state=loaded_state, prior_box_size=loaded_size)
                    solver.solve(iterations=5000)
                    if solver.check_valid():
                        # Squeeze back to original size
                        solver = PackingSolver(n, loaded_size, compression_weight=100.0, 
                                             prior_state=solver.state, prior_box_size=relaxed_size)
                        solver.solve(iterations=10000)
                        if solver.check_valid():
                            best_solver = solver

            # B. Cold Start (If file load failed or didn't exist)
            if best_solver is None:
                init_mode = 'random' if n < 15 else 'lattice'
                print(f"    [N={n}] Fresh Anchor Start ({init_mode})...")
                
                min_theory = max(np.sqrt(0.5 * n), 1.0)
                start_size = min_theory * (2.0 if n < 15 else 1.3)
                
                while True:
                    solver = PackingSolver(n, start_size, compression_weight=0.0, init_mode=init_mode)
                    solver.solve(iterations=5000)
                    if solver.check_valid():
                        break
                    start_size *= 1.1
                best_solver = solver
            
            # Optimize Anchor (Squeeze as much as possible before chaining)
            current_size = best_solver.box_size
            while True:
                target_size = current_size * SHRINK_RATE
                success = False
                for i in range(3): # Anchor retries
                    temp = PackingSolver(n, target_size, compression_weight=100.0, 
                                       prior_state=best_solver.state, prior_box_size=best_solver.box_size)
                    temp.solve(iterations=iters)
                    if temp.check_valid():
                        best_solver = temp
                        current_size = target_size
                        success = True
                        break
                if not success: break
                
        else:
            # CHAIN STRATEGY (Waterfall)
            # Inherit from N+1 (in memory, so high precision)
            
            inherited_state = previous_solver_state[:n].copy()
            current_size = previous_box_size
            
            # Robust Initialization
            # Run a quick "Untangle" (0 compression) to settle trees
            solver = PackingSolver(n, current_size, compression_weight=0.0, 
                                 prior_state=inherited_state, prior_box_size=current_size)
            solver.solve(iterations=2000)
            
            best_solver = solver
            
            # Squeeze Loop
            while True:
                target_size = current_size * SHRINK_RATE
                success = False
                for i in range(2): # Chain retries (lower because start is good)
                    temp = PackingSolver(n, target_size, compression_weight=100.0, 
                                       prior_state=best_solver.state, prior_box_size=best_solver.box_size)
                    temp.solve(iterations=iters)
                    if temp.check_valid():
                        best_solver = temp
                        current_size = target_size
                        success = True
                        break
                if not success: break

        # --- SAVE ---
        if n % 5 == 0 or n == start_n:
             print(f"      [DONE] N={n} Size: {best_solver.box_size:.4f}")
        
        rows = generate_csv_rows(best_solver, n)
        results.extend(rows)
        
        previous_solver_state = best_solver.state
        previous_box_size = best_solver.box_size
        
    return results

def load_all_data_for_warm_start():
    """Loads source submission file to use as anchors."""
    if not os.path.exists(SOURCE_FILE): return {}
    
    print(f"Loading {SOURCE_FILE} to use as Warm Anchors...")
    df = pd.read_csv(SOURCE_FILE)
    df['n_group'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    
    data_map = {}
    for n, group in df.groupby('n_group'):
        # Skip grids
        is_grid = group['deg'].astype(str).str.contains('s0.000000').all()
        if not is_grid:
            state = parse_state_from_rows(group)
            size = get_box_size(group)
            data_map[n] = {'state': state, 'size': size}
            
    return data_map

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    print("--- SMART REVERSE WATERFALL SOLVER ---")
    
    # 1. Load Existing Data for Anchors
    existing_data_map = load_all_data_for_warm_start()
    print(f"Found {len(existing_data_map)} potential anchors.")
    
    # 2. Batching (Use 8 workers for safety)
    num_workers = max(1, multiprocessing.cpu_count() - 4)
    
    ranges = []
    batch_size = TARGET_N // num_workers
    for i in range(num_workers):
        start = TARGET_N - (i * batch_size)
        end = start - batch_size + 1
        if i == num_workers - 1: end = 1 
        ranges.append((start, end, existing_data_map))
        
    print(f"Batch Assignments: {[ (r[0], r[1]) for r in ranges]}")
    
    # 3. Launch
    pool = multiprocessing.Pool(processes=num_workers, maxtasksperchild=1)
    
    all_data_rows = []
    
    for batch_rows in pool.imap_unordered(solve_batch_task, ranges):
        all_data_rows.extend(batch_rows)
        
        print(f">> Batch finished. Saving progress...")
        
        df_out = pd.DataFrame(all_data_rows, columns=['id', 'x', 'y', 'deg'])
        df_out['sort_key'] = df_out['id'].apply(lambda x: tuple(map(int, x.split('_'))))
        df_out = df_out.sort_values('sort_key').drop(columns=['sort_key'])
        df_out.to_csv(SUBMISSION_FILE, index=False)
        
    pool.close()
    pool.join()
    print("\nAll reversed batches complete.")