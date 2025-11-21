import pandas as pd
import numpy as np
import multiprocessing
import os
import time
from optimizer import PackingSolver

SUBMISSION_FILE = 'submission.csv'
REPAIR_SHRINK_RATE = 0.99    
MAX_LOOPS = 5 

def get_box_size(group):
    xs = group['x'].astype(str).str.replace('s', '').astype(float)
    ys = group['y'].astype(str).str.replace('s', '').astype(float)
    return max(xs.max(), ys.max())

def parse_state_from_rows(group):
    group = group.sort_values('id') 
    xs = group['x'].astype(str).str.replace('s', '').astype(float).values
    ys = group['y'].astype(str).str.replace('s', '').astype(float).values
    degs = group['deg'].astype(str).str.replace('s', '').astype(float).values
    return np.column_stack((xs, ys, degs))

def get_dynamic_iterations(n, mode='standard'):
    if mode == 'init':
        return max(2000, min(10000, n * 100))
    elif mode == 'squeeze':
        return max(10000, int(n * 400))
    elif mode == 'search':
        return max(10000, int(n * 300))
    return 5000

def solve_repair_task(args):
    n, neighbor_size, inherited_state = args
    
    strict_target = neighbor_size - 0.000001
    
    best_solver = None
    valid_start_found = False
    
    iter_init = get_dynamic_iterations(n, 'init')
    iter_squeeze = get_dynamic_iterations(n, 'squeeze')
    iter_search = get_dynamic_iterations(n, 'search')

    if inherited_state is not None and len(inherited_state) >= n:
        print(f"  [FIXING N={n}] Target < {neighbor_size:.4f} (Inheritance)")
        my_state = inherited_state[:n].copy()
        
        solver = PackingSolver(n, strict_target, compression_weight=100.0, 
                             prior_state=my_state, prior_box_size=neighbor_size)
        solver.solve(iterations=iter_init) 
        if solver.check_valid():
            valid_start_found = True
            best_solver = solver
            print(f"    [N={n}] Direct inheritance valid.")
        
        if not valid_start_found:
            relaxed_size = neighbor_size * 1.05 
            solver = PackingSolver(n, relaxed_size, compression_weight=100.0, 
                                 prior_state=my_state, prior_box_size=neighbor_size)
            solver.solve(iterations=iter_init * 2) 
            
            if solver.check_valid():
                current_squeeze = relaxed_size
                squeezed_solver = solver
                while current_squeeze > strict_target:
                    step_down = max(strict_target, current_squeeze * 0.98)
                    temp_solver = PackingSolver(n, step_down, compression_weight=100.0,
                                              prior_state=squeezed_solver.state,
                                              prior_box_size=squeezed_solver.box_size)
                    temp_solver.solve(iterations=iter_init)
                    if temp_solver.check_valid():
                        squeezed_solver = temp_solver
                        current_squeeze = step_down
                    else:
                        break
                if current_squeeze <= strict_target:
                    valid_start_found = True
                    best_solver = squeezed_solver
                    print(f"    [N={n}] Recovery successful!")

    if not valid_start_found:
        current_start_size = strict_target * 0.99 
        print(f"  [FIXING N={n}] Inheritance failed. Entering INFINITE SEARCH < {strict_target:.4f}")
        
        attempt_count = 0
        
        while not valid_start_found:
            attempt_count += 1
            
            if attempt_count % 10 == 0:
                print(f"    [N={n}] Persistence Check: Attempt #{attempt_count}...")

            solver = PackingSolver(n, current_start_size, compression_weight=0.0)
            solver.solve(iterations=iter_search) 
            
            if solver.check_valid():
                valid_start_found = True
                best_solver = solver
                print(f"    [N={n}] Brute force SUCCESS on attempt #{attempt_count}!")
                break
            
            if attempt_count == 50:
                print(f"    [N={n}] Relaxing search to exact limit {strict_target:.4f}")
                current_start_size = strict_target

    current_size = best_solver.box_size
    while True:
        target_size = current_size * REPAIR_SHRINK_RATE
        success = False
        for i in range(2): 
            solver = PackingSolver(n, target_size, compression_weight=100.0, 
                                 prior_state=best_solver.state, 
                                 prior_box_size=best_solver.box_size)
            solver.solve(iterations=iter_squeeze)
            if solver.check_valid():
                best_solver = solver
                current_size = target_size
                success = True
                print(f"      [N={n}] Improved! New best: {current_size:.4f}")
                break
        if not success: break 
            
    rows = []
    for i in range(n):
        row_id = f"{n:03d}_{i}"
        x, y, deg = f"s{best_solver.state[i,0]:.15f}", f"s{best_solver.state[i,1]:.15f}", f"s{best_solver.state[i,2]:.15f}"
        rows.append([row_id, x, y, deg])
        
    return (n, rows, current_size)

def find_global_flaws(df):
    stats = []
    df['n_group'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    for n, group in df.groupby('n_group'):
        size = get_box_size(group)
        stats.append({'n': n, 'size': size})
    stats_df = pd.DataFrame(stats).sort_values('n', ascending=False)
    
    flaws = []
    min_future_size = float('inf')
    best_source_n = -1 
    
    for index, row in stats_df.iterrows():
        n = int(row['n'])
        size = row['size']
        if size >= min_future_size:
            flaws.append((n, min_future_size, best_source_n))
        else:
            min_future_size = size
            best_source_n = n
            
    flaws.sort(key=lambda x: x[0])
    return flaws

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if not os.path.exists(SUBMISSION_FILE): exit()
    
    print("--- ANALYZING SUBMISSION (INFINITE PERSISTENCE) ---")
    for loop in range(MAX_LOOPS):
        print(f"\n>>> REPAIR LOOP {loop + 1}/{MAX_LOOPS} <<<")
        df = pd.read_csv(SUBMISSION_FILE)
        flaws_metadata = find_global_flaws(df)
        
        if not flaws_metadata:
            print("No flaws found.")
            break
            
        print(f"Found {len(flaws_metadata)} flaws. Starting infinite repair...")
        
        tasks = []
        for item in flaws_metadata:
            n_flaw, target_size, source_n_id = item
            inherited_state = None
            if source_n_id != -1:
                source_rows = df[df['id'].str.startswith(f"{source_n_id:03d}_")]
                if not source_rows.empty:
                    inherited_state = parse_state_from_rows(source_rows)
            tasks.append((n_flaw, target_size, inherited_state))

        num_workers = max(1, multiprocessing.cpu_count() - 4)
        pool = multiprocessing.Pool(processes=num_workers, maxtasksperchild=1)
        
        any_fixed = False
        for result in pool.imap_unordered(solve_repair_task, tasks):
            n, rows, final_size = result
            if rows is not None:
                print(f"[FIXED] N={n}: New Size {final_size:.4f}")
                current_df = pd.read_csv(SUBMISSION_FILE)
                current_df = current_df[~current_df['id'].str.startswith(f"{n:03d}_")]
                new_df = pd.DataFrame(rows, columns=['id', 'x', 'y', 'deg'])
                current_df = pd.concat([current_df, new_df], ignore_index=True)
                current_df['sort_key'] = current_df['id'].apply(lambda x: tuple(map(int, x.split('_'))))
                current_df = current_df.sort_values('sort_key').drop(columns=['sort_key'])
                current_df.to_csv(SUBMISSION_FILE, index=False)
                any_fixed = True
                
        pool.close()
        pool.join()
        
        if not any_fixed: break
    print("\nDone.")