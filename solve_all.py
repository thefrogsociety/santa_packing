import pandas as pd
import numpy as np
import os
import time
import multiprocessing
import sys
from optimizer import PackingSolver 

TARGET_N = 200              
SUBMISSION_FILE = 'submission_final_run.csv'
BATCH_SAVE_SIZE = 5

SMALL_N_CUTOFF = 15
SMALL_N_ITERATIONS = 5000     
SMALL_N_RETRIES = 100         
SMALL_N_SHRINK_RATE = 0.98    
LARGE_N_RETRIES = 5           
LARGE_N_SHRINK_RATE = 0.99    
LARGE_N_ITERATIONS = 30000    

def solve_small_n(n):
    """High-quality brute-force shrink for small N with smarter starting size."""
    start_time = time.time()

    if n <= 2:
        class SimpleSolver:
            def __init__(self, n, box_size):
                self.box_size = box_size
                self.state = np.zeros((n, 3), dtype=float)
                if n == 1:
                    self.state[0, :2] = [box_size/2.0, box_size/2.0]
                else:
                    self.state[0, :2] = [box_size*0.25, box_size/2.0]
                    self.state[1, :2] = [box_size*0.75, box_size/2.0]
            def check_valid(self): 
                return True
            def solve(self, iterations=0): 
                return
        min_theory_side = max(np.sqrt(0.5 * n), 1.0)
        box_size = min_theory_side  
        return SimpleSolver(n, box_size)

    min_theory_side = max(np.sqrt(0.5 * n), 1.0)
    
    current_size = min_theory_side * 1.5 
    
    print(f"\n[SMALL N START] N={n:03d}. Theoretical min: {min_theory_side:.4f}. Starting size: {current_size:.4f}")
    
    solver = PackingSolver(n, current_size, compression_weight=100.0) 
    while not solver.check_valid():
        if time.time() - start_time > SMALL_N_MAX_TIME:
            print(f"[SMALL N TIMEOUT] N={n:03d}. Seeding timed out after {SMALL_N_MAX_TIME}s. Returning last seed (may be invalid).")
            return solver
        current_size *= 1.1 
        print(f"[SMALL N SEED] N={n:03d}. Current box size invalid. Expanding to {current_size:.4f}")
        solver = PackingSolver(n, current_size, compression_weight=100.0) 
        solver.solve(iterations=4000)
    
    best_solver = solver 
    print(f"[SMALL N SEED] N={n:03d}. Valid seed found at {current_size:.4f}.")

    while True:
        if time.time() - start_time > SMALL_N_MAX_TIME:
            print(f"[SMALL N TIMEOUT] N={n:03d}. Optimization timed out after {SMALL_N_MAX_TIME}s. Returning best found.")
            break

        target_size = current_size * SMALL_N_SHRINK_RATE
        print(f"[SMALL N OPT] N={n:03d}. Attempting shrink to {target_size:.6f} (Current best: {current_size:.6f})")
        success = False
        
        for i in range(SMALL_N_RETRIES):
            if time.time() - start_time > SMALL_N_MAX_TIME:
                print(f"[SMALL N TIMEOUT] N={n:03d}. Retry loop timed out after {SMALL_N_MAX_TIME}s.")
                break

            temp_solver = PackingSolver(n, target_size, compression_weight=100.0, 
                                        prior_state=best_solver.state, 
                                        prior_box_size=current_size)
            temp_solver.solve(iterations=SMALL_N_ITERATIONS)
            
            if temp_solver.check_valid():
                best_solver = temp_solver
                current_size = target_size
                success = True
                print(f"[SMALL N SUCCESS] N={n:03d}. Shrink successful on retry {i+1}. New best size: {current_size:.6f}")
                break 
            else:
                print(f"[SMALL N RETRY] N={n:03d}. Retry {i+1}/{SMALL_N_RETRIES} failed to validate.")
        
        if not success: 
            print(f"[SMALL N STOP] N={n:03d}. Failed to shrink further after {SMALL_N_RETRIES} retries or timeout.")
            break 
            
    return best_solver

submission_cache_df = None
last_save_count = 0

def get_box_size(group):
    """Calculates the box size from the string format s12.34."""
    xs = group['x'].astype(str).str.replace('s', '').astype(float)
    ys = group['y'].astype(str).str.replace('s', '').astype(float)
    return max(xs.max(), ys.max())

def get_dynamic_iterations(n, is_anchor=False):
    """Scale effort based on complexity. Original multipliers remain."""
    base_multiplier = 400 if is_anchor else 200
    iters = int(n * base_multiplier)
    
    if is_anchor:
        return max(10000, min(60000, iters))
    else:
        return max(3000, min(30000, iters))

def generate_csv_rows(solver, n_trees):
    """Formats the solver state into the competition CSV format (HIGH PRECISION)."""
    rows = []
    if solver and hasattr(solver, 'state'):
        for i in range(n_trees):
            row_id = f"{n_trees:03d}_{i}"
            x = f"s{solver.state[i,0]:.15f}" 
            y = f"s{solver.state[i,1]:.15f}"
            deg = f"s{solver.state[i,2]:.15f}"
            rows.append([row_id, x, y, deg])
    else:
        cols = int(np.ceil(np.sqrt(n_trees)))
        spacing_x, spacing_y = 1.0, 1.2
        print(f"[FALLBACK] Using grid initialization for N={n_trees}.")
        for i in range(n_trees):
            row_id = f"{n_trees:03d}_{i}"
            x = f"s{(i%cols)*spacing_x:.15f}"
            y = f"s{(i//cols)*spacing_y:.15f}"
            rows.append([row_id, x, y, "s0.000000000000000"])
    return rows

def solve_small_n(n):
    """High-quality brute-force shrink for small N with smarter starting size."""
    min_theory_side = max(np.sqrt(0.5 * n), 1.0)
    
    current_size = min_theory_side * 1.5 
    
    print(f"\n[SMALL N START] N={n:03d}. Theoretical min: {min_theory_side:.4f}. Starting size: {current_size:.4f}")
    
    solver = PackingSolver(n, current_size, compression_weight=100.0) 
    while not solver.check_valid():
        current_size *= 1.1 
        print(f"[SMALL N SEED] N={n:03d}. Current box size invalid. Expanding to {current_size:.4f}")
        solver = PackingSolver(n, current_size, compression_weight=100.0) 
        solver.solve(iterations=4000)
    
    best_solver = solver 
    print(f"[SMALL N SEED] N={n:03d}. Valid seed found at {current_size:.4f}.")

    while True:
        target_size = current_size * SMALL_N_SHRINK_RATE
        print(f"[SMALL N OPT] N={n:03d}. Attempting shrink to {target_size:.6f} (Current best: {current_size:.6f})")
        success = False
        
        for i in range(SMALL_N_RETRIES):
            temp_solver = PackingSolver(n, target_size, compression_weight=100.0, 
                                        prior_state=best_solver.state, 
                                        prior_box_size=current_size)
            temp_solver.solve(iterations=SMALL_N_ITERATIONS)
            
            if temp_solver.check_valid():
                best_solver = temp_solver
                current_size = target_size
                success = True
                print(f"[SMALL N SUCCESS] N={n:03d}. Shrink successful on retry {i+1}. New best size: {current_size:.6f}")
                break 
            else:
                print(f"[SMALL N RETRY] N={n:03d}. Retry {i+1}/{SMALL_N_RETRIES} failed to validate.")
        
        if not success: 
            print(f"[SMALL N STOP] N={n:03d}. Failed to shrink further after {SMALL_N_RETRIES} retries.")
            break 
            
    return best_solver

def solve_large_n(n):
    """High-quality adaptive squeeze for large N with smarter starting size."""
    min_theory_side = np.sqrt(0.5 * n)
    
    current_size = min_theory_side * 1.15
    
    best_solver = None
    iters = get_dynamic_iterations(n)
    
    print(f"\n[LARGE N START] N={n:03d}. Theoretical min: {min_theory_side:.4f}. Starting size: {current_size:.4f}. Iters: {iters}")

    while True:
        solver = PackingSolver(n, current_size, compression_weight=0.0, init_mode='lattice')
        solver.solve(iterations=10000) 
        if solver.check_valid():
            best_solver = solver
            print(f"[LARGE N SEED] N={n:03d}. Valid seed found at {current_size:.4f}.")
            break
        current_size *= 1.1
        print(f"[LARGE N SEED] N={n:03d}. Initial box size invalid. Expanding to {current_size:.4f}")

    current_size = best_solver.box_size
    
    while True:
        target_size = current_size * LARGE_N_SHRINK_RATE
        print(f"[LARGE N OPT] N={n:03d}. Attempting shrink to {target_size:.6f} (Current best: {current_size:.6f})")
        
        if current_size - target_size < 0.0005: 
            print(f"[LARGE N STOP] N={n:03d}. Improvement too small (<0.0005). Stopping.")
            break 
        
        success = False
        for i in range(LARGE_N_RETRIES):
            solver = PackingSolver(n, target_size, compression_weight=100.0, 
                                   prior_state=best_solver.state, 
                                   prior_box_size=best_solver.box_size)
            solver.solve(iterations=iters) 
            
            if solver.check_valid():
                best_solver = solver
                current_size = target_size
                success = True
                print(f"[LARGE N SUCCESS] N={n:03d}. Shrink successful on retry {i+1}. New best size: {current_size:.6f}")
                break
            else:
                print(f"[LARGE N RETRY] N={n:03d}. Retry {i+1}/{LARGE_N_RETRIES} failed to validate.")
        
        if not success:
            semi_target_size = current_size * (1.0 - (1.0 - LARGE_N_SHRINK_RATE) / 2.0)
            if semi_target_size < current_size:
                 print(f"[LARGE N TWEAK] N={n:03d}. Retries failed. Trying gentle shrink to {semi_target_size:.6f}.")
                 solver = PackingSolver(n, semi_target_size, compression_weight=100.0, 
                                   prior_state=best_solver.state, 
                                   prior_box_size=best_solver.box_size)
                 solver.solve(iterations=iters // 2)
                 if solver.check_valid():
                    best_solver = solver
                    current_size = semi_target_size
                    print(f"[LARGE N TWEAK SUCCESS] N={n:03d}. Gentle shrink successful. New best size: {current_size:.6f}")
                    continue 
            
            print(f"[LARGE N STOP] N={n:03d}. Failed to shrink further after {LARGE_N_RETRIES} retries and tweak.")
            break 
            
    return best_solver

def worker_task(n):
    """The function that runs inside each CPU core."""
    start_time = time.time()
    solver = None
    try:

        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        if n < SMALL_N_CUTOFF:
            solver = solve_small_n(n)
        else:
            solver = solve_large_n(n)
        
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        final_size = solver.box_size if solver else 9999.0
        rows = generate_csv_rows(solver, n)
        elapsed = time.time() - start_time
        return (n, rows, elapsed, final_size)
    except Exception as e:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        print(f"[CRITICAL ERROR] N={n:03d} failed completely: {e}")
        return (n, None, str(e), 9999.0)

def load_initial_cache(submission_file):
    """Reads existing submission data into memory."""
    global submission_cache_df
    if not os.path.exists(submission_file): 
        print(f"[PROGRESS] {submission_file} not found. Starting from scratch.")
        submission_cache_df = pd.DataFrame(columns=['id', 'x', 'y', 'deg'])
    else:
        submission_cache_df = pd.read_csv(submission_file)
        print(f"[PROGRESS] Loaded {len(submission_cache_df)} rows from previous run. Resuming.")
        
    completed_n = set(submission_cache_df['id'].apply(lambda x: int(x.split('_')[0])))
    return completed_n

def save_cache_to_disk(force=False):
    """Writes the in-memory cache to disk, consolidating I/O."""
    global submission_cache_df
    global last_save_count

    if force or (len(submission_cache_df) - last_save_count >= BATCH_SAVE_SIZE * TARGET_N):
        print(f"\n[DISK SAVE] Writing {len(submission_cache_df)} rows to disk. Consolidating I/O...")
        
        submission_cache_df['sort_key'] = submission_cache_df['id'].apply(lambda x: tuple(map(int, x.split('_'))))
        df_to_save = submission_cache_df.sort_values('sort_key').drop(columns=['sort_key'])
        df_to_save.to_csv(SUBMISSION_FILE, index=False)
        last_save_count = len(submission_cache_df)
        print("[DISK SAVE] Save complete.")


def process_results(results_iterator, total_jobs):
    """Processes results, updates in-memory cache, and batches disk saves."""
    global submission_cache_df
    solved_count = 0

    for n, rows, elapsed, final_size in results_iterator:
        if rows is not None:
            solved_count += 1
            
            print(f"[FINAL RESULT] N={n:03d} | Size: {final_size:.6f} in {elapsed:.1f}s. ({solved_count}/{total_jobs} complete)")
            
            submission_cache_df = submission_cache_df[~submission_cache_df['id'].str.startswith(f"{n:03d}_")]
            
            new_df = pd.DataFrame(rows, columns=['id', 'x', 'y', 'deg'])
            submission_cache_df = pd.concat([submission_cache_df, new_df], ignore_index=True)

            if solved_count % BATCH_SAVE_SIZE == 0:
                save_cache_to_disk()
        
        else:
            print(f"[SAVE ERROR] N={n:03d} failed to solve. Skipping cache update.")
    
    save_cache_to_disk(force=True)


def solve_all():
    """Main entry point for solving all packing problems using multiprocessing."""
    
    print("--- LAUNCHING TURBO QUALITY/VERBOSE MODE (I/O OPTIMIZED) ---")
    
    multiprocessing.freeze_support()
    
    completed_n = load_initial_cache(SUBMISSION_FILE)
    
    work_queue = [n for n in range(1, TARGET_N + 1) if n not in completed_n]
    total_jobs = len(work_queue)
        
    if not work_queue:
        print("ALL TARGETS SOLVED!")
        return 
    
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Launching {num_workers} workers to solve {total_jobs} puzzles.")
        
    pool = multiprocessing.Pool(processes=num_workers, maxtasksperchild=1)
        
    results_iterator = pool.imap_unordered(worker_task, work_queue)
        
    process_results(results_iterator, total_jobs)
        
    pool.close()
    pool.join()

    print("\nAll high-quality optimization jobs complete.")


if __name__ == "__main__":
    solve_all()