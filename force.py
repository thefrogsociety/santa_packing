import pandas as pd
import numpy as np
import os

SUBMISSION_FILE = 'submission1.csv'

def get_box_size(group):
    """Calculates the box size from the string format s12.34."""
    try:
        xs = group['x'].astype(str).str.replace('s', '').astype(float)
        ys = group['y'].astype(str).str.replace('s', '').astype(float)
        return max(xs.max(), ys.max())
    except Exception as e:
        print(f"Error parsing box size: {e}")
        return 9999.9

def force_fix_submission():
    if not os.path.exists(SUBMISSION_FILE):
        print("No submission file found.")
        return

    print(f"Reading {SUBMISSION_FILE}...")
    df = pd.read_csv(SUBMISSION_FILE)
    
    df['n_group'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    
    sizes = {}
    grouped = df.groupby('n_group')
    for n, group in grouped:
        sizes[n] = get_box_size(group)
    
    print("Scanning for non-monotonic flaws (Backwards from 199)...")
    
    fixed_count = 0
    
    for n in range(199, 0, -1):
        current_size = sizes.get(n, 9999.0)
        next_size = sizes.get(n+1, 9999.0)
        
        if current_size > next_size:
            print(f"[FIX] N={n} ({current_size:.4f}) is larger than N={n+1} ({next_size:.4f}). Overwriting...")
            

            better_rows = df[df['n_group'] == (n + 1)].sort_values('id').copy()
            

            better_rows = better_rows.iloc[:n]
            

            new_ids = [f"{n:03d}_{i}" for i in range(n)]
            better_rows['id'] = new_ids
            better_rows['n_group'] = n 
            

            df = df[df['n_group'] != n]
            

            df = pd.concat([df, better_rows], ignore_index=True)
            
            sizes[n] = next_size
            fixed_count += 1
            
    if fixed_count > 0:
        print(f"\nApplied {fixed_count} fixes. Saving...")
        
        if 'n_group' in df.columns:
            del df['n_group']
            
        df['sort_key'] = df['id'].apply(lambda x: tuple(map(int, x.split('_'))))
        df = df.sort_values('sort_key').drop(columns=['sort_key'])
        
        df.to_csv(SUBMISSION_FILE, index=False)
        print("Done! Submission is now strictly monotonic.")
    else:
        print("\nNo flaws found. Your submission is already monotonic.")

if __name__ == "__main__":
    force_fix_submission()