import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIG ---
SUBMISSION_FILE = 'submission.csv'

def get_box_size(group):
    # Extract numbers from "s12.345" format
    xs = group['x'].astype(str).str.replace('s', '').astype(float)
    ys = group['y'].astype(str).str.replace('s', '').astype(float)
    
    # Get bounds
    max_x = xs.max()
    max_y = ys.max()
    
    # Note: We technically need to check the polygon bounds, 
    # but checking the center coordinate is a very close proxy for spotting outliers.
    # For exactness, we'd need the geometry, but this is fast.
    return max(max_x, max_y)

def check_monotonicity():
    if not os.path.exists(SUBMISSION_FILE):
        print("No submission file found.")
        return

    df = pd.read_csv(SUBMISSION_FILE)
    
    # Add N column
    df['N'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    
    # Calculate size for each N
    results = []
    for n, group in df.groupby('N'):
        size = get_box_size(group)
        # Density = N / Area
        density = n / (size ** 2) if size > 0 else 0
        results.append({'N': n, 'Size': size, 'Density': density})
    
    stats = pd.DataFrame(results).sort_values('N')
    
    # Detect Flaws
    # A flawed N is one where Size(N) > Size(N+1)
    # (Because we could just take the N+1 solution and delete a tree!)
    bad_n_list = []
    
    print("\n--- ANOMALY REPORT ---")
    for i in range(len(stats) - 1):
        current_row = stats.iloc[i]
        next_row = stats.iloc[i+1]
        
        if current_row['Size'] > next_row['Size']:
            diff = current_row['Size'] - next_row['Size']
            print(f"[FLAW] N={int(current_row['N'])} is larger than N={int(next_row['N'])}!")
            print(f"       Size {current_row['Size']:.4f} vs {next_row['Size']:.4f} (Diff: {diff:.4f})")
            bad_n_list.append(int(current_row['N']))

    if not bad_n_list:
        print("No monotonicity flaws found! (All N < N+1)")
    else:
        print(f"\nFound {len(bad_n_list)} puzzles to redo: {bad_n_list}")

    # PLOT
    plt.figure(figsize=(12, 6))
    plt.plot(stats['N'], stats['Size'], label='Box Size')
    
    # Highlight bad points
    bad_stats = stats[stats['N'].isin(bad_n_list)]
    plt.scatter(bad_stats['N'], bad_stats['Size'], color='red', label='Suboptimal', zorder=5)
    
    plt.xlabel('Number of Trees (N)')
    plt.ylabel('Box Size')
    plt.title('Packing Efficiency Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return bad_n_list

if __name__ == "__main__":
    import os
    check_monotonicity()