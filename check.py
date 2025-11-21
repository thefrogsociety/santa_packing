import pandas as pd
import matplotlib.pyplot as plt
from geometry import TreePoly
import sys

SUBMISSION_FILE = 'submission.csv'
DEFAULT_VIEW_N = [22] 

def parse_s_val(s_str):
    """Converts 's12.34' to float 12.34"""
    try:
        return float(s_str.replace('s', ''))
    except ValueError:
        return 0.0

def plot_n(df, n):
    prefix = f"{n:03d}_"
    subset = df[df['id'].str.startswith(prefix)]
    
    if len(subset) == 0:
        print(f"No entries found for N={n} in {SUBMISSION_FILE}")
        return

    print(f"Plotting N={n} ({len(subset)} trees)...")
    
    fig, ax = plt.subplots(figsize=(8,8))
    
    max_extent_x = 0
    max_extent_y = 0
    min_extent_x = 999
    min_extent_y = 999
    
    for _, row in subset.iterrows():
        x = parse_s_val(row['x'])
        y = parse_s_val(row['y'])
        deg = parse_s_val(row['deg'])
        
        t = TreePoly(0) 
        p = t.get_transformed_poly(x, y, deg)
        
        minx, miny, maxx, maxy = p.bounds
        max_extent_x = max(max_extent_x, maxx)
        max_extent_y = max(max_extent_y, maxy)
        min_extent_x = min(min_extent_x, minx)
        min_extent_y = min(min_extent_y, miny)
        
        px, py = p.exterior.xy
        ax.fill(px, py, alpha=0.7, fc='forestgreen', ec='black')
        
    score_side = max(max_extent_x, max_extent_y)
    
    rect = plt.Rectangle((0,0), score_side, score_side, 
                         linewidth=3, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    padding = 1.0
    limit = score_side + padding
    ax.set_xlim(-padding, limit)
    ax.set_ylim(-padding, limit)
    ax.set_aspect('equal')
    ax.set_title(f"N={n} | Max Side: {score_side:.4f}")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    try:
        df = pd.read_csv(SUBMISSION_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {SUBMISSION_FILE}. Run solve_all.py first.")
        sys.exit(1)

    if len(sys.argv) > 1:
        targets = [int(arg) for arg in sys.argv[1:]]
    else:
        targets = DEFAULT_VIEW_N

    for n in targets:
        plot_n(df, n)