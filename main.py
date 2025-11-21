import matplotlib.pyplot as plt
from optimizer import PackingSolver

# --- CONFIG ---
N_TREES = 10
BOX_SIZE = 20.0
ITERATIONS = 5000

def plot_solution(solver):
    polys = solver.get_polys_from_state(solver.state)
    fig, ax = plt.subplots(figsize=(8,8))
    
    rect = plt.Rectangle((0,0), solver.box_size, solver.box_size, 
                         linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    
    for p in polys:
        x, y = p.exterior.xy
        ax.fill(x, y, alpha=0.6, fc='forestgreen', ec='black')
        
    ax.set_xlim(-5, solver.box_size + 5)
    ax.set_ylim(-5, solver.box_size + 5)
    ax.set_aspect('equal')
    plt.show()

def generate_s_string(solver):
    print("id,x,y,deg")
    for i in range(solver.n):
        row_id = f"{solver.n:03d}_{i}"
        x = f"s{solver.state[i,0]:.6f}"
        y = f"s{solver.state[i,1]:.6f}"
        deg = f"s{solver.state[i,2]:.6f}"
        print(f"{row_id},{x},{y},{deg}")

if __name__ == "__main__":
    solver = PackingSolver(N_TREES, BOX_SIZE)
    final_cost = solver.solve(ITERATIONS)
    
    print(f"Final Cost: {final_cost:.4f}")
    plot_solution(solver)
    generate_s_string(solver)