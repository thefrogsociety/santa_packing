import matplotlib.pyplot as plt
from geometry import TreePoly

def test_plot():
    # Create one tree at (0,0) with 0 rotation
    tree = TreePoly(1)
    p = tree.get_transformed_poly(0, 0, 0)
    
    x, y = p.exterior.xy
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.fill(x, y, color='forestgreen', ec='black')
    ax.set_aspect('equal')
    ax.set_title("Official Competition Tree Shape")
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    test_plot()