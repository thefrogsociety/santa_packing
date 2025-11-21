import numpy as np
import os
from shapely.geometry import Polygon
from geometry import TreePoly
if os.environ.get('IN_WORKER') == '1':
    from tqdm import tqdm as tqdm_base
    def tqdm(*args, **kwargs):
        return args[0]
else:
    from tqdm import tqdm as tqdm_base
    tqdm = tqdm_base

class PackingSolver:
    def __init__(self, n_trees, box_size, compression_weight=100.0, prior_state=None, prior_box_size=None, init_mode='random', hole_pull_coords=None):
        self.n = n_trees
        self.box_size = box_size
        self.compression_weight = compression_weight
        self.trees = [TreePoly(i) for i in range(n_trees)]
        self.hole_pull_coords = hole_pull_coords 
        
        if prior_state is not None and prior_box_size is not None:
            self.state = prior_state.copy()
            scale_factor = box_size / prior_box_size
            self.state[:, 0] *= scale_factor
            self.state[:, 1] *= scale_factor
        else:
            if init_mode == 'lattice':
                self.state = self.initialize_lattice()
            else:
                self.state = np.random.rand(n_trees, 3)
                self.state[:, 0] = self.state[:, 0] * (self.box_size * 0.8) + (self.box_size * 0.1)
                self.state[:, 1] = self.state[:, 1] * (self.box_size * 0.8) + (self.box_size * 0.1)
                self.state[:, 2] *= 360 # Rotation
        
        self.polys = []
        self.all_bounds = np.zeros((self.n, 4)) 
        
        for i in range(self.n):
            p = self.trees[i].get_transformed_poly(self.state[i,0], self.state[i,1], self.state[i,2])
            self.polys.append(p)
            self.all_bounds[i] = p.bounds
        
        self.current_overlap, self.current_boundary = self.calculate_full_cost(check_polys=False)
        self.current_compression = self.get_compression_cost()
        self.current_total = self.current_overlap + self.current_boundary + self.current_compression

    def initialize_lattice(self):
        """Generates a dense triangular lattice (Zipper pattern)."""
        state = np.zeros((self.n, 3))
        dx = 0.5 
        dy = 0.9
        cols = int(np.sqrt(self.n * (dy/dx))) + 1
        
        for i in range(self.n):
            row = i // cols
            col = i % cols
            x = col * dx + (0.0 if row % 2 == 0 else 0.25)
            y = row * dy
            rot = 0.0 
            x = x + 1.0
            y = y + 1.0
            x = x % self.box_size
            y = y % self.box_size
            state[i] = [x, y, rot]
        return state

    def get_individual_cost(self, idx, poly_override=None, bounds_override=None):
        """Calculates boundary cost and overlap cost for one tree against all others."""
        target_poly = poly_override if poly_override else self.polys[idx]
        minx, miny, maxx, maxy = bounds_override if bounds_override is not None else self.all_bounds[idx]
        
        b_cost = 0.0
        if minx < 0: b_cost += abs(minx) * 5000 
        if miny < 0: b_cost += abs(miny) * 5000
        if maxx > self.box_size: b_cost += (maxx - self.box_size) * 5000
        if maxy > self.box_size: b_cost += (maxy - self.box_size) * 5000
        

        candidates_mask = (
            (self.all_bounds[:, 2] >= minx) & (self.all_bounds[:, 0] <= maxx) &
            (self.all_bounds[:, 3] >= miny) & (self.all_bounds[:, 1] <= maxy)
        )
        candidates_mask[idx] = False
        candidate_indices = np.where(candidates_mask)[0]
        
        o_cost = 0.0
        for j in candidate_indices:
            other = self.polys[j]
            if target_poly.intersects(other):
                o_cost += target_poly.intersection(other).area * 100000 
                
        return o_cost, b_cost

    def calculate_full_cost(self, check_polys=True):
        """Calculates the overlap and boundary cost for the entire packing (O(N^2))."""
        total_overlap = 0.0
        total_boundary = 0.0
        
        if check_polys:
            for i in range(self.n):
                 self.update_poly(i)
        
        for i in range(self.n):
            o, b = self.get_individual_cost(i) 
            total_boundary += b
            total_overlap += o 
            
        return (total_overlap / 2), total_boundary

    def get_compression_cost(self):
        """Calculates the cost based on the packing's overall bounding box."""
        max_x = np.max(self.all_bounds[:, 2])
        max_y = np.max(self.all_bounds[:, 3])
        

        total_comp = max(max_x, max_y) * self.compression_weight 
        
        total_comp += (max_x * 50.0) + (max_y * 50.0) 
        
        return total_comp

    def get_pull_penalty(self):
        """Calculates the hole pull penalty."""
        if self.hole_pull_coords is not None:
            hx, hy = self.hole_pull_coords
            pull_penalty = 0.0
            
            for tree_state in self.state:
                dist_sq = (tree_state[0] - hx)**2 + (tree_state[1] - hy)**2
                pull_penalty += dist_sq
            
            # Apply a moderate weight
            return pull_penalty * 5.0
        return 0.0

    def solve(self, iterations=5000):
        
        temperature = 1.0
        cooling_rate = 0.001**(1/iterations)
        
        pbar = tqdm(range(iterations), desc=f"Box {self.box_size:.2f}", leave=False, disable=(os.environ.get('IN_WORKER') == '1'))
        
        for _ in pbar:
            if np.random.rand() < 0.01 and self.n > 1:
                idx1, idx2 = np.random.choice(self.n, 2, replace=False)
                
                old_state1 = self.state[idx1].copy()
                old_state2 = self.state[idx2].copy()
                old_poly1 = self.polys[idx1]
                old_poly2 = self.polys[idx2]
                old_bounds1 = self.all_bounds[idx1].copy()
                old_bounds2 = self.all_bounds[idx2].copy()
                
                self.state[idx1, :2] = old_state2[:2]
                self.state[idx2, :2] = old_state1[:2]
                
                self.update_poly(idx1)
                self.update_poly(idx2)
                
                new_overlap, new_boundary = self.calculate_full_cost(check_polys=False)
                new_compression = self.get_compression_cost()
                new_total = new_overlap + new_boundary + new_compression + self.get_pull_penalty()
                
                if new_total < self.current_total:
                    self.current_total = new_total
                    self.current_overlap = new_overlap
                    self.current_boundary = new_boundary
                    self.current_compression = new_compression
                else:
                    delta = new_total - self.current_total
                    prob = np.exp(-delta / (temperature + 1e-9))
                    if np.random.rand() < prob:
                        self.current_total = new_total
                        self.current_overlap = new_overlap
                        self.current_boundary = new_boundary
                        self.current_compression = new_compression
                    else:
                        self.state[idx1] = old_state1
                        self.state[idx2] = old_state2
                        self.polys[idx1] = old_poly1
                        self.polys[idx2] = old_poly2
                        self.all_bounds[idx1] = old_bounds1
                        self.all_bounds[idx2] = old_bounds2
                
                continue

            idx = np.random.randint(0, self.n)
            
            old_state_row = self.state[idx].copy()
            old_poly = self.polys[idx]
            old_bounds = self.all_bounds[idx].copy()
            
            old_o, old_b = self.get_individual_cost(idx, old_poly, old_bounds)
            
            move_scale = 0.5 * temperature 
            rot_scale = 45 * temperature
            
            self.state[idx, 0] += np.random.uniform(-move_scale, move_scale)
            self.state[idx, 1] += np.random.uniform(-move_scale, move_scale)
            self.state[idx, 2] += np.random.uniform(-rot_scale, rot_scale)
            
            self.state[idx, 0] = np.clip(self.state[idx, 0], -1, self.box_size + 1)
            self.state[idx, 1] = np.clip(self.state[idx, 1], -1, self.box_size + 1)
            
            new_coords = self.trees[idx].get_transformed_coords(
                self.state[idx,0], self.state[idx,1], self.state[idx,2]
            )
            minx, maxx = np.min(new_coords[:,0]), np.max(new_coords[:,0])
            miny, maxy = np.min(new_coords[:,1]), np.max(new_coords[:,1])
            new_bounds = (minx, miny, maxx, maxy)
            new_poly = Polygon(new_coords) 
            
            new_o, new_b = self.get_individual_cost(idx, new_poly, new_bounds)
            

            delta_local = (new_o + new_b) - (old_o + old_b)
            
            old_max_x = np.max(self.all_bounds[:, 2])
            old_max_y = np.max(self.all_bounds[:, 3])
            
            self.all_bounds[idx] = new_bounds 
            
            new_max_x = np.max(self.all_bounds[:, 2])
            new_max_y = np.max(self.all_bounds[:, 3])
            
            old_compression = self.get_compression_cost()
            
            self.polys[idx] = new_poly 

            new_compression = self.get_compression_cost()
            delta_compression = new_compression - old_compression
            

            new_pull_penalty = self.get_pull_penalty()
            old_pull_penalty = self.current_total - (self.current_overlap + self.current_boundary + self.current_compression)
            delta_pull = new_pull_penalty - old_pull_penalty
            
            delta_total = delta_local + delta_compression + delta_pull
            
            new_total = self.current_total + delta_total
            
            
            if new_total < self.current_total:
                self.current_total = new_total
                self.current_overlap += (new_o - old_o)
                self.current_boundary += (new_b - old_b)
                self.current_compression = new_compression 
            else:
                delta = new_total - self.current_total
                prob = np.exp(-delta / (temperature + 1e-9))
                if np.random.rand() < prob:
                    self.current_total = new_total
                    self.current_overlap += (new_o - old_o)
                    self.current_boundary += (new_b - old_b)
                    self.current_compression = new_compression
                else:
                    self.state[idx] = old_state_row
                    self.polys[idx] = old_poly
                    self.all_bounds[idx] = old_bounds
            
            temperature *= cooling_rate
            
            pbar.set_postfix({'Total': f'{self.current_total:.2f}', 'Comp': f'{self.current_compression:.2f}', 'T': f'{temperature:.4f}'})
            
        return self.current_total

    def update_poly(self, idx):
        """Updates the polygon and bounds cache for a single tree."""
        p = self.trees[idx].get_transformed_poly(
            self.state[idx, 0], self.state[idx, 1], self.state[idx, 2]
        )
        self.polys[idx] = p
        self.all_bounds[idx] = p.bounds

    def check_valid(self):
        """Final check for validity (no overlap, fully contained in box)."""
        
        polys = []
        bounds_cache = np.zeros((self.n, 4))
        for i in range(self.n):
             p = self.trees[i].get_transformed_poly(self.state[i,0], self.state[i,1], self.state[i,2])
             polys.append(p)
             bounds_cache[i] = p.bounds
             
        for p in polys:
            minx, miny, maxx, maxy = p.bounds
            if minx < -1e-6 or miny < -1e-6 or maxx > self.box_size + 1e-6 or maxy > self.box_size + 1e-6: 
                return False
                
        for i in range(self.n):
            for j in range(i + 1, self.n):

                if bounds_cache[i, 2] < bounds_cache[j, 0] or bounds_cache[i, 0] > bounds_cache[j, 2] or \
                   bounds_cache[i, 3] < bounds_cache[j, 1] or bounds_cache[i, 1] > bounds_cache[j, 3]: 
                    continue
                
                if polys[i].intersects(polys[j]): 
                    if polys[i].intersection(polys[j]).area > 1e-8:
                         return False
                         
        return True