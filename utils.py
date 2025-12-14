import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class FastBoard:
    def __init__(self, N, board=None):
        self.N = N
        self.energies = []
        
        if board is not None:
            self.board = board.copy()
        else:
            self.board = np.random.randint(0, N, size=(N, N))
            #self.board = np.zeros((N, N), dtype=int)
        self.offset = N
        self.counters = {
            'row_x': np.zeros((N, N), dtype=int),
            'row_y': np.zeros((N, N), dtype=int),
            'diag_xy_sum': np.zeros((N, 2*N), dtype=int),
            'diag_xy_diff': np.zeros((N, 2*N), dtype=int),
            'diag_xz_sum': np.zeros((N, 2*N), dtype=int),
            'diag_xz_diff': np.zeros((N, 2*N), dtype=int),
            'diag_yz_sum': np.zeros((N, 2*N), dtype=int),
            'diag_yz_diff': np.zeros((N, 2*N), dtype=int),
            'space_1': np.zeros((2*N, 2*N), dtype=int),
            'space_2': np.zeros((2*N, 2*N), dtype=int),
            'space_3': np.zeros((2*N, 2*N), dtype=int),
            'space_4': np.zeros((2*N, 2*N), dtype=int),
        }
        
        self.current_energy = 0
        self._compute_initial_energy()
        self.energies.append(self.current_energy)

    def convert(self):
        """Converts internal board representation to list of (x,y,z) coords."""
        coords = []
        for x in range(self.N):
            for y in range(self.N):
                z = self.board[x,y]
                coords.append((int(x), int(y), int(z)))
        return coords


    def step(self, temperature, n=1):
        """
        Performs n Metropolis-Hastings steps.
        """
        if isinstance(temperature, (int, float)):
            temps = [temperature] * n
        else:
            temps = temperature

        for i in range(n):
            x = np.random.randint(self.N)
            y = np.random.randint(self.N)
            current_z = self.board[x, y]
            new_z = np.random.randint(self.N)

            if current_z == new_z:
                self.energies.append(self.current_energy)
                continue

            dE_remove = self._get_energy_change(x, y, current_z, -1) 
            dE = self._calculate_transition_delta(x, y, current_z, new_z)


            if dE < 0 or np.random.rand() < np.exp(-dE / temps[i]):
                self._update_counters(x, y, current_z, -1)
                self._update_counters(x, y, new_z, 1)
                self.board[x, y] = new_z
                self.current_energy += dE

            self.energies.append(self.current_energy)

    def _calculate_transition_delta(self, x, y, old_z, new_z):
        """
        Calculates dE without modifying state.
        dE = (Cost of new pos) - (Cost of old pos)
        """
        loss = self._sum_conflicts(x, y, old_z) - 12
        gain = self._sum_conflicts(x, y, new_z)
        return gain - loss

    def _sum_conflicts(self, x, y, z):
        """Returns total queens sharing lines with (x,y,z) from current counters."""
        off = self.offset
        total = 0
        total += self.counters['row_x'][y, z]
        total += self.counters['row_y'][x, z]
        total += self.counters['diag_xy_sum'][z, x + y]
        total += self.counters['diag_xy_diff'][z, x - y + off]
        total += self.counters['diag_xz_sum'][y, x + z]
        total += self.counters['diag_xz_diff'][y, x - z + off]
        total += self.counters['diag_yz_sum'][x, y + z]
        total += self.counters['diag_yz_diff'][x, y - z + off]
        total += self.counters['space_1'][x - y + off, y - z + off]
        total += self.counters['space_2'][x - y + off, y + z]
        total += self.counters['space_3'][x + y, y - z + off]
        total += self.counters['space_4'][x + y, y + z]
        return total

    def _update_counters(self, x, y, z, delta):
        """Updates internal counters when a queen is added (+1) or removed (-1)."""
        off = self.offset
        self.counters['row_x'][y, z] += delta
        self.counters['row_y'][x, z] += delta
        self.counters['diag_xy_sum'][z, x + y] += delta
        self.counters['diag_xy_diff'][z, x - y + off] += delta
        self.counters['diag_xz_sum'][y, x + z] += delta
        self.counters['diag_xz_diff'][y, x - z + off] += delta
        self.counters['diag_yz_sum'][x, y + z] += delta
        self.counters['diag_yz_diff'][x, y - z + off] += delta
        self.counters['space_1'][x - y + off, y - z + off] += delta
        self.counters['space_2'][x - y + off, y + z] += delta
        self.counters['space_3'][x + y, y - z + off] += delta
        self.counters['space_4'][x + y, y + z] += delta

    def _compute_initial_energy(self):
        """One-time expensive setup (O(N^2))."""
        self.current_energy = 0
        for key in self.counters:
            self.counters[key].fill(0)

        for x in range(self.N):
            for y in range(self.N):
                z = self.board[x, y]
                self.current_energy += self._sum_conflicts(x, y, z)
                self._update_counters(x, y, z, 1)
    
    def _get_energy_change(self, x, y, z, delta):
        """Helper to compute energy delta if we add/remove queen at x,y,z."""
        count = self._sum_conflicts(x, y, z)
        if delta == -1: return -(count - 12)
        if delta == 1: return count
        return 0

    def get_3d_board(self):
        """Returns the full NxNxN boolean board for visualization."""
        b = np.zeros((self.N, self.N, self.N), dtype=int)
        for x in range(self.N):
            for y in range(self.N):
                b[x, y, self.board[x, y]] = 1
        return b

def average_energy_fixed_T(N, steps, runs=5, temperature=0.5):
    """
    For a fixed N, represent the energy as a function of time.
    The energy is averaged across multiple runs (no simulated annealing).
    """

    all_energies = []

    for r in tqdm(range(runs)):
        board = FastBoard(N)
        board.step(temperature, n=steps)
        all_energies.append(board.energies)

    avg_energy = np.mean(np.array(all_energies), axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(avg_energy, label=f"Average Energy (T={temperature})")
    plt.xlabel("Step")
    plt.ylabel("Energy (Collisions)")
    plt.title(f"Average Energy vs Time for N={N} ({runs} runs)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return avg_energy

def run_ensemble(N, steps, temp_schedule, runs=5):
    """
    Runs the simulation 'runs' times and returns the average energy history.
    """
    all_energies = []
    
    for r in tqdm(range(runs)):
        board = FastBoard(N)
        board.step(temp_schedule, n=steps)
        all_energies.append(board.energies)
        
    # Average across runs (axis 0)
    avg_energies = np.mean(np.array(all_energies), axis=0)
    return avg_energies



def make_linear_schedule(T0, T1, steps):
    return np.linspace(T0, T1, steps)

def make_polynomial_schedule(T0, T1, steps, power=2):
    x = np.linspace(0, 1, steps)
    return T0 + (T1 - T0) * (x ** power)

def make_exponential_schedule(T0, T1, steps):
    return T0 * (T1 / T0) ** (np.linspace(0, 1, steps))

def make_batch_schedule(T0, T1, steps, batches=5):
    temps = []
    batch_size = steps // batches
    delta = (T0 - T1) / batches
    for i in range(batches):
        Ti = T0 - i * delta
        temps += [Ti] * batch_size
    # Complete remaining steps in case of integer mismatch
    while len(temps) < steps:
        temps.append(T1)
    return np.array(temps)



def plot_multiple_schedules(N, steps, runs=10):
    """Compare the 5 methods""" 





    schedules = {  #Bests parameters found with tuning
        "Constant (T=0.7)": 0.7,  
        "Linear": make_linear_schedule(0.8, 0.4, steps),
        "Polynomial (power=2)": make_polynomial_schedule(0.7, 0.1, steps, power=2),
        "Exponential": make_exponential_schedule(0.8, 0.4, steps),
        "piecewise constant(as seen in course)": make_batch_schedule(0.7, 0.3, steps, batches=5),
    }

    plt.figure(figsize=(12, 7))

    for name, schedule in schedules.items():
        print(f"Simulating → {name}")
        energies = run_ensemble(N, steps, schedule, runs=runs)
        plt.plot(energies[1000:], label=name)

    plt.title(f"Comparison of Best Cooling Schedules starting from 1000 steps for N={N}")
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



def evaluate_schedule(N, steps, schedule, trials=5):

    energies = run_ensemble(N, steps, schedule, runs=trials)
    return np.min(energies)

def tune_constant(N, steps, temps, trials=5):
    best = (float('inf'), None)
    for T in temps:
        score = evaluate_schedule(N, steps, T, trials)
        if score < best[0]:
            best = (score, T)
    return best

def tune_linear(N, steps, T0_values, T1_values, trials=5):
    best = (float('inf'), None)
    for T0 in T0_values:
        for T1 in T1_values:
            sched = make_linear_schedule(T0, T1, steps)
            score = evaluate_schedule(N, steps, sched, trials)
            if score < best[0]:
                best = (score, (T0, T1))
    return best

def tune_polynomial(N, steps, T0_values, T1_values, powers, trials=5):
    best = (float('inf'), None)
    for T0 in T0_values:
        for T1 in T1_values:
            for p in powers:
                sched = make_polynomial_schedule(T0, T1, steps, power=p)
                score = evaluate_schedule(N, steps, sched, trials)
                if score < best[0]:
                    best = (score, (T0, T1, p))
    return best

def tune_exponential(N, steps, T0_values, T1_values, trials=5):
    best = (float('inf'), None)
    for T0 in T0_values:
        for T1 in T1_values:
            sched = make_exponential_schedule(T0, T1, steps)
            score = evaluate_schedule(N, steps, sched, trials)
            if score < best[0]:
                best = (score, (T0, T1))
    return best

def tune_batch(N, steps, T0_values, T1_values, batches_values, trials=5):
    best = (float('inf'), None)
    for T0 in T0_values:
        for T1 in T1_values:
            for b in batches_values:
                sched = make_batch_schedule(T0, T1, steps, batches=b)
                score = evaluate_schedule(N, steps, sched, trials)
                if score < best[0]:
                    best = (score, (T0, T1, b))
    return best

def tune_all(N=11, steps=200000, trials=5):

    temps = np.linspace(0.1, 3, 30)   #Parameters to try and tune
    T0_values = np.linspace(0.5, 2, 16)
    T1_values = np.array([0.1, 0.2, 0.3, 0.4])
    powers = [2, 3, 4]
    batch_values = [5]


    print("\n=== Tuning Constant ===")
    best_const = tune_constant(N, steps, temps, trials)
    print("Constant:", best_const)

    print("\n=== Tuning Batch ===")
    best_batch = tune_batch(N, steps, T0_values, T1_values, batch_values, trials)
    print("Batch:", best_batch)

    print("\n=== Tuning Linear ===")
    best_linear = tune_linear(N, steps, T0_values, T1_values, trials)
    print("Linear:", best_linear)

    print("\n=== Tuning Polynomial ===")
    best_poly = tune_polynomial(N, steps, T0_values, T1_values, powers, trials)
    print("Polynomial:", best_poly)

    print("\n=== Tuning Exponential ===")
    best_expo = tune_exponential(N, steps, T0_values, T1_values, trials)
    print("Exponential:", best_expo)
    



def compare_N_all_methods(n_values, steps, repeats=10):
    """
    Tasks 3 & 4: Minimum energy vs. N for ALL annealing methods.
    For each N, test all methods and keep the best result.
    """
    
    results = {}
    methods = ['constant', 'linear', 'polynomial', 'exponential', 'batch']
    
    for N in tqdm(n_values, desc="Testing N sizes"):
        best_e_for_N = float('inf')
        best_method_for_N = None

        for method in methods:
            if method == 'constant':
                t_schedule = 0.7
            elif method == 'linear':
                t_schedule = make_linear_schedule(0.8, 0.4, steps)
            elif method == 'polynomial':
                t_schedule = make_polynomial_schedule(0.7, 0.1, steps, power=2)
            elif method == 'exponential':
                t_schedule = make_exponential_schedule(0.8, 0.4, steps)
            elif method == 'batch':
                t_schedule = make_batch_schedule(0.7, 0.3, steps, batches=5)
            
            for _ in range(repeats):
                board = FastBoard(N)
                board.step(t_schedule, n=steps)
                current_min = min(board.energies)
                
                if current_min < best_e_for_N:
                    best_e_for_N = current_min
                    best_method_for_N = method
                
                # If perfect solution, stop
                if best_e_for_N == 0:
                    solution = board.convert()
                    print(f"Perfect solution found for N = {N} with method {method}: {solution}")
                    break
            
            if best_e_for_N == 0:
                break
        
        results[N] = {
            'best_energy': best_e_for_N,
            'best_method': best_method_for_N
        }
        print(f"  N={N}, Best Energy Found: {best_e_for_N} (method: {best_method_for_N})")
    
    n_values_sorted = sorted(results.keys())
    min_energies = [results[N]['best_energy'] for N in n_values_sorted]
    best_methods = [results[N]['best_method'] for N in n_values_sorted]
    
    for i, N in enumerate(n_values_sorted):
        print(f"N={N}: Minimum energy = {min_energies[i]} (method: {best_methods[i]})")
    
    zeros = [N for N, e in zip(n_values_sorted, min_energies) if e == 0]
    if zeros:
        print(f"\nPerfect solutions found for N = {zeros}")
    
    plt.figure(figsize=(12, 7))
    
    method_colors = {
        'constant': 'blue',
        'linear': 'red',
        'polynomial': 'green',
        'exponential': 'orange',
        'batch': 'purple'
    }
    
    for i, (N, energy, method) in enumerate(zip(n_values_sorted, min_energies, best_methods)):
        plt.scatter(N, energy, color=method_colors[method], s=100, alpha=0.7, 
                   edgecolors='black', linewidth=1)
    
    plt.plot(n_values_sorted, min_energies, linestyle='-', color='gray', alpha=0.5)
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=method_colors[method], 
                             markersize=10, label=method)
                      for method in method_colors]
    plt.legend(handles=legend_elements, title="Methods", loc='upper left')
    
    plt.title(f"Minimum Energy Achieved vs. Board Size N\n(best method among all)")
    plt.xlabel("N (Board Size NxNxN)")
    plt.ylabel("Minimum Energy Found")
    plt.xticks(n_values_sorted)
    plt.grid(True, alpha=0.3)
    
    for N, energy in zip(n_values_sorted, min_energies):
        if energy == 0:
            plt.annotate('✓', xy=(N, energy), xytext=(0, 10),
                        textcoords='offset points', ha='center',
                        fontsize=12, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return results