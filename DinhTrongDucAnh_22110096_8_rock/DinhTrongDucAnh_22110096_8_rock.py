import tkinter as tk
from tkinter import ttk
from collections import deque
import heapq
import math
import random


BOARD_SIZE = 8

ALGORITHMS = [
    'BFS',
    'DFS',
    'UCS',
    'IDS',
    'Greedy',
    'A*',
    'Hill-Climbing',
    'Simulated Annealing',
    'Beam Search',
    'Genetic Algorithm',
    'Backtracking',
    'Forward Checking',
    'AC-3',
    'Minimax',
    'Alpha-Beta',
]


class EightRooksApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("8 XE - 15 THUẬT TOÁN")
        self.window.geometry("950x620")
        self.window.configure(bg='lightgray')

        self.solutions = []
        self.solution_scores = []
        self.current_solution_index = 0

        self.create_interface()

    def create_interface(self):
        tk.Label(self.window, text="Bài toán 8 Xe", font=("Arial", 16, "bold"), bg='lightgray').pack(pady=8)

        main = tk.Frame(self.window, bg='lightgray')
        main.pack(pady=8, fill=tk.BOTH, expand=True)

        left = tk.Frame(main, bg='lightgray')
        left.pack(side=tk.LEFT, padx=16, fill=tk.BOTH, expand=True)

        tk.Label(left, text="Nhật ký thuật toán", font=("Arial", 12, "bold"), bg='lightgray').pack()

        log_container = tk.Frame(left, bg='lightgray')
        log_container.pack(pady=6, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(log_container, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text = tk.Text(
            log_container,
            width=38,
            height=28,
            state='disabled',
            bg='#f0f0f0',
            wrap='word',
            yscrollcommand=scrollbar.set,
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)

        right = tk.Frame(main, bg='lightgray')
        right.pack(side=tk.RIGHT, padx=16, fill=tk.Y)
        tk.Label(right, text="Lời giải", font=("Arial", 12, "bold"), bg='lightgray').pack()

        self.solution_info_label = tk.Label(right, text="Chưa tìm", bg='lightgray', fg='green')
        self.solution_info_label.pack()
        self.solution_score_label = tk.Label(right, text="Điểm: 0", bg='lightgray', fg='blue')
        self.solution_score_label.pack()

        sol_board = tk.Frame(right, bg='lightgray')
        sol_board.pack(pady=4)
        self.solution_buttons = []
        for i in range(BOARD_SIZE):
            row_buttons = []
            for j in range(BOARD_SIZE):
                color = 'white' if (i + j) % 2 == 0 else 'gray'
                btn = tk.Button(sol_board, width=4, height=2, bg=color, font=("Arial", 16), state='disabled')
                btn.grid(row=i, column=j, padx=1, pady=1)
                row_buttons.append(btn)
            self.solution_buttons.append(row_buttons)

        nav = tk.Frame(right, bg='lightgray')
        nav.pack(pady=6)
        self.prev_button = tk.Button(nav, text="Trước", command=self.prev_solution, state='disabled')
        self.prev_button.pack(side=tk.LEFT, padx=4)
        self.next_button = tk.Button(nav, text="Tiếp", command=self.next_solution, state='disabled')
        self.next_button.pack(side=tk.LEFT, padx=4)

        algo_frame = tk.Frame(right, bg='lightgray')
        algo_frame.pack(pady=6, fill=tk.X)
        tk.Label(algo_frame, text="Thuật toán:", bg='lightgray').pack(side=tk.LEFT)

        self.algo_var = tk.StringVar(value=ALGORITHMS[0])
        algo_dropdown = ttk.Combobox(algo_frame, textvariable=self.algo_var, values=ALGORITHMS, state='readonly', width=25)
        algo_dropdown.pack(side=tk.LEFT, padx=6)

        tk.Button(algo_frame, text="Tìm lời giải", bg='orange', command=self.find_solutions).pack(side=tk.LEFT, padx=6)

        sort_frame = tk.Frame(right, bg='lightgray')
        sort_frame.pack(pady=4)
        tk.Label(sort_frame, text="Sắp xếp:", bg='lightgray').pack(side=tk.LEFT)
        self.sort_var = tk.StringVar(value='Thứ tự tìm thấy')
        sort_dropdown = ttk.Combobox(
            sort_frame,
            textvariable=self.sort_var,
            values=['Thứ tự tìm thấy', 'Điểm thấp', 'Điểm cao'],
            state='readonly',
            width=18,
        )
        sort_dropdown.pack(side=tk.LEFT, padx=4)
        sort_dropdown.bind('<<ComboboxSelected>>', self.sort_solutions)

        formula_frame = tk.Frame(right, bg='lightgray', bd=2, relief=tk.GROOVE)
        formula_frame.pack(pady=6, fill=tk.X)
        tk.Label(
            formula_frame,
            text="Công thức điểm:",
            bg='lightgray',
            font=("Arial", 10, "bold"),
        ).pack(anchor='w', padx=6, pady=(4, 0))
        tk.Label(
            formula_frame,
            text="Điểm = Σ (1 + |hàng - cột|) cho từng quân xe",
            bg='lightgray',
            font=("Arial", 10),
        ).pack(anchor='w', padx=6, pady=(0, 4))

        self.log("Ứng dụng 8 Xe đã sẵn sàng.")

    # --- util ---
    def is_safe(self, assign, row, col):
        for r, c in enumerate(assign):
            if c == col:
                return False
        return True

    def is_safe_map(self, assign_map, row, col):
        for r, c in assign_map.items():
            if c == col:
                return False
        return True

    def assignment_to_board(self, assign):
        board = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for row, col in enumerate(assign):
            board[row][col] = True
        return board

    def assignment_cost(self, assign):
        return round(sum(self.position_cost(r, c) for r, c in enumerate(assign)), 2)

    def position_cost(self, row, col):
        return 1 + abs(row - col)

    def count_conflicts(self, assign):
        conflicts = 0
        for i in range(len(assign)):
            for j in range(i + 1, len(assign)):
                if assign[i] == assign[j]:
                    conflicts += 1
        return conflicts

    def map_to_assignment(self, assignment_map):
        return tuple(assignment_map[row] for row in range(BOARD_SIZE))

    # --- algorithms ---
    def bfs_search(self, max_solutions=20):
        queue = deque([()])
        results = []
        while queue and len(results) < max_solutions:
            assign = queue.popleft()
            if len(assign) == BOARD_SIZE:
                results.append(assign)
                continue
            row = len(assign)
            for col in range(BOARD_SIZE):
                if self.is_safe(assign, row, col):
                    queue.append(assign + (col,))
        return results

    def dfs_search(self, max_solutions=20):
        stack = [()]
        results = []
        while stack and len(results) < max_solutions:
            assign = stack.pop()
            if len(assign) == BOARD_SIZE:
                results.append(assign)
                continue
            row = len(assign)
            for col in reversed(range(BOARD_SIZE)):
                if self.is_safe(assign, row, col):
                    stack.append(assign + (col,))
        return results

    def ucs_search(self, max_solutions=20):
        heap = []
        heapq.heappush(heap, (0.0, ()))
        seen = {}
        results = []
        while heap and len(results) < max_solutions:
            cost, assign = heapq.heappop(heap)
            if len(assign) == BOARD_SIZE:
                results.append(assign)
                continue
            row = len(assign)
            for col in range(BOARD_SIZE):
                if not self.is_safe(assign, row, col):
                    continue
                new_assign = assign + (col,)
                step_cost = self.position_cost(row, col)
                new_cost = round(cost + step_cost, 2)
                if seen.get(new_assign, float('inf')) <= new_cost:
                    continue
                seen[new_assign] = new_cost
                heapq.heappush(heap, (new_cost, new_assign))
        return results

    def ids_search(self, max_depth=BOARD_SIZE, max_solutions=20):
        results = []

        def dls(assign, depth_limit):
            if len(results) >= max_solutions:
                return
            if len(assign) == BOARD_SIZE:
                results.append(assign)
                return
            if len(assign) >= depth_limit:
                return
            row = len(assign)
            for col in range(BOARD_SIZE):
                if self.is_safe(assign, row, col):
                    dls(assign + (col,), depth_limit)

        for depth in range(1, max_depth + 1):
            dls((), depth)
            if len(results) >= max_solutions:
                break
        return results

    def greedy_search(self, max_solutions=20):
        heap = []
        heapq.heappush(heap, (0, ()))
        results = []
        visited = set()
        while heap and len(results) < max_solutions:
            priority, assign = heapq.heappop(heap)
            if len(assign) == BOARD_SIZE:
                if assign not in results:
                    results.append(assign)
                continue
            row = len(assign)
            for col in range(BOARD_SIZE):
                if not self.is_safe(assign, row, col):
                    continue
                new_assign = assign + (col,)
                if new_assign in visited:
                    continue
                visited.add(new_assign)
                next_row = len(new_assign)
                safe_next = sum(1 for c in range(BOARD_SIZE) if next_row < BOARD_SIZE and self.is_safe(new_assign, next_row, c))
                heapq.heappush(heap, (-safe_next, new_assign))
        return results

    def a_star_search(self, max_solutions=20):
        heap = []
        heapq.heappush(heap, (0.0, 0.0, ()))
        results = []
        seen = {}
        while heap and len(results) < max_solutions:
            f_cost, g_cost, assign = heapq.heappop(heap)
            if len(assign) == BOARD_SIZE:
                results.append(assign)
                continue
            row = len(assign)
            for col in range(BOARD_SIZE):
                if not self.is_safe(assign, row, col):
                    continue
                new_assign = assign + (col,)
                step_cost = self.position_cost(row, col)
                g_new = round(g_cost + step_cost, 2)
                h_new = BOARD_SIZE - len(new_assign)
                f_new = g_new + h_new
                if seen.get(new_assign, float('inf')) <= g_new:
                    continue
                seen[new_assign] = g_new
                heapq.heappush(heap, (f_new, g_new, new_assign))
        return results

    def hill_climbing(self, restarts=25):
        best_assignments = []
        for _ in range(restarts):
            current = [random.randint(0, BOARD_SIZE - 1) for _ in range(BOARD_SIZE)]
            current_conflicts = self.count_conflicts(current)
            steps_without_improve = 0
            while current_conflicts > 0 and steps_without_improve < 100:
                neighbor = None
                neighbor_conflicts = current_conflicts
                for row in range(BOARD_SIZE):
                    original_col = current[row]
                    for col in range(BOARD_SIZE):
                        if col == original_col:
                            continue
                        current[row] = col
                        conflicts = self.count_conflicts(current)
                        if conflicts < neighbor_conflicts:
                            neighbor = list(current)
                            neighbor_conflicts = conflicts
                    current[row] = original_col
                if neighbor and neighbor_conflicts < current_conflicts:
                    current = neighbor
                    current_conflicts = neighbor_conflicts
                    steps_without_improve = 0
                else:
                    steps_without_improve += 1
            if current_conflicts == 0:
                best_assignments.append(tuple(current))
        return best_assignments

    def simulated_annealing(self, attempts=10):
        successes = []
        for _ in range(attempts):
            current = [random.randint(0, BOARD_SIZE - 1) for _ in range(BOARD_SIZE)]
            current_conflicts = self.count_conflicts(current)
            temperature = 10.0
            while temperature > 0.001 and current_conflicts > 0:
                row = random.randint(0, BOARD_SIZE - 1)
                new_col = random.randint(0, BOARD_SIZE - 1)
                old_col = current[row]
                current[row] = new_col
                new_conflicts = self.count_conflicts(current)
                delta = new_conflicts - current_conflicts
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_conflicts = new_conflicts
                else:
                    current[row] = old_col
                temperature *= 0.97
            if current_conflicts == 0:
                successes.append(tuple(current))
        return successes

    def beam_search(self, beam_width=10, max_solutions=20):
        beams = [()]
        for row in range(BOARD_SIZE):
            candidates = []
            for assign in beams:
                for col in range(BOARD_SIZE):
                    if self.is_safe(assign, row, col):
                        new_assign = assign + (col,)
                        heur = -self.count_conflicts(new_assign)
                        heur -= self.estimated_remaining_options(new_assign)
                        candidates.append((heur, new_assign))
            if not candidates:
                return self.bfs_search(max_solutions)
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = [assign for _, assign in candidates[:beam_width]]
        unique = []
        seen = set()
        for assign in beams:
            if assign not in seen:
                seen.add(assign)
                unique.append(assign)
            if len(unique) == max_solutions:
                break
        return unique

    def estimated_remaining_options(self, assign):
        row = len(assign)
        if row >= BOARD_SIZE:
            return 0
        safe = sum(1 for col in range(BOARD_SIZE) if self.is_safe(assign, row, col))
        return safe

    def genetic_algorithm(self, population_size=60, generations=250, mutation_rate=0.05):
        def random_individual():
            return [random.randint(0, BOARD_SIZE - 1) for _ in range(BOARD_SIZE)]

        def fitness(ind):
            return 1 / (1 + self.count_conflicts(ind))

        population = [random_individual() for _ in range(population_size)]
        for _ in range(generations):
            population.sort(key=lambda ind: self.count_conflicts(ind))
            if self.count_conflicts(population[0]) == 0:
                return [tuple(population[0])]
            total_fitness = sum(fitness(ind) for ind in population)
            if total_fitness == 0:
                population = [random_individual() for _ in range(population_size)]
                continue

            def select_parent():
                pick = random.random() * total_fitness
                current = 0
                for ind in population:
                    current += fitness(ind)
                    if current >= pick:
                        return ind
                return population[-1]

            new_population = []
            elite_count = max(1, population_size // 10)
            new_population.extend(population[:elite_count])
            while len(new_population) < population_size:
                parent1 = select_parent()
                parent2 = select_parent()
                cut = random.randint(1, BOARD_SIZE - 2)
                child = parent1[:cut] + parent2[cut:]
                if random.random() < mutation_rate:
                    row = random.randint(0, BOARD_SIZE - 1)
                    child[row] = random.randint(0, BOARD_SIZE - 1)
                new_population.append(child)
            population = new_population
        population.sort(key=lambda ind: self.count_conflicts(ind))
        if self.count_conflicts(population[0]) == 0:
            return [tuple(population[0])]
        return []

    def backtracking_search(self, max_solutions=20):
        results = []

        def backtrack(assign):
            if len(results) >= max_solutions:
                return
            if len(assign) == BOARD_SIZE:
                results.append(assign)
                return
            row = len(assign)
            for col in range(BOARD_SIZE):
                if self.is_safe(assign, row, col):
                    backtrack(assign + (col,))

        backtrack(())
        return results

    def forward_checking_search(self, max_solutions=20):
        domains = {row: set(range(BOARD_SIZE)) for row in range(BOARD_SIZE)}
        results = []

        def forward(assign, domains_local):
            if len(results) >= max_solutions:
                return
            row = len(assign)
            if row == BOARD_SIZE:
                results.append(assign)
                return
            available = sorted(domains_local[row])
            for col in available:
                if not self.is_safe(assign, row, col):
                    continue
                new_domains = {r: set(cols) for r, cols in domains_local.items()}
                for rr in range(row + 1, BOARD_SIZE):
                    if col in new_domains[rr]:
                        new_domains[rr].remove(col)
                    if not new_domains[rr]:
                        break
                else:
                    forward(assign + (col,), new_domains)

        forward((), domains)
        return results

    def ac3_search(self):
        domains = {row: set(range(BOARD_SIZE)) for row in range(BOARD_SIZE)}
        queue = deque((i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if i != j)

        def revise(xi, xj):
            revised = False
            to_remove = set()
            for x in domains[xi]:
                if not any(self.is_consistent_pair(xi, x, xj, y) for y in domains[xj]):
                    to_remove.add(x)
            if to_remove:
                domains[xi] -= to_remove
                revised = True
            return revised

        while queue:
            xi, xj = queue.popleft()
            if revise(xi, xj):
                if not domains[xi]:
                    return []
                for xk in range(BOARD_SIZE):
                    if xk != xi and xk != xj:
                        queue.append((xk, xi))

        return self.csp_backtracking(domains)

    def is_consistent_pair(self, row1, col1, row2, col2):
        return col1 != col2

    def csp_backtracking(self, domains):
        assignment = {}
        results = []

        def select_unassigned_var(domains_local, assignment_local):
            unassigned = [(len(domains_local[row]), row) for row in range(BOARD_SIZE) if row not in assignment_local]
            _, row = min(unassigned)
            return row

        def backtrack():
            if len(assignment) == BOARD_SIZE:
                results.append(self.map_to_assignment(assignment))
                return
            row = select_unassigned_var(domains, assignment)
            for col in sorted(domains[row]):
                if self.is_safe_map(assignment, row, col):
                    assignment[row] = col
                    backtrack()
                    del assignment[row]

        backtrack()
        return results

    def minimax_search(self):
        score, solution = self._minimax_recursive([], 0, True, None, None)
        if solution:
            return [solution]
        return []

    def alpha_beta_search(self):
        score, solution = self._minimax_recursive([], 0, True, -float('inf'), float('inf'))
        if solution:
            return [solution]
        return []

    def _minimax_recursive(self, assign, row, maximizing, alpha, beta):
        if row == BOARD_SIZE:
            return 0, tuple(assign)

        valid_cols = [col for col in range(BOARD_SIZE) if self.is_safe(assign, row, col)]
        if not valid_cols:
            remaining = BOARD_SIZE - row
            return -remaining, None

        if maximizing:
            best_score = -float('inf')
            best_solution = None
            for col in valid_cols:
                assign.append(col)
                score, solution = self._minimax_recursive(assign, row + 1, False, alpha, beta)
                assign.pop()
                if score > best_score:
                    best_score = score
                    best_solution = solution
                if alpha is not None:
                    alpha = max(alpha, best_score)
                    if beta is not None and alpha >= beta:
                        break
            return best_score, best_solution

        worst_score = float('inf')
        worst_solution = None
        for col in valid_cols:
            assign.append(col)
            score, solution = self._minimax_recursive(assign, row + 1, True, alpha, beta)
            assign.pop()
            if solution is None:
                continue
            if score < worst_score:
                worst_score = score
                worst_solution = solution
            if beta is not None:
                beta = min(beta, worst_score)
                if alpha is not None and beta <= alpha:
                    break
        if worst_solution is None:
            return -1, None
        return worst_score, worst_solution

    # --- gui helpers ---
    def find_solutions(self):
        self.solutions = []
        self.solution_scores = []
        algo = self.algo_var.get()
        self.solution_info_label.config(text='Đang tìm...')
        self.window.update_idletasks()

        handlers = {
            'BFS': lambda: self.bfs_search(),
            'DFS': lambda: self.dfs_search(),
            'UCS': lambda: self.ucs_search(),
            'IDS': lambda: self.ids_search(),
            'Greedy': lambda: self.greedy_search(),
            'A*': lambda: self.a_star_search(),
            'Hill-Climbing': lambda: self.hill_climbing(),
            'Simulated Annealing': lambda: self.simulated_annealing(),
            'Beam Search': lambda: self.beam_search(),
            'Genetic Algorithm': lambda: self.genetic_algorithm(),
            'Backtracking': lambda: self.backtracking_search(),
            'Forward Checking': lambda: self.forward_checking_search(),
            'AC-3': lambda: self.ac3_search(),
            'Minimax': lambda: self.minimax_search(),
            'Alpha-Beta': lambda: self.alpha_beta_search(),
        }

        assignments = handlers.get(algo, lambda: [])()

        unique_assignments = []
        seen = set()
        for assign in assignments:
            if assign is None:
                continue
            if len(assign) != BOARD_SIZE:
                continue
            if assign in seen:
                continue
            seen.add(assign)
            unique_assignments.append(assign)

        self.solutions = [self.assignment_to_board(assign) for assign in unique_assignments]
        self.solution_scores = [self.assignment_cost(assign) for assign in unique_assignments]

        chosen_index = None
        if self.solutions:
            best_score = min(self.solution_scores)
            chosen_index = random.randrange(len(self.solutions))
            sample_score = self.solution_scores[chosen_index]
            log_message = (
                f"[{algo}] Tìm được {len(self.solutions)} lời giải, điểm lệch chéo tốt nhất {best_score}. "
                f"Hiển thị ngẫu nhiên lời giải có điểm {sample_score}."
            )
        else:
            log_message = f"[{algo}] Không tìm thấy lời giải."
        self.log(log_message)

        if not self.solutions:
            self.solution_info_label.config(text='Không tìm thấy')
            self.solution_score_label.config(text='Điểm: 0')
            self.prev_button.config(state='disabled')
            self.next_button.config(state='disabled')
            self.clear_solution_board()
            return

        total = len(self.solutions)
        self.solution_info_label.config(text=f'Tìm được {total} lời giải')
        self.current_solution_index = chosen_index if chosen_index is not None else 0
        self.show_current_solution()

    def clear_solution_board(self):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                self.solution_buttons[i][j]['text'] = ''

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def sort_solutions(self, _event=None):
        if not self.solutions:
            return
        combined = list(zip(self.solutions, self.solution_scores))
        choice = self.sort_var.get()
        if choice == 'Điểm thấp':
            combined.sort(key=lambda x: x[1])
        elif choice == 'Điểm cao':
            combined.sort(key=lambda x: x[1], reverse=True)
        self.solutions, self.solution_scores = zip(*combined)
        self.solutions = list(self.solutions)
        self.solution_scores = list(self.solution_scores)
        self.current_solution_index = 0
        self.show_current_solution()

    def show_current_solution(self):
        if not self.solutions:
            return
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                self.solution_buttons[i][j]['text'] = ''
        board = self.solutions[self.current_solution_index]
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j]:
                    self.solution_buttons[i][j]['text'] = '♖'
                    self.solution_buttons[i][j].config(disabledforeground='blue')
        score = self.solution_scores[self.current_solution_index]
        self.solution_info_label.config(text=f'Lời giải {self.current_solution_index + 1}/{len(self.solutions)}')
        self.solution_score_label.config(text=f'Điểm: {score}')
        self.prev_button.config(state='normal' if self.current_solution_index > 0 else 'disabled')
        self.next_button.config(state='normal' if self.current_solution_index < len(self.solutions) - 1 else 'disabled')

    def prev_solution(self):
        if self.current_solution_index > 0:
            self.current_solution_index -= 1
            self.show_current_solution()

    def next_solution(self):
        if self.current_solution_index < len(self.solutions) - 1:
            self.current_solution_index += 1
            self.show_current_solution()

    def run(self):
        self.window.mainloop()


if __name__ == '__main__':
    app = EightRooksApp()
    app.run()
