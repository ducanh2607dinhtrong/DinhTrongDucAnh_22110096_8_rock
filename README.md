# Báo cáo bài tập cá nhân — Bài toán 8 Xe

Thông tin sinh viên

- Họ và tên: Đinh Trọng Đức Anh
- MSSV: 22110096
- Môn: Trí tuệ nhân tạo

Mục tiêu

- Tìm các cấu hình đặt 8 quân xe trên bàn 8x8 sao cho không có hai quân cùng cột (phiên bản đơn giản của bài 8 quân cờ), và so sánh nhiều thuật toán tìm kiếm trên bài toán này.

File chính: `DinhTrongDucAnh_22110096_8_rock.py`

Thuật toán đã triển khai (chi tiết)

- **BFS (Breadth-First Search)**: trạng thái biểu diễn bằng danh sách vị trí đã đặt trên từng hàng; mở rộng theo cột hợp lệ của hàng kế tiếp. BFS dùng hàng đợi nên luôn tìm được cấu hình đầu tiên với số hàng tối thiểu (tức đủ 8 xe) nhưng tiêu tốn bộ nhớ lớn vì phải lưu toàn bộ biên.

  ```python
  def bfs_search(self, max_solutions=20):
      queue = deque([()])
      while queue and len(results) < max_solutions:
    	  assign = queue.popleft()
    	  ...
    	  queue.append(assign + (col,))
  ```
- **DFS (Depth-First Search)**: dùng ngăn xếp hoặc đệ quy đặt xe trên từng hàng đến khi bế tắc thì lùi lại. Triển khai có thêm kiểm tra xung đột cột để cắt sớm, tuy nhiên dễ mắc vào nhánh không có lời giải và không đảm bảo tìm được lời giải tốt nhất.

  ```python
  def dfs_search(self, max_solutions=20):
      stack = [()]
      while stack and len(results) < max_solutions:
    	  assign = stack.pop()
    	  ...
    	  stack.append(assign + (col,))
  ```
- **UCS (Uniform Cost Search)**: mỗi bước đặt xe có chi phí `1 + |hàng - cột|`; tổng chi phí phản ánh độ "lệch" của toàn cấu hình. Hàng đợi ưu tiên giúp UCS luôn tìm lời giải chi phí thấp nhất theo hàm điểm đã định nghĩa, phù hợp khi muốn tối ưu bố cục hơn là chỉ cần đủ 8 xe.

  ```python
  def ucs_search(self, max_solutions=20):
      heap = [(0.0, ())]
      while heap and len(results) < max_solutions:
    	  cost, assign = heapq.heappop(heap)
    	  ...
    	  heapq.heappush(heap, (new_cost, new_assign))
  ```
- **IDS (Iterative Deepening Search)**: lặp lại DFS với giới hạn độ sâu tăng dần từ 0 đến 8. Mỗi lần lặp đảm bảo duyệt hết không gian độ sâu tương ứng như BFS nhưng chỉ giữ một nhánh trên ngăn xếp, do đó tiết kiệm bộ nhớ đáng kể.

  ```python
  def ids_search(self, max_depth=BOARD_SIZE, max_solutions=20):
      def dls(assign, depth_limit):
    	  if len(assign) >= depth_limit:
    		  return
    	  ...
    	  dls(assign + (col,), depth_limit)
  ```
- **Greedy Best-First Search**: mỗi nút được gán giá trị heuristic `h` bằng số hàng chưa đặt xe cộng chi phí cục bộ nhỏ. Hàng đợi ưu tiên chọn nút có `h` nhỏ nhất, cho phép lao nhanh đến lời giải nhưng có thể lạc hướng vì bỏ qua chi phí đã đi (`g`).

  ```python
  def greedy_search(self, max_solutions=20):
      heap = [(0, ())]
      while heap and len(results) < max_solutions:
    	  priority, assign = heapq.heappop(heap)
    	  ...
    	  heapq.heappush(heap, (-safe_next, new_assign))
  ```
- **A* Search**: kết hợp `g` (chi phí thực tế) và `h` (ước lượng hàng còn lại). Heuristic được thiết kế chấp nhận được (không vượt quá số hàng thật sự còn thiếu) nên A* luôn tìm được lời giải tối ưu theo hàm chi phí, đồng thời ít phải duyệt hơn UCS thuần.

  ```python
  def a_star_search(self, max_solutions=20):
      heap = [(0.0, 0.0, ())]
      while heap and len(results) < max_solutions:
    	  f_cost, g_cost, assign = heapq.heappop(heap)
    	  ...
    	  heapq.heappush(heap, (f_new, g_new, new_assign))
  ```
- **Hill-Climbing**: khởi tạo ngẫu nhiên 8 vị trí hợp lệ rồi thay đổi vị trí từng xe để giảm số xung đột. Khi không thể cải thiện thêm, thuật toán dừng ở cực trị địa phương; mã nguồn có cơ chế khởi động lại nhiều lần để tăng cơ hội tìm cấu hình tốt.

  ```python
  def hill_climbing(self, restarts=25):
      current = [random.randint(0, BOARD_SIZE - 1) for _ in range(BOARD_SIZE)]
      while current_conflicts > 0 and steps_without_improve < 100:
    	  ...
    	  current[row] = col
  ```
- **Simulated Annealing**: mở rộng từ hill-climbing bằng cách cho phép nhận nghiệm xấu hơn với xác suất giảm dần theo nhiệt độ. Lịch làm lạnh tuyến tính/exp được tinh chỉnh để cân bằng giữa khám phá (nhiệt độ cao) và khai thác (nhiệt độ thấp).

  ```python
  def simulated_annealing(self, attempts=10):
      while temperature > 0.001 and current_conflicts > 0:
    	  row = random.randint(0, BOARD_SIZE - 1)
    	  ...
    	  temperature *= 0.97
  ```
- **Beam Search**: duy trì `k` trạng thái tốt nhất ở mỗi bậc dựa trên heuristic (mặc định `k = 5`). Mỗi trạng thái sinh các con hợp lệ, rồi cắt tỉa về `k` nút tốt nhất. Beam search giảm mạnh độ phình nhánh nhưng có nguy cơ bỏ sót lời giải nếu `k` quá nhỏ.

  ```python
  def beam_search(self, beam_width=10, max_solutions=20):
      for row in range(BOARD_SIZE):
    	  for assign in beams:
    		  ...
    		  candidates.append((heur, new_assign))
  ```
- **Genetic Algorithm**: biểu diễn cá thể bằng hoán vị cột của 8 hàng (đảm bảo không trùng cột). Hàm fitness dựa trên số cặp xung đột; thuật toán dùng chọn lọc theo roulette, lai ghép theo hai điểm và đột biến tráo đổi vị trí để duy trì đa dạng.

  ```python
  def genetic_algorithm(self, population_size=60, generations=250, mutation_rate=0.05):
      population = [random_individual() for _ in range(population_size)]
      for _ in range(generations):
    	  ...
    	  child[row] = random.randint(0, BOARD_SIZE - 1)
  ```
- **Backtracking**: giải pháp cơ bản cho CSP; đặt xe theo hàng, kiểm tra xung đột cột trước khi đi sâu. Khi gặp bế tắc, lùi về hàng trước để thử cột khác. Đây là đường cơ sở để so sánh vì luôn tìm được lời giải với thời gian hợp lý.

  ```python
  def backtracking_search(self, max_solutions=20):
      def backtrack(assign):
    	  if len(assign) == BOARD_SIZE:
    		  results.append(assign)
    		  return
    	  ...
    	  backtrack(assign + (col,))
  ```
- **Forward Checking & AC-3**: kết hợp cùng backtracking. Forward checking giới hạn miền cột hợp lệ của các hàng chưa gán sau mỗi bước, còn AC-3 duy trì tính arc-consistent giữa các biến hàng/cột, giúp phát hiện bế tắc sớm và giảm số lần quay lui.

  ```python
  def forward_checking_search(self, max_solutions=20):
      domains = {row: set(range(BOARD_SIZE)) for row in range(BOARD_SIZE)}
      ...
      forward(assign + (col,), new_domains)

  def ac3_search(self):
      domains = {row: set(range(BOARD_SIZE)) for row in range(BOARD_SIZE)}
      queue = deque((i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if i != j)
      ...
      return self.csp_backtracking(domains)
  ```
- **Minimax / Alpha-Beta (phiên bản thử nghiệm)**: mô hình hóa việc đặt xe như trò chơi hai người xen kẽ chọn vị trí; minimax đánh giá điểm sau mỗi lượt, alpha-beta cắt tỉa các nhánh chắc chắn kém. Dù không phải thiết lập đối kháng thực sự, việc sử dụng cho thấy cách tái sử dụng bộ khung minimax trong bài toán ràng buộc.

  ```python
  def _minimax_recursive(self, assign, row, maximizing, alpha, beta):
      if row == BOARD_SIZE:
    	  return 0, tuple(assign)
      ...
      return best_score, best_solution
  ```

Thước đo đánh giá

- Số lời giải tìm được.
- Thời gian (thời gian thực nghiệm — có thể đo bằng time.time khi chạy các thuật toán).
- Chi phí/điểm lời giải: chương trình dùng hàm điểm: Σ (1 + |hàng - cột|) cho từng quân xe (giá trị nhỏ tốt hơn).

Hướng dẫn chạy nhanh

```powershell
python DinhTrongDucAnh_22110096_8_rock.py
```

Chọn thuật toán từ menu và nhấn "Tìm lời giải". Kết quả hiển thị số lời giải, điểm và có thể duyệt giữa các lời giải.

Video minh hoạ (YouTube)

```markdown
[![Xem demo]](https://www.youtube.com/watch?v=zT0aK9ONUds)
```

Ghi chú ngắn

- Đối với bài toán ràng buộc như này, các kỹ thuật CSP (Backtracking + Forward Checking + AC-3) thường rất hiệu quả. Thuật toán heuristic và metaheuristic (A*, GA, Hill-Climbing, Simulated Annealing) cho phép khám phá nhanh không gian tìm kiếm nhưng cần tinh chỉnh tham số.

---
