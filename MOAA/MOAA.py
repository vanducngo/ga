from MOAA.operators import *
from MOAA.Solutions import *
import numpy as np
import time

def p_selection(it, p_init, n_queries):
    it = int(it / n_queries * 10000)
    if 0 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 5
    elif 500 < it <= 1000:
        p = p_init / 6
    elif 1000 < it <= 2000:
        p = p_init / 8
    elif 2000 < it <= 4000:
        p = p_init / 10
    elif 4000 < it <= 6000:
        p = p_init / 12
    elif 6000 < it <= 8000:
        p = p_init / 15
    elif 8000 < it:
        p = p_init / 20
    else:
        p = p_init

    return p


class Population:
    def __init__(self, solutions: list, loss_function, include_dist):
        # Là danh sách các giải pháp (solutions) ban đầu, được truyền vào khi khởi tạo quần thể.
        self.population = solutions
        # 
        self.fronts = None
        # Hàm mất mát được sử dụng để đánh giá chất lượng của từng giải pháp.
        self.loss_function = loss_function
        # Biến boolean xác định xem khoảng cách (distance) giữa giải pháp và dữ liệu gốc 
        # có được tính là một mục tiêu trong quá trình tối ưu hóa hay không.
        self.include_dist = include_dist

    def evaluate(self):
        for pi in self.population:
            # Gọi phương thức evaluate trên từng giải pháp trong quần thể
            pi.evaluate(self.loss_function, self.include_dist)

    # Hàm này tìm kiếm và trả về các giải pháp đối kháng (adversarial solutions) từ quần thể hiện tại.
    def find_adv_solns(self, max_dist):
        adv_solns = []
        for pi in self.population:
            # Giải pháp phải được đánh dấu là đối kháng (thường được đánh giá trong evaluate).
            # pi.fitnesses[1] <= max_dist: Khoảng cách L2 giữa giải pháp và dữ liệu gốc phải nhỏ 
            # hơn hoặc bằng giá trị max_dist
            if pi.is_adversarial and pi.fitnesses[1] <= max_dist:
                adv_solns.append(pi)
    
        return adv_solns


class Attack:
    def __init__(self, params):
        self.params = params
        self.fitness = []

        self.data = []

    # def update_data(self, front):

    def completion_procedure(self, population: Population, loss_function, fe, success):

        #print(success, fe)
        #print(1/0)

        adversarial_labels = []
        for soln in population.fronts[0]:
            adversarial_labels.append(loss_function.get_label(soln.generate_image()))

        d = {"front0_imgs": [soln.generate_image() for soln in population.fronts[0]],
             "queries": fe,
             "true_label": loss_function.true,
             "adversarial_labels": adversarial_labels,
             "front0_fitness": [soln.fitnesses for soln in population.fronts[0]],
             "fitness_process": self.fitness,
             "success": success
             }

        # print(d["true_label"], d["adversarial_labels"])
        np.save(self.params["save_directory"], d, allow_pickle=True)

    def attack(self, loss_function):
        start = time.time()
        # print(loss_function(self.params["x"]))
        # print(self.params["n_pixels"])
        # Minimizes
        h, w, c = self.params["x"].shape[0:] # Trích xuất chiều cao và chiều rộng của ảnh đầu vào.
        pm = self.params["pm"] # mutation parameter
        n_pixels = h * w # Tổng số pixels
        all_pixels = np.arange(n_pixels) # Danh sách all_pixels chứa các chỉ số của tất cả các pixel.
        ones_prob = (1 - self.params["zero_probability"]) / 2   # Vì sao chia 2 ở đây? [-1, 1]

        '''
        Tạo một danh sách các giải pháp ban đầu (init_solutions).

        np.random.choice(all_pixels, size=(self.params["eps"]), replace=False): Chọn ngẫu nhiên một số lượng pixel từ tập hợp all_pixels
        np.random.choice([-1, 1, 0], size=(self.params["eps"], 3)
        '''
        init_solutions = [Solution(np.random.choice(all_pixels,
                                                    size=(self.params["eps"]), replace=False),
                                   np.random.choice([-1, 1, 0], size=(self.params["eps"], 3),
                                                    p=(ones_prob, ones_prob, self.params["zero_probability"])),
                                   self.params["x"].copy(), self.params["p_size"]) for _ in
                          range(self.params["pop_size"])]

        # Khởi tạo quần thể
        population = Population(init_solutions, loss_function, self.params["include_dist"])
        # Đánh giá quần thể
        population.evaluate()
        # Số lượng lần hàm mất mát được tính toán.
        fe = len(population.population)

        # Vòng lặp tiến hóa
        for it in range(1, self.params["iterations"]):
            #pm = p_selection(it, self.params["pm"], self.params["iterations"])
            pm = self.params["pm"] # mutation parameter
            
            # Phân loại không trội
            population.fronts = fast_nondominated_sort(population.population)
            

            # Kiểm tra giải pháp đối kháng
            # self.params["max_dist"]: Khoảng cách L2 tối đa được chấp nhận giữa hình ảnh gốc và hình ảnh bị thay đổi.
            # 1e-5: Là giá trị rất nhỏ (0.00001), biểu thị rằng mẫu đối kháng phải duy trì sự giống nhau cao so với hình ảnh gốc.

            # Kiểm tra nếu có giải pháp nào là mẫu đối kháng (adversarial solution) dựa trên khoảng cách tối đa (max_dist)
            adv_solns = population.find_adv_solns(self.params["max_dist"])
            if len(adv_solns) > 0:
                # Tìm thấy => 
                #  + Cập nhật độ phù hợp (fitness) của quần thể.
                #  + Gọi hàm completion_procedure để xử lý kết quả.
                # Kết thúc vòng lặp tiến hóa.
                self.fitness.append(min(population.population, key=attrgetter('loss')).fitnesses)
                self.completion_procedure(population, loss_function, fe, True)
                return


            self.fitness.append(min(population.population, key=attrgetter('loss')).fitnesses)

            #print(fe, self.fitness[-1])
            
            
            for front in population.fronts:
                # Tính toán khoảng cách đông đúc (crowding distance)
                # Dùng để đánh giá mức độ phân tán của các giải pháp trong cùng một mặt trận.
                calculate_crowding_distance(front)
            
            # Chọn cha mẹ từ quần thể hiện tại bằng phương pháp chọn lọc giải đấu (tournament selection)
            parents = tournament_selection(population.population, self.params["tournament_size"])
            # Tạo ra con cái thông qua lai ghép và đột biến.
            # Lai ghép: pc => crossover probability
            # Đột biến: pm => mutation probability
            children = generate_offspring(parents,
                                          self.params["pc"],
                                          pm,
                                          all_pixels,
                                          self.params["zero_probability"])

            # Tạo một quần thể mới chứa các con cái.
            offsprings = Population(children, loss_function, self.params["include_dist"])
            # 
            fe += len(offsprings.population)
            # Đánh giá các con cái bằng cách tính hàm mất mát.
            offsprings.evaluate()

            # Kết hợp quần thể con cái và cha mẹ để tạo thành quần thể mới.
            population.population.extend(offsprings.population)

            # Phân loại quần thể kết hợp
            population.fronts = fast_nondominated_sort(population.population)
            front_num = 0
            
            # Giữ lại những giải pháp tốt nhất để đảm bảo kích thước quần thể không vượt quá giới hạn (pop_size)
            new_solutions = []
            while len(new_solutions) + len(population.fronts[front_num]) <= self.params["pop_size"]:
                calculate_crowding_distance(population.fronts[front_num])
                new_solutions.extend(population.fronts[front_num])
                front_num += 1


            calculate_crowding_distance(population.fronts[front_num])
            population.fronts[front_num].sort(key=attrgetter("crowding_distance"), reverse=True)
            new_solutions.extend(population.fronts[front_num][0:self.params["pop_size"] - len(new_solutions)])

            population = Population(new_solutions, loss_function, self.params["include_dist"])

        population.fronts = fast_nondominated_sort(population.population)
        # Ghi lại độ phù hợp cuối cùng của quần thể.
        self.fitness.append(min(population.population, key=attrgetter('loss')).fitnesses)
        # Gọi completion_procedure để xử lý kết quả.
        self.completion_procedure(population, loss_function, fe, False)
        #print(time.time() - start)ff
        return
