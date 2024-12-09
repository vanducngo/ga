import numpy as np
from copy import deepcopy
from operator import attrgetter


'''
Solution đại diện cho một giải pháp trong thuật toán tiến hóa
    + Quản lý trạng thái của một giải pháp, bao gồm các pixel được thay đổi, giá trị của chúng, và hình ảnh bị tấn công.
    + Đánh giá chất lượng của giải pháp thông qua hàm mất mát (loss_function) và các chỉ số khác.
    + Cung cấp các công cụ để kiểm tra sự ưu việt (dominance) so với các giải pháp khác
'''
class Solution:
    def __init__(self, pixels, values, x, p_size):
        # pixels: Danh sách các pixel (vị trí) được thay đổi trong hình ảnh.
        self.pixels = pixels  # list of Integers
        # values: Danh sách giá trị được áp dụng cho các pixel được chọn (có thể là -1, 1, hoặc 0).
        self.values = values  # list of Binary tuples, i.e. [0, 1, 1]
        # Hình ảnh gốc (dưới dạng ma trận  w×w×3, tương ứng với chiều cao, chiều rộng, và kênh màu).
        self.x = x  # (w x w x 3)
        # fitnesses: Mảng lưu trữ các giá trị hàm mục tiêu của giải pháp.
        self.fitnesses = []
        # is_adversarial: Biến boolean, xác định liệu giải pháp có phải là đối kháng hay không.
        self.is_adversarial = None
        # Kích thước chiều rộng/chiều cao của hình ảnh (giả định là hình vuông)
        self.w = x.shape[0]
        # Số lượng pixel được thay đổi trong giải pháp
        self.delta = len(self.pixels)
        
        # Các thuộc tính liên quan đến thuật toán tiến hóa đa mục tiêu

        # Số lần phương án này bị thống trị (dominated) bởi phương án khác
        self.domination_count = None
        # Danh sách các giải pháp mà giải pháp này chi phố
        self.dominated_solutions = None
        # rank: Cấp bậc của giải pháp trong phân loại không trội (Pareto front).
        self.rank = None
        # Khoảng cách đông đúc, đo lường mức độ phân tán của giải pháp trong mặt trận Pareto.
        self.crowding_distance = None

        self.loss = None
        self.p_size = p_size

    def copy(self):
        a = deepcopy(self)
        return deepcopy(self)

    def euc_distance(self, img):
        '''
         + Tính khoảng cách L2 (bình phương khoảng cách Euclidean) giữa hình ảnh bị tấn công (img) và hình ảnh gốc (self.x).
         + Được sử dụng để đánh giá mức độ thay đổi của giải pháp.
        '''
        return np.sum((img - self.x.copy()) ** 2)

    def generate_image(self):
        '''
        Tạo hình ảnh đối kháng (x_adv) bằng cách áp dụng nhiễu lên hình ảnh gốc.
        '''
        x_adv = self.x.copy()
        
        for i in range(self.delta):
            row = self.pixels[i] // self.w
            col = self.pixels[i] % self.w
            # print(f"Before Update: {x_adv[row, col]}")
            x_adv[row, col] += np.uint8(self.values[i] * self.p_size)
            # print(f"After Update: {x_adv[row, col]}")
            # x_adv[self.pixels[i] // self.w, self.pixels[i] % self.w] += np.uint8(self.values[i] * self.p_size)

        # x_adv = x_adv / 255
        return np.clip(x_adv, 0, 255)
        # return x_adv

    def evaluate(self, loss_function, include_dist):
        # Tạo hình ảnh đối kháng (img_adv)
        img_adv = self.generate_image()
        # Tính giá trị hàm mất mát thông qua loss_function
        # Hàm loss trả về một danh sách, với 
        #    + phần tử đầu tiên là boolean xác định tính đối kháng 
        #    + các phần tử tiếp theo là giá trị hàm mục tiêu
        fs = loss_function(img_adv)
        # Xác định xem giải pháp có phải là đối kháng (is_adversarial) hay không (giá trị boolean từ fs[0]).
        self.is_adversarial = fs[0]  # Assume first element is boolean always
        # Lưu các giá trị hàm mục tiêu vào fitnesses.
        self.fitnesses = fs[1:]
        
        # Nếu include_dist là True, thêm khoảng cách 𝐿2 vào fitnesses.
        if include_dist:
            dist = self.euc_distance(img_adv)
            self.fitnesses.append(dist)
        else:
            self.fitnesses.append(0)

        self.fitnesses = np.array(self.fitnesses)
        self.loss = fs[1]

    # Xác định xem giải pháp hiện tại có chi phối (dominate) một giải pháp khác (soln) hay không.
    def dominates(self, soln):
        if self.is_adversarial is True and soln.is_adversarial is False:
            # Nếu giải pháp hiện tại là đối kháng và giải pháp kia không phải, thì giải pháp hiện tại chi phối.
            return True

        if self.is_adversarial is False and soln.is_adversarial is True:
            # Nếu giải pháp kia là đối kháng và giải pháp hiện tại không phải, thì giải pháp hiện tại không chi phối.
            return False

        if self.is_adversarial is True and soln.is_adversarial is True:
            # Nếu cả hai đều là đối kháng:
            # So sánh giá trị mục tiêu thứ hai (fitnesses[1], ví dụ: khoảng cách 𝐿2).
            return True if self.fitnesses[1] < soln.fitnesses[1] else False

        if self.is_adversarial is False and soln.is_adversarial is False:
            # Nếu cả hai không phải là đối kháng
            # So sánh giá trị mục tiêu thứ nhất (fitnesses[0], ví dụ: giá trị hàm mất mát)
            return True if self.fitnesses[0] < soln.fitnesses[0] else False

'''
Phân loại quần thể các giải pháp trong thuật toán tiến hóa đa mục tiêu thành các mặt trận Pareto (Pareto fronts). 
Đây là một bước quan trọng trong việc chọn lọc các giải pháp dựa trên độ ưu việt (dominance)

- Front 0 (Pareto-optimal front): Tập hợp các giải pháp không bị chi phối bởi bất kỳ giải pháp nào khác.
- Front 1: Các giải pháp bị chi phối trực tiếp bởi Front 0, nhưng không bị chi phối bởi giải pháp nào khác trong Front 1.
- Front 2, 3,...: Tương tự.
'''
def fast_nondominated_sort(population):
    fronts = [[]]
    # Với mỗi individual trong quần thể
    for individual in population:
        # số lượng giải pháp chi phối individual
        individual.domination_count = 0
        # Danh sách các giải pháp bị individual chi phối
        individual.dominated_solutions = []
        
        for other_individual in population:
            if individual.dominates(other_individual):
                # individual chi phối other_individual
                individual.dominated_solutions.append(other_individual)
            elif other_individual.dominates(individual):
                # other_individual chi phối individual
                individual.domination_count += 1
        
        if individual.domination_count == 0:
            # Front 0 (Pareto-optimal front)
            individual.rank = 0
            fronts[0].append(individual)
    
    # Phân loại các mặt trận Pareto tiếp theo
    i = 0
    while len(fronts[i]) > 0:
        temp = []
        for individual in fronts[i]:
            for other_individual in individual.dominated_solutions:
                other_individual.domination_count -= 1
                if other_individual.domination_count == 0:
                    other_individual.rank = i + 1
                    temp.append(other_individual)
        i = i + 1
        fronts.append(temp)

    return fronts

'''
Tính khoảng cách đông đúc (crowding distance) cho các giải pháp trong một mặt trận Pareto (front)
=> được sử dụng để đánh giá mức độ đa dạng của các giải pháp trong một mặt trận.

Khoảng cách đông đúc giúp xác định:
  + Những giải pháp ở gần ranh giới (boundary) của mặt trận được ưu tiên hơn, vì chúng có khoảng cách đông đúc cao.
  + Giải pháp có khoảng cách đông đúc cao hơn được giữ lại trong quá trình chọn lọc để đảm bảo sự đa dạng của quần thể.

Ý nghĩa:
  + Các giải pháp nằm gần nhau hơn sẽ có khoảng cách đông đúc nhỏ hơn.
  + Khoảng cách đông đúc cao khuyến khích sự đa dạng trong quá trình chọn lọc.

'''
def calculate_crowding_distance(front):
    if len(front) > 0:
        # Số lượng giải pháp trong mặt trận front
        solutions_num = len(front)
        
        # Khởi tạo crowding_distance cho từng giải pháp
        for individual in front:
            individual.crowding_distance = 0
        # len(front[0].fitnesses) là số lượng hàm mục tiêu trong bài toán (thường là đa mục tiêu).
        for m in range(len(front[0].fitnesses)):
            # Sắp xếp các giải pháp trong mặt trận theo giá trị của hàm mục tiêu thứ m
            front.sort(key=lambda individual: individual.fitnesses[m])
            # Gán giá trị lớn cho các giải pháp biên
            # Giải pháp có giá trị nhỏ nhất và lớn nhất trên hàm mục tiêu thứ m được ưu tiên hơn vì chúng nằm ở biên của mặt trận.
            front[0].crowding_distance = 10 ** 9
            front[solutions_num - 1].crowding_distance = 10 ** 9
            
            # Lấy giá trị của hàm mục tiêu thứ m cho tất cả các giải pháp trong mặt trận.
            m_values = [individual.fitnesses[m] for individual in front]
            # Tính thang đo (scale)
            scale = max(m_values) - min(m_values)
            
            # Nếu scale = 0 (tất cả các giá trị của hàm mục tiêu là giống nhau), gán scale = 1 để tránh lỗi chia cho 0.
            if scale == 0: scale = 1

            # Với mỗi giải pháp (trừ giải pháp đầu tiên và cuối cùng), tính khoảng cách đông đúc dựa trên chênh 
            # lệch giá trị của các giải pháp lân cận trên hàm mục tiêu thứ m
            for i in range(1, solutions_num - 1):
                front[i].crowding_distance += (front[i + 1].fitnesses[m] - front[i - 1].fitnesses[m]) / scale

'''
So sánh hai giải pháp (individual và other_individual) trong một thuật toán tiến hóa đa mục tiêu. So sánh này dựa trên hai yếu tố:
    + Rank (cấp bậc): Mức độ chi phối của giải pháp, hay thứ tự mặt trận Pareto mà giải pháp thuộc về.
    + Crowding Distance (khoảng cách đông đúc): Mức độ phân tán của giải pháp trong mặt trận Pareto mà nó thuộc về.
=> Quyết định giải pháp nào được ưu tiên hơn.
'''
def crowding_operator(individual, other_individual):
    if (individual.rank < other_individual.rank) or ((individual.rank == other_individual.rank) and (
            individual.crowding_distance > other_individual.crowding_distance)):
        return 1
    else:
        return -1

'''
Thực hiện một vòng chọn lọc giải đấu (tournament selection) từ quần thể (population)
Từ một số lượng giới hạn các giải pháp tham gia (tournament_size), chọn ra giải pháp tốt nhất dựa trên hàm crowding_operator.
'''
def __tournament(population, tournament_size):
    # Chọn ngẫu nhiên tournament_size giải pháp từ quần thể (population) => Khoong trùng lặp (replace=False)
    participants = np.random.choice(population, size=(tournament_size,), replace=False)
    best = None
    for participant in participants:
        # Nếu best chưa được khởi tạo hoặc participant tốt hơn best (dựa trên hàm crowding_operator), cập nhật best.
        if best is None or (
                crowding_operator(participant, best) == 1):  # and self.__choose_with_prob(self.tournament_prob)):
            best = participant

    return best

'''
Sử dụng hàm __tournament để chọn cặp cha mẹ từ quần thể cho quá trình lai ghép và đột biến.
'''
def tournament_selection(population, tournament_size):
    parents = []
    # len(population) // 2: Số cặp cha mẹ bằng một nửa kích thước quần thể.
    while len(parents) < len(population) // 2:
        parent1 = __tournament(population, tournament_size)
        parent2 = __tournament(population, tournament_size)

        parents.append([parent1, parent2])
    return parents
