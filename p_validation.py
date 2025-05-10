import numpy as np
from scipy.stats import t

# scores_group1的具体分数
scores_group1 = np.array([75, 80, 85, 90, 95])

# scores_group2的统计数据
mean_group2 = 70  # scores_group2的平均值
std_dev_group2 = 10  # scores_group2的标准差
sample_size_group2 = 25  # scores_group2的样本大小

# 计算scores_group1的平均值
mean_group1 = scores_group1.mean()

# 计算scores_group1的标准差
std_dev_group1 = scores_group1.std()

# 由于只有一个样本的方差是已知的，我们使用以下公式来估计总体方差：
# pooled_std = sqrt(((n1 - 1) * sd1^2 + (n2 - 1) * sd2^2) / (n1 + n2 - 2))
pooled_std = ((std_dev_group1**2 * (len(scores_group1) - 1) + (std_dev_group2**2 * (sample_size_group2 - 1))) /
              (len(scores_group1) + sample_size_group2 - 2)) ** 0.5

# 计算t统计量和p值
# 注意：自由度需要根据pooled_std和样本大小进行调整
df = len(scores_group1) + sample_size_group2 - 2
t_stat = (mean_group1 - mean_group2) / pooled_std
p_value = t.sf(abs(t_stat), df) * 2  # 双边t检验

print(f"t统计量: {t_stat}")
print(f"p值: {p_value}")