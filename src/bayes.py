from bayes_opt import BayesianOptimization

def make_optimizer(yhat_raw, y_val, label_index):

    def objective(threshold):

        y_true=y_val[:,label_index]
        # 根据阈值计算模型预测
        y_pred = (yhat_raw[:,label_index] > threshold).astype(int)
        inter = intersect_size(y_pred, y_true, 0)
        yhat_sum = y_pred.sum(axis=0)
        y_sum = y_true.sum(axis=0)
        f1 = pre_indivisual_macro_f1(yhat_sum, y_sum, inter)
        return -f1
    pbounds = {'threshold': (0, 1)}
    return BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)

def bayes_get_theta(yhat_raw, y):
    num_labels=y.shape[1]
    # 为每个标签创建优化器
    optimizers = [make_optimizer(yhat_raw, y, i) for i in range(num_labels)]

    best_thresholds = {}
    for i, optimizer in enumerate(optimizers):
        # 运行优化过程
        optimizer.maximize(init_points=5, n_iter=20)
        # 获取最优阈值
        best_thresholds[i] = optimizer.max['params']['threshold']
    return best_thresholds

# from sklearn.metrics import f1_score
#
# def objective_function(threshold, model, X_val, y_val):
#     # 应用阈值到所有标签
#     y_pred = (model.predict_proba(X_val) > threshold).astype(int)
#     # 计算所有标签的micro F1分数
#     f1_scores = f1_score(y_val, y_pred, average='micro', labels=range(num_labels))
#     # 计算平均micro F1分数作为目标函数值
#     return -np.mean(f1_scores)  # 返回负值，因为贝叶斯优化是最小化问题

