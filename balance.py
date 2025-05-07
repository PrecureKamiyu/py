# 工作负载平衡度 \(B\) 定义为所有服务器工作负载的标准差，计算公式如下：
#
#  \f[ B = \sqrt{\frac{1}{|S|} \sum_{j=1}^{|S|} (w_j - \bar{w})^2} \f]
#
# 其中：
# - \f$|S|\f$ 表示服务器的总数
# - \f$w_{j}\f$ 表示第 \f$j\f$ 个服务器的工作负载
# - \f$\bar {w}\f$ 表示所有服务器的平均工作负载

def calculate_workload_balance(server_workloads):
    if not server_workloads:
        raise ValueError("服务器工作负载列表不能为空")

    num_servers = len(server_workloads)
    mean_workload = sum(server_workloads) / num_servers
    variance_sum = sum((w - mean_workload) ** 2 for w in server_workloads)
    balance = (variance_sum / num_servers) ** 0.5
    return balance


if __name__ == "__main__":
    # 测试用例 1：简单场景
    simple_workloads = [10, 20, 30]
    try:
        balance = calculate_workload_balance(simple_workloads)
        print(f"测试用例 1 - 简单场景的工作负载平衡度: {balance:.2f}")
    except ValueError as e:
        print(f"错误: {e}")

    # 测试用例 2：更复杂的场景
    complex_workloads = [15.5, 22.0, 10.0, 35.7, 18.3]  # 5个服务器的负载
    try:
        balance = calculate_workload_balance(complex_workloads)
        print(f"测试用例 2 - 复杂场景的工作负载平衡度: {balance:.2f}")

        # 可选：打印详细信息
        mean = sum(complex_workloads) / len(complex_workloads)
        print(f"平均工作负载: {mean:.2f}")
        print("每个服务器的负载:", complex_workloads)
    except ValueError as e:
        print(f"错误: {e}")

    # 测试用例 3：完全平衡的情况
    balanced_workloads = [20, 20, 20, 20]
    try:
        balance = calculate_workload_balance(balanced_workloads)
        print(f"测试用例 3 - 完全平衡的工作负载平衡度: {balance:.2f}")
    except ValueError as e:
        print(f"错误: {e}")
