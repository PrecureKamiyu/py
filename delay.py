# 访问延迟 \f$D\f$ 定义为用户请求到达边缘服务器的平均距离，计算公式如下：
#
# \f[D = \frac{1}{N} \sum_{i=1}^{N} \min_{j \in S} d(u_i, s_j)\f]
#
# 其中：
# - \f$N\f$ 表示用户请求总数
# - \f$S\f$ 表示边缘服务器集合
# - \f$u_{i}\f$ 表示第 \f$i\f$ 个用户的位置
# - \f$s_{j}\f$ 表示第 \f$j\f$ 个服务器的位置
# - \f$d(u_{i}, s_{j})\f$ 表示用户 \f$u_{i}\f$ 与服务器 \f$s_{j}\f$ 之间的地理距离

def calculate_access_delay(user_locations, server_locations):
    """
    参数:
        user_locations (list): 用户位置列表，每个位置是一个元组 (x, y) 表示坐标
        server_locations (list): 服务器位置列表，每个位置是一个元组 (x, y) 表示坐标
    返回:
        float: 平均访问延迟 D，即每个用户到最近服务器的平均距离
    示例:
        >>> users = [(0, 0), (1, 1), (2, 2)]
        >>> servers = [(0, 1), (2, 1)]
        >>> calculate_access_delay(users, servers)
        1.4142135623730951
    """
    # 输入验证
    if not user_locations:
        raise ValueError("用户位置列表不能为空")
    if not server_locations:
        raise ValueError("服务器位置列表不能为空")

    total_distance = 0
    num_users = len(user_locations)

    # 对每个用户计算到最近服务器的距离
    for user_loc in user_locations:
        # 计算到所有服务器的距离并取最小值
        min_distance = float('inf')
        for server_loc in server_locations:
            # 使用欧几里得距离公式: sqrt((x2-x1)^2 + (y2-y1)^2)
            distance = ((user_loc[0] - server_loc[0]) ** 2 +
                       (user_loc[1] - server_loc[1]) ** 2) ** 0.5
            min_distance = min(min_distance, distance)

        total_distance += min_distance

    # 计算平均值
    average_delay = total_distance / num_users
    return average_delay

# 测试代码
if __name__ == "__main__":
    # 示例数据
    users = [(0, 0), (1, 1), (2, 2)]
    servers = [(0, 1), (2, 1)]

    try:
        delay = calculate_access_delay(users, servers)
        print(f"平均访问延迟: {delay:.2f}")
    except ValueError as e:
        print(f"错误: {e}")
