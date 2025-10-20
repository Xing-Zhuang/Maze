import socket

def get_available_ports(n=2):
    """
    获取当前机器上 n 个可用的 TCP 端口。
    
    参数:
        n (int): 需要获取的端口数量，默认为2。
    
    返回:
        List[int]: 包含 n 个可用端口号的列表。
    """
    ports = []
    sockets = []

    try:
        for _ in range(n):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('', 0))  # 绑定到任意地址，端口由系统自动分配
            port = sock.getsockname()[1]
            ports.append(port)
            sockets.append(sock)  # 保持 socket 打开，防止端口被立即复用
        return ports
    finally:
        # 关闭所有打开的 sockets
        for sock in sockets:
            sock.close()
