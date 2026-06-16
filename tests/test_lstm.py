from termcolor import cprint

# 定义不同级别的日志函数
def log_debug(msg):
    cprint(f"[DEBUG] {msg}", "cyan")  # 青色：用于调试信息

def log_info(msg):
    cprint(f"[INFO] {msg}", "green")  # 绿色：用于常规运行信息
    cprint(f"[DEFAULT] {msg}", None)

def log_warning(msg):
    cprint(f"[WARNING] {msg}", "yellow")  # 黄色：用于警告信息

def log_error(msg):
    cprint("\n" + "=" * 60, "red")
    cprint(f"[ERROR] {msg}", "red", attrs=["bold"])  # 红色加粗：用于错误信息
    cprint("\n" + "=" * 60, "red")


# --- 测试一下效果 ---
log_debug("正在连接数据库...")
log_info("用户登录成功！")
log_warning("内存使用率已达 85%，请注意。")
log_error("无法读取配置文件，程序即将退出！")