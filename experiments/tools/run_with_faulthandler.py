"""以 unbuffered 方式运行 train.py, 并注册 SIGUSR1 → 打印所有线程的 Python 栈 (不需要 ptrace 权限)。
用法: python -u tools/run_with_faulthandler.py --config configs/xxx.yaml   (参数原样传给 train.py)
诊断: kill -USR1 <pid> 后看日志里的 "Thread ... (most recent call first)"。2026-10-09 为排查 exp309 卡死加。"""
import faulthandler, signal, sys, runpy, os
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
sys.argv = ["train.py"] + sys.argv[1:]
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
runpy.run_path("train.py", run_name="__main__")
