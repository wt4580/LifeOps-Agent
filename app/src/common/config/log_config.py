import logging
# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 抑制 APScheduler 的周期任务日志：每分钟一次的 _push_due_todos_job 不再刷屏，只在出错时输出。
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logger = logging.getLogger("LifeOps-Agent")