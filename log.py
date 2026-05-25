# tools/log.py
import sys
import os
from datetime import datetime

class SimpleLogger:
    def __init__(self, log_file=None, level="INFO", use_color=True):
        """
        :param log_file: 日志文件路径，若指定则同时写入文件
        :param level: 日志级别 DEBUG, INFO, WARNING, ERROR
        :param use_color: 是否在控制台使用颜色
        """
        self.log_file = log_file
        self.level = level.upper()
        self.use_color = use_color and sys.stdout.isatty()
        
        if self.log_file:
            log_dir = os.path.dirname(self.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
    
    def _log(self, level, msg):
        if self._should_log(level):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            text = f"[{timestamp}] [{level}] {msg}"
            # 控制台输出（带颜色）
            if self.use_color:
                color = self._get_color(level)
                print(f"{color}{text}\033[0m")
            else:
                print(text)
            # 写入文件
            if self.log_file:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(text + '\n')
    
    def _should_log(self, level):
        levels = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3}
        return levels.get(level, 1) >= levels.get(self.level, 1)
    
    def _get_color(self, level):
        colors = {'DEBUG': '\033[36m',   # 青色
                  'INFO': '\033[32m',    # 绿色
                  'WARNING': '\033[33m', # 黄色
                  'ERROR': '\033[31m'}   # 红色
        return colors.get(level, '\033[0m')
    
    def debug(self, msg):
        self._log('DEBUG', msg)
    
    def info(self, msg):
        self._log('INFO', msg)
    
    def warning(self, msg):
        self._log('WARNING', msg)
    
    def error(self, msg):
        self._log('ERROR', msg)

# 创建全局 logger 实例，可根据需要修改日志文件路径
logger = SimpleLogger(log_file="training.log", level="INFO", use_color=True)
