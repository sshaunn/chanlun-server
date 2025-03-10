import os
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta


class ConfigLoader:
    """配置加载器 - 支持多环境配置"""

    _instance = None

    def __new__(cls, env=None):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, env=None):
        """
        初始化配置加载器

        参数:
        env (str): 环境名称，如果为None则从环境变量获取
        """
        # 避免重复初始化
        if self._initialized:
            return

        # 确定当前环境
        self.env = env or os.environ.get('FLASK_ENV', 'local')
        print(f"当前环境: {self.env}")

        # 项目根目录
        self.root_dir = Path(__file__).parent.parent
        self.resources_dir = self.root_dir / "resources"

        # 加载配置
        self.config = self._load_config()

        # 处理环境变量替换
        self._process_env_variables(self.config)

        # 派生配置（基于已有配置计算的值）
        self._derive_config_values()

        self._initialized = True

    def _load_config(self):
        """加载配置文件"""
        config = {}

        # 加载基础配置
        base_config_path = self.resources_dir / "application.yml"
        if base_config_path.exists():
            with open(base_config_path, 'r', encoding='utf-8') as f:
                config.update(yaml.safe_load(f) or {})
        else:
            print(f"警告: 基础配置文件不存在 - {base_config_path}")

        # 加载环境特定配置
        env_config_path = self.resources_dir / f"application-{self.env}.yml"
        if env_config_path.exists():
            with open(env_config_path, 'r', encoding='utf-8') as f:
                env_config = yaml.safe_load(f) or {}
                # 递归合并配置
                self._merge_configs(config, env_config)
        else:
            print(f"警告: 环境配置文件不存在 - {env_config_path}")

        return config

    def _merge_configs(self, base, override):
        """递归合并配置字典"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def _process_env_variables(self, config_dict):
        """处理配置中的环境变量引用，格式为${ENV_VAR}"""
        if isinstance(config_dict, dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    self._process_env_variables(value)
                elif isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    env_value = os.environ.get(env_var)
                    if env_value is not None:
                        config_dict[key] = env_value
                    else:
                        print(f"警告: 环境变量 {env_var} 未定义，使用原始值")

    def _derive_config_values(self):
        """根据已有配置计算派生值"""
        # 计算默认起始日期
        days = self.get('stock.default_start_date_days', 365)
        default_start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        default_end_date = datetime.now().strftime('%Y%m%d')

        # 添加到配置
        if 'stock' not in self.config:
            self.config['stock'] = {}
        self.config['stock']['default_start_date'] = default_start_date
        self.config['stock']['default_end_date'] = default_end_date

        # 确保缓存目录存在
        cache_dir = self.get('cache.dir')
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # 确保日志目录存在
        log_file = self.get('logging.file')
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

    def get(self, key, default=None):
        """获取配置值，支持点分隔的路径"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_all(self):
        """获取所有配置"""
        return self.config

    def get_db_config(self):
        """获取数据库配置"""
        db_enabled = self.get('database.enabled', False)
        if not db_enabled:
            return None

        db_type = self.get('database.type', 'sqlite')
        db_config = self.get(f'database.{db_type}', {})

        # 添加数据库类型
        db_config['type'] = db_type
        return db_config

    def load_strategy_config(self, strategy_type='rsi'):
        """加载特定类型的策略配置"""
        strategy_file = self.resources_dir / self.get('strategy.config_dir',
                                                      'strategies') / f"{strategy_type}_strategy.json"

        if not strategy_file.exists():
            print(f"警告: 策略配置文件不存在 - {strategy_file}")
            return {}

        try:
            with open(strategy_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载策略配置失败: {e}")
            return {}

    def get_strategy_params(self, strategy_type=None, strategy_name='default'):
        """获取特定策略的参数"""
        if strategy_type is None:
            strategy_type = self.get('strategy.default_strategy', 'rsi')

        strategy_config = self.load_strategy_config(strategy_type)
        return strategy_config.get(strategy_name, strategy_config.get('default', {}))

    def setup_logging(self):
        """配置日志系统"""
        log_level_name = self.get('logging.level', 'INFO')
        log_level = getattr(logging, log_level_name.upper(), logging.INFO)

        log_format = self.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = self.get('logging.file')

        # 配置根日志记录器
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(),  # 控制台输出
                logging.FileHandler(log_file) if log_file else logging.NullHandler()  # 文件输出
            ]
        )

        # 降低一些库的日志级别
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)