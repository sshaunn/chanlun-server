"""
A股筛选系统 - API服务器

提供股票数据获取、分析和筛选功能的RESTful API。
适合使用Postman等工具进行测试。
"""

import os
from flask import Flask, jsonify
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置环境变量
os.environ['FLASK_ENV'] = os.environ.get('FLASK_ENV', 'local')
os.environ['LOG_LEVEL'] = os.environ.get('LOG_LEVEL', 'INFO')
os.environ['LOG_FILE'] = os.environ.get('LOG_FILE', 'logs/app.log')

# 导入配置模块
from config import get_config, reload_config
from utils.logger import setup_logger, get_logger
from api.routes import register_routes

logger = get_logger(__name__)


def create_app():
    """创建并配置Flask应用"""
    # 创建Flask应用
    app = Flask(__name__)

    # 从配置加载
    app.config['SECRET_KEY'] = get_config('server.secret_key', 'dev_secret_key')
    app.config['DEBUG'] = get_config('server.debug', True)

    # 设置日志
    setup_logger(app)

    # 注册API路由
    register_routes(app)

    # 健康检查路由
    @app.route('/')
    @app.route('/health')
    def health():
        return jsonify({
            'status': 'healthy',
            'environment': os.environ.get('FLASK_ENV', 'local'),
            'version': get_config('app.version', '1.0.0'),
            'api_endpoints': [
                {
                    'path': '/api/stock/list',
                    'method': 'GET',
                    'description': '获取所有A股列表'
                },
                {
                    'path': '/api/stock/data/<code>',
                    'method': 'GET',
                    'description': '获取单个股票数据',
                    'params': {
                        'start_date': '起始日期 (可选)',
                        'end_date': '结束日期 (可选)',
                        'adjust': '复权方式 (可选, 默认为qfq)'
                    }
                },
                {
                    'path': '/api/stock/screen',
                    'method': 'POST',
                    'description': '筛选股票',
                    'body': {
                        'strategy_type': '策略类型 (可选, 默认为rsi)',
                        'strategy_name': '策略名称 (可选, 默认为default)',
                        'max_stocks': '最大处理股票数量 (可选)',
                        'days_window': '查找信号的天数窗口 (可选, 默认为5)'
                    }
                },
                {
                    'path': '/api/stock/strategies',
                    'method': 'GET',
                    'description': '获取可用策略列表'
                },
                {
                    'path': '/api/stock/last_results',
                    'method': 'GET',
                    'description': '获取最近一次筛选结果'
                }
            ]
        })

    # 添加自定义错误处理
    @app.errorhandler(404)
    def page_not_found(e):
        return jsonify({
            'error': 'Not Found',
            'message': 'The requested URL was not found on the server.'
        }), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An internal server error occurred.'
        }), 500

    logger.info(f"应用已创建，环境: {os.environ.get('FLASK_ENV', 'local')}")
    return app


if __name__ == '__main__':
    # 创建应用
    app = create_app()

    # 获取服务器配置
    host = get_config('server.host', '0.0.0.0')
    port = int(get_config('server.port', 8080))
    debug = get_config('server.debug', True)

    # 运行应用
    logger.info(f"启动API服务器 - http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)