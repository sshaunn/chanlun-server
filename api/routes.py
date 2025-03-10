"""
API路由注册模块

集中注册所有API蓝图。
"""

from api.stock_routes import stock_bp

def register_routes(app):
    """注册所有API路由"""
    # 注册股票相关API
    app.register_blueprint(stock_bp)

    # 返回应用实例，方便链式调用
    return app