"""
股票数据API路由模块

提供股票数据和筛选相关的REST API。
"""
import pandas as pd
from flask import Blueprint, jsonify, request, current_app

from services.screen_service import ScreeningService

# 创建蓝图
stock_bp = Blueprint('stock', __name__, url_prefix='/api/stock')

# 创建筛选服务实例
screening_service = ScreeningService()


@stock_bp.route('/list', methods=['GET'])
def get_stock_list():
    """获取所有A股列表"""
    try:
        stocks = screening_service.stock_fetcher.get_all_stocks()
        return jsonify({
            'success': True,
            'data': stocks.to_dict('records') if not stocks.empty else [],
            'count': len(stocks)
        })
    except Exception as e:
        current_app.logger.error(f"获取股票列表失败: {e}")
        return jsonify({
            'success': False,
            'message': f"获取股票列表失败: {str(e)}"
        }), 500


@stock_bp.route('/data/<code>', methods=['GET'])
def get_stock_data(code):
    """获取单个股票数据"""
    try:
        # 获取查询参数
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        adjust = request.args.get('adjust', 'qfq')

        # 获取股票数据
        data = screening_service.stock_fetcher.get_stock_data(
            code, start_date=start_date, end_date=end_date, adjust=adjust
        )

        if data is None:
            return jsonify({
                'success': False,
                'message': f"无法获取股票 {code} 的数据"
            }), 404

        return jsonify({
            'success': True,
            'data': data.reset_index().to_dict('records'),
            'count': len(data)
        })
    except Exception as e:
        current_app.logger.error(f"获取股票数据失败: {e}")
        return jsonify({
            'success': False,
            'message': f"获取股票数据失败: {str(e)}"
        }), 500


@stock_bp.route('/screen', methods=['POST'])
def screen_stocks():
    """筛选股票"""
    try:
        # 获取请求参数
        data = request.json or {}
        strategy_type = data.get('strategy_type', 'rsi')
        strategy_name = data.get('strategy_name', 'default')
        max_stocks = data.get('max_stocks')
        days_window = data.get('days_window', 5)

        # 运行筛选
        results = screening_service.run_screening(
            strategy_type=strategy_type,
            strategy_name=strategy_name,
            max_stocks=max_stocks,
            days_window=days_window
        )

        return jsonify({
            'success': True,
            'data': results,
            'count': len(results),
            'strategy': {
                'type': strategy_type,
                'name': strategy_name
            }
        })
    except Exception as e:
        current_app.logger.error(f"股票筛选失败: {e}")
        return jsonify({
            'success': False,
            'message': f"股票筛选失败: {str(e)}"
        }), 500


@stock_bp.route('/strategies', methods=['GET'])
def get_strategies():
    """获取可用策略列表"""
    try:
        strategies = screening_service.get_available_strategies()
        return jsonify({
            'success': True,
            'data': strategies
        })
    except Exception as e:
        current_app.logger.error(f"获取策略列表失败: {e}")
        return jsonify({
            'success': False,
            'message': f"获取策略列表失败: {str(e)}"
        }), 500


@stock_bp.route('/last_results', methods=['GET'])
def get_last_results():
    """获取最近一次筛选结果"""
    try:
        results = screening_service.get_last_results()
        return jsonify({
            'success': True,
            'data': results
        })
    except Exception as e:
        current_app.logger.error(f"获取最近筛选结果失败: {e}")
        return jsonify({
            'success': False,
            'message': f"获取最近筛选结果失败: {str(e)}"
        }), 500


@stock_bp.route('/test_rsi/<code>', methods=['GET'])
def test_rsi(code):
    """测试单个股票的RSI指标计算"""
    try:
        # 导入必要的库
        import pandas as pd
        import numpy as np
        import json

        # 获取查询参数
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        period = int(request.args.get('period', 14))

        # 获取股票数据
        stock_data = screening_service.stock_fetcher.get_stock_data(
            code, start_date=start_date, end_date=end_date
        )

        if stock_data is None or stock_data.empty:
            return jsonify({
                'success': False,
                'message': f"无法获取股票 {code} 的数据"
            }), 404

        # 创建RSI指标计算器
        from indicators.rsi import RSIIndicator
        rsi_indicator = RSIIndicator(periods=[period], include_stochastic=True)

        # 计算RSI指标
        data_with_rsi = rsi_indicator.calculate(stock_data)

        # 获取股票名称
        stock_name = None
        if '股票名称' in data_with_rsi.columns:
            stock_name = data_with_rsi['股票名称'].iloc[0]
            if isinstance(stock_name, (np.ndarray, list)):
                stock_name = stock_name[0] if len(stock_name) > 0 else None

        # 提取RSI列
        rsi_column = f'rsi{period}'
        if rsi_column not in data_with_rsi.columns:
            return jsonify({
                'success': False,
                'message': f"计算RSI指标失败: 找不到{rsi_column}列"
            }), 500

        # 准备结果数据
        # 首先保存原始索引名称
        original_index_name = data_with_rsi.index.name or 'date'

        # 最近30天的RSI值
        rsi_values = data_with_rsi[[rsi_column]].tail(30).copy()
        rsi_values = rsi_values.reset_index()

        # 构建干净的结果记录列表
        result_records = []
        for _, row in rsi_values.iterrows():
            record = {}
            # 处理日期/索引
            index_value = row[original_index_name]
            if pd.api.types.is_datetime64_any_dtype(index_value) or isinstance(index_value, pd.Timestamp):
                record[original_index_name] = index_value.strftime('%Y-%m-%d')
            else:
                record[original_index_name] = str(index_value)

            # 处理RSI值
            rsi_value = row[rsi_column]
            record[rsi_column] = float(rsi_value) if not pd.isna(rsi_value) else None

            # 如果存在背离数据，添加它
            if 'rsi_divergence' in data_with_rsi.columns:
                # 找到对应索引的背离数据
                idx = data_with_rsi.index.get_loc(row[original_index_name]) if row[
                                                                                   original_index_name] in data_with_rsi.index else None
                if idx is not None:
                    divergence = data_with_rsi['rsi_divergence'].iloc[idx]
                    if divergence is not None:
                        # 确保背离数据是可序列化的
                        if isinstance(divergence, dict):
                            # 转换所有NumPy类型为Python原生类型
                            clean_divergence = {}
                            for k, v in divergence.items():
                                if isinstance(v, (np.integer, np.floating)):
                                    clean_divergence[k] = float(v)
                                elif isinstance(v, (np.bool_)):
                                    clean_divergence[k] = bool(v)
                                else:
                                    clean_divergence[k] = v
                            record['rsi_divergence'] = clean_divergence
                        else:
                            record['rsi_divergence'] = str(divergence)

            result_records.append(record)

        # 分析最新的RSI值
        latest_rsi = float(data_with_rsi[rsi_column].iloc[-1]) if not pd.isna(
            data_with_rsi[rsi_column].iloc[-1]) else None

        rsi_analysis = RSIIndicator.analyze_rsi_conditions(
            data_with_rsi[rsi_column],
            overbought=70,
            oversold=30
        )

        # 确保RSI分析中的所有值都是可序列化的
        clean_analysis = {}
        for k, v in rsi_analysis.items():
            if isinstance(v, (np.integer, np.floating)):
                clean_analysis[k] = float(v)
            elif isinstance(v, (np.bool_)):
                clean_analysis[k] = bool(v)
            else:
                clean_analysis[k] = v

        # 测试数据是否可以被JSON序列化
        response_data = {
            'success': True,
            'code': code,
            'name': str(stock_name) if stock_name is not None else code,
            'period': period,
            'latest_rsi': latest_rsi,
            'rsi_analysis': clean_analysis,
            'data': result_records,
            'count': len(result_records)
        }

        # 测试序列化
        try:
            json.dumps(response_data)
        except TypeError as e:
            current_app.logger.error(f"JSON序列化错误: {e}")
            # 尝试更严格的清理
            response_data = {
                'success': True,
                'code': code,
                'name': str(stock_name) if stock_name is not None else code,
                'period': period,
                'latest_rsi': latest_rsi,
                'rsi_analysis': str(rsi_analysis),  # 转为字符串
                'data': str(result_records),  # 转为字符串
                'count': len(result_records)
            }

        return jsonify(response_data)

    except Exception as e:
        current_app.logger.error(f"测试RSI指标失败: {e}")
        import traceback
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f"测试RSI指标失败: {str(e)}"
        }), 500