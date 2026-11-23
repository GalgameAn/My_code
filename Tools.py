from datetime import datetime
from urllib.parse import quote 
import requests


def get_current_time():
    """报时"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #datetime.now()>>获取当前时间  strftime("%Y-%m-%d %H:%M:%S")>>转换 年-月-日 时-分-秒 的格式


def calculator(operation, a, b):
    """计算器函数：支持加减乘除"""
    try:
        operation = operation.strip().lower()
        a = float(a)
        b = float(b)
        if operation == '+':
            return str(a + b)
        elif operation == '-':
            return str(a - b)
        elif operation == '×' or operation == '*':
            return str(a * b)
        elif operation == '÷' or operation == '/':
            if b == 0:
                return "错误：除数不能为零"
            return str(a / b)
        else:
            return f"错误：不支持的操作 '{operation}'，仅支持 + - × ÷"
    except (TypeError, AttributeError, ValueError) as e:
        return f"请求错误：{e}"

    
def get_weather(city='武汉'):
    """查询指定城市的天气"""
    url = f"https://wttr.in/{quote(city)}?format=3&lang=zh-cn"   # 加上 format=3 让它只返回一行简短天气
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.text.strip()
    except Exception:
        return f"暂时查不到{city}的天气"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间，当用户询问现在几点、今天几号时使用",
            "parameters": {     
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算，支持加减乘除。当用户需要计算数学表达式时使用",
            "parameters": {     # 创建一个参数包
                "type": "object",   # 让参数按照标准格式（键值对）排序的一个命令
                "properties": {     # 包含参数具体内容de'bao
                    'operation':{   # 参数名称
                        'type':'string',    # 参数类型
                        'description':'运算符，可选值：+（加）、-（减）、×（乘）、÷（除）'
                        },
                    'a':{
                        'type':'number',
                        'description':'第一个数'                         
                        },
                    'b':{
                        'type':'number',
                        'description':'第二个数',
                        }
                    },
                "required": ['operation', 'a', 'b']   #["必须提供的参数列表"]
            }
        }
    },
    {
        'type':'function',
        'function':{
            'name':'get_weather',
            'description':'查询指定城市天气',
            'parameters':{
                'type':'object',
                'properties':{
                    'city':{
                        'type':'string',
                        'description':'城市名称，如武汉、北京',
                    }
                },
                'required':['city']
            }
        }
    }
]