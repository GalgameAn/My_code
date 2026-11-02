import re

def desensitize(text):
    """文本脱敏"""
    # 手机号
    text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[敏感信息已脱敏]", text)    # (?<!\d) >>前面不能是数字,(?!\d) >>后面不能是数字

    # 身份证号
    text = re.sub(r"\d{17}[xX0-9]", "[敏感信息已脱敏]", text)

    # 账号密码
    text = re.sub(r"(password\s*=\s*|\s*密码是\s*|\s*账号是\s*)\S[^，,。！!？?\s]+", r"\1[敏感信息已脱敏]", text)

    return text
