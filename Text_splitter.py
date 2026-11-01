def split_text(text, chunk_size=500, overlap=50):
    """切割长文本"""
    chunks = []     # 创建模块库库
    start = 0
    text_legth = len(text)

    while start < text_legth:
        # 确定结束位置
        end = start + chunk_size

        # 截快并存储
        chunk = text[start:end]
        chunks.append(chunk)

        if end >= text_legth:
            break

        start = end - overlap   # 下一次截取位置

    return chunks

if  __name__ == "__main__":
    long_text = "0123456789" * 20
    result = split_text(long_text, chunk_size=50, overlap=5)

    for i, chunk in enumerate(result):
        print(f"第{i+1}块，长度：{len(chunk)}，内容开头：{chunk[:10]}...")

