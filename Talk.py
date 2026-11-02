import OllamaClient_ST

client = OllamaClient_ST.OllamaClient()  
client.load()
while True:
    try:
        prompt = input("安：").strip()

        if prompt == "再见":
            print("索蕾娜：……再见,安")
            break
        elif prompt == "备份":
            client.backup()
            client.clear_history()
            continue
        elif not prompt:
            continue

        client.chat(prompt)
        print()
        
    except KeyboardInterrupt:
        print("索蕾娜：……再见,安")
        break
    except Exception as e:
        print(f"请求错误：{e}")