import OllamaClient_ST

client = OllamaClient_ST.OllamaClient()  
client.load()
while True:
    try:
        prompt = input("master：").strip()

        if prompt == "再见":
            print("solena：……再见,master")
            break
        elif prompt == "clear":
            client.backup()
            client.clear_history()
            continue
        elif not prompt:
            continue

        client.chat(prompt)
        print()
        
    except KeyboardInterrupt:
        print("solena：……再见,master")
        break
    except Exception as e:
        print(f"请求错误：{e}")