import os.path
import getpass
import json

"""
`getpass`是Python标准库中的模块，用于安全地获取用户输入（尤其是密码等敏感信息），需要在使用前显式导入。

***示例： 
```python
# load_key.py
import getpass  # 添加这一行

def load_key(key_name):
    # 其他代码保持不变...
    keyval = getpass.getpass("配置文件中没有相应配置，请输入对应配置信息:").strip()
    # 其他代码...
``` 
"""

def load_key(keyname: str) -> object:
    file_name = "cfg/keys.json"
    if os.path.exists(file_name):
        with open(file_name,"r") as file:
            Key = json.load(file)
        if keyname in Key and Key[keyname]:
            return Key[keyname]
        else:
            keyval=getpass.getpass("配置文件中没有相应就，请输入对应配置信息：").strip()
            Key[keyname] = keyval
            with open(file_name,"w") as file:
                json.dump(Key,file,indent=4)
            return keyval
    else:
        keyval = getpass.getpass("配置文件中没有相应就，请输入对应配置信息:").strip()
        Key = {
            keyname:keyval
        }
        with open(file_name,"w") as file:
            json.dump(Key,file,indent=4)
        return keyval

if __name__ == "__main__":
    print(Load_key("LANGSMITH_API_KEY2"))