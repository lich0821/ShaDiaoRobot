# 群主的沙雕机器人
基于 [Rasa](https://github.com/rasaHQ/rasa) 的一个机器人示例。

## 环境搭建
### 安装 Python
官方说支持 `3.7`，`3.8`，`3.9` 和 `3.10`，但 `3.7` 太旧了；`3.10` 又要求蛮多；建议 `3.8` 或者 `3.9`。

### 创建虚拟环境
```sh
python -m venv .env

# 激活虚拟环境
source .env/bin/activate
```

### 安装 RASA 开源版
```sh
# 更新 pip
pip install -U pip

# 安装 RASA
pip install rasa
```

## 快速开始
### 克隆项目
```sh
git clone
```

### 安装依赖
```sh
pip install -r requirements.txt
```

### 训练模型
```sh
rasa train
# 训练时遇到网络报错的话，需要科学一下
```

### 运行 Action 服务
去[知心天气](https://www.seniverse.com/)注册一个账号，可以免费使用天气接口。

```sh
# 设置 KEY
export WEATHER_KEY=your_key

rasa run actions
```

### 启动 RASA API
```sh
rasa run --enable-api
```

### 验证
* cURL
```sh
curl --location 'http://localhost:5005/webhooks/rest/webhook' \
--data '{
    "sender": "user1",
    "message": "明天北京天气怎么样？"
}'
```

* Python
```py
import requests

url = "http://localhost:5005/webhooks/rest/webhook"

payload = "{\n    \"sender\": \"user1\",\n    \"message\": \"明天北京天气怎么样？\"\n}"
headers = {}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
```

## 从零创建项目
忽略一个 `SQLAlchemy` 告警：
```sh
export SQLALCHEMY_SILENCE_UBER_WARNING=1
```

### 初始化项目
```sh
rasa init
```

```txt
Welcome to Rasa! 🤖

To get started quickly, an initial project will be created.
If you need some help, check out the documentation at https://rasa.com/docs/rasa.
Now let's start! 👇🏽

? Please enter a path where the project will be created [default: current directory]
? Directory 'RasaDemo' is not empty. Continue? Yes
……
Created project directory at 'RasaDemo'.
Finished creating project structure.
? ? Do you want to train an initial model? 💪🏽 No
No problem 👍🏼. You can also train a model later by going to the project directory and running 'rasa train'.
```

这样就有一个示例项目；也可以选择训练跑跑。然后基示例项目改。

Once you've done the following, you can train your bot and try it out!

* Add RulePolicy to your policies and ResponseSelector to your pipeline in config.yml
* Add at least one rule for responding to FAQs/chitchat
* Add examples for your FAQs/chitchat intents
* Add responses for your FAQs/chitchat intents
* Update the intents in your domain

## 调试
```sh
rasa shell --debug
```
