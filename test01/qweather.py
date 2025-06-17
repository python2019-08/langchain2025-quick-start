def get_weather(city,api_key): 
    #和风天气API的URL
    url ="https://devapi.qweather.com/v7/weather/now"
    #请求参数
    params = {
    'location':city, # 城市名称或ID
    'key':api_key # 你的API密钥
    }
    #发送GET请求
    response = requests.get(url, params=params)
    #检查请求是否成功
    if response.status_code == 200:
        #解析JSON响应
        data = response.json()
        if data['code'] =='200':
            #提取天气信息
            weather_info = data['now']
            print(f"城市：{city}")
            print(f"天气状况：{weather_info['text']}")
            print(f"温度：{weather_info['temp']}C")
            print(f"体感温度：{weather_info['feelsLike']}C")
            print(f"相对湿度：{weather_info['humidity']}%")
            print(f"降水量：{weather_info['precip']}mm")
            print(f"风向: {weather_info['windDir']}")
            print(f"风速:{weather_info['windSpeed']}km/h")
        else:
            print(f"错误：{data['code']}-{data['msg']}")
    else:
        print(f"请求失败,状态码：{response.status_code}")


#使用示例
api_key ='b5f9a4ed60904ca7adab0a2bba3645ad'# 申请的API密钥
city='101010100' #北京的城市ID
get_weather(city, api_key)        