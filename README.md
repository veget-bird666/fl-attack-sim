# fl-attack-sim
## 文件说明：
1.ml-latest-small为数据集，是不同用户的电影点击数据。
2.data_utils.py和model.py为客户端和服务端共有的，用于处理数据集，定义相应deepFM模型。
3.malicious_server.py是恶意服务器，它会泄露训练中的数据，保存每轮全局参数和每轮训练第一个连接的客户端的更新参数，默认保存至leaked_data里。
4.attack.py是根据leaked_data目录中的一个全局参数和一个更新参数进行推理攻击的脚本，目前需要用命令行执行，即需要用anaconda prompt的终端加载虚拟环境，在指定目录下输入：
python attack.py --global-weights ./leaked_data/global_round_1.npz --client-weights ./leaked_data/client_f4a008lljdwa8dla2_round_1.npz
的格式方可运行。
## 运行说明：
1.server.py可直接运行。
2.client.py需要命令行执行（需用anaconda prompt的终端加载虚拟环境，然后进入指定目录，输入:
python client.py --client-id 1  或
python client.py --client-id 2  等等，注意这里输入的id是用户名，对应rating.csv数据集里的用户id）。
3.这里为了方便，写了launch_clients.py，可直接加载10个终端自动按上述2的方式运行id从1到10的命令。
4.
