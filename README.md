# NLP-Comp-Template
nlp 比赛代码模版

## 文件结构
--net 总体模型文件  
--layers 公共网络模块  
--engine 训练、验证和推断模块  
--metrics 指标模块  
--tools 训练与测试入口  
--config 实验配置模块  
--utils 工具类模块  
--preprocessing 预处理模块  
--ensamble 集成模块  
--pipeline 多模执行模块  
--pt_models 预训练模型  
--log 实验日志  
--cache 实验模型保存  
--submit 提交文件夹  
--raw_data 训练、测试、数据  
&emsp;&emsp;|--train  
&emsp;&emsp;|--test  
--run.sh 运行脚本


## TODO
1.程序入口
2.统一metric模块 -> 添加随step打印指标
3.实现pipeline模块
4.inference模块