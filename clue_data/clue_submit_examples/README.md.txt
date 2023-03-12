# 中文任务基准测评 CLUE benchmark 任务提交说明

--------------------更新：2021-09-04：进一步明确CLUE1.1版本的提交规范----------------------
提交文件统一为 <任务名>_predict.json，除非更新了测试集的任务：
1）tnews: 
1.1版本的提交文件名：tnews11_predict.json; 1.0提交文件名：tnews10_predict.json
2）cluewsc2020: 
1.1版本的提交文件名：cluewsc11_predict.json; 1.0提交文件名：cluewsc10_predict.json
3）c3: 
1.1版本的提交文件名：c311_predict.json; 1.0提交文件名：c310_predict.json
4）chid: 
1.1版本的提交文件名：chid11_predict.json; 1.0提交文件名：chid10_predict.json

总结：提交文件总体结构为： <任务名>版本_predict.json
------------------------------------------------------------------------------------



基准测评包含9个任务，其中分类或句子对任务6个，阅读理解型任务3个

## 提交文件的命名规范
所有提交文件命名规范：
- 1、包含任务集的小写字母，如提交 afqmc的数据集预测结果文件命名需满足：afqmc_predict.json
- 2、把所有任务的预测结果打包成zip压缩（文件名称为submit_myteamname_mydate.zip）提交（推荐）
- 3、用户也可以不提交某个任务的数据集，只需提交已经做了的任务预测文件即可，如只提交分类任务或阅读理解任务。

## 分类数据集

```
AFQMC	TNEWS'	IFLYTEK' OCNLI	WSC	CSL
```
这几个数据集提交的格式，每一行为一个json string, 需要带上 label值, 严格按照 预测的输入文件 test.json 对应的行顺序进行输出（打分过程中只会读取label里面的属性值，与真实值逐个比较），参考格式如下：
```
{"id": 0, "label": "0"}
{"id": 1, "label": "1"}
```

## 阅读理解数据集

整个文件为 一个json，可以直接使用 json.loads(open(file_path).read())，参考格式如下：
```json
{
    "TEST_0_QUERY_0": "美国海军提康德罗加级",
    "TEST_0_QUERY_1": "第二艘",
    "TEST_0_QUERY_2": "1862年"
}
```

## 诊断集上预测结果的提交（可选,推荐提交）

诊断集，用于评估不同模型在9种语言学家总结的中文语言现象上的表现

使用在OCNLI上训练过的模型，直接预测在这个诊断集上的结果，提交格式和OCNLI一致。


## 技术支持与问题讨论

官方网站：www.CLUEbenchmarks.com

项目地址：https://github.com/cluebenchmark/CLUE

联系邮箱：CLUEbenchmark@163.com；QQ技术交流群:836811304


