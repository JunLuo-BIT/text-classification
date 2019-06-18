# text-classification
</b>
Using TextCNN、LSTM、GRU、Attention、Transformer 进行文本分类
</b>

## 一、TextCNN 
### 1、超参数
`embedding_dim`:  `0.1`
`filter_sizes`: `128`
`num_filters`: `3,4,5`
`dropout_keep_prob`: `0.5`
`l2_reg_lambda`: `0.01`

### 2、训练参数
`batch_size`: `64`
`num_epochs`: `200`
`evaluate_every`: `500`  

### 3、结果
 最终的  `accuracy` : `0.962852`