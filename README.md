# <p align=center>`Longformer-chinese`</p>
All work is based on `Longformer`(https://github.com/allenai/longformer)

`Longformer-chinese` 提供了：基于BERT的中文预训练模型、在分类任务上的实现

### WHAT'S DIFFERENT

`Longformer-chinese` 基于BERT框架进行修改，在embedding层会与原版的稍有区别。加载时使用longformer.longformer：

```
from longformer.longformer import *
config = LongformerConfig.from_pretrained('schen/longformer-chinese-base-4096')
model = Longformer.from_pretrained('schen/longformer-chinese-base-4096', config=config)
```


 使用`schen/longformer-chinese-base-4096`会自动从transformers下载预训练模型，也可以自行下载后替换成所在目录：
 https://huggingface.co/schen/longformer-chinese-base-4096

### How to use

1. Download pretrained model
  * [`longformer-base-4096`](https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-base-4096.tar.gz)
  * [`longformer-large-4096`](https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-large-4096.tar.gz)

2. Install environment and code

    ```bash
    conda create --name longformer python=3.7
    conda activate longformer
    conda install cudatoolkit=10.0
    pip install git+https://github.com/allenai/longformer.git
    ```

3. Run the model

    ```python
    import torch
    from longformer.longformer import Longformer, LongformerConfig
    from longformer.sliding_chunks import pad_to_window_size
    from transformers import RobertaTokenizer

    config = LongformerConfig.from_pretrained('longformer-base-4096/') 
    # choose the attention mode 'n2', 'tvm' or 'sliding_chunks'
    # 'n2': for regular n2 attantion
    # 'tvm': a custom CUDA kernel implementation of our sliding window attention
    # 'sliding_chunks': a PyTorch implementation of our sliding window attention
    config.attention_mode = 'sliding_chunks'

    model = Longformer.from_pretrained('longformer-base-4096/', config=config)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.model_max_length = model.config.max_position_embeddings

    SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
 
    input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

    # TVM code doesn't work on CPU. Uncomment this if `config.attention_mode = 'tvm'`
    # model = model.cuda(); input_ids = input_ids.cuda()

    # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
    attention_mask[:, [1, 4, 21,]] =  2  # Set global attention based on the task. For example,
                                         # classification: the <s> token
                                         # QA: question tokens

    # padding seqlen to the nearest multiple of 512. Needed for the 'sliding_chunks' attention
    input_ids, attention_mask = pad_to_window_size(
            input_ids, attention_mask, config.attention_window[0], tokenizer.pad_token_id)

    output = model(input_ids, attention_mask=attention_mask)[0]
    ```

### Model pretraining

[This notebook](https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb) demonstrates our procedure for training Longformer starting from the RoBERTa checkpoint. The same procedure can be followed to get a long-version of other existing pretrained models. 

### TriviaQA

* Training scripts: `scripts/triviaqa.py`
* Pretrained large model: [`here`](https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/triviaqa-longformer-large.tar.gz) (replicates leaderboard results)
* Instructions: `scripts/cheatsheet.txt`


### CUDA kernel

Our custom CUDA kernel is implemented in TVM.  For now, the kernel only works on GPUs and Linux. We tested it on Ubuntu, Python 3.7, CUDA10, PyTorch >= 1.2.0. If it doesn't work for your environment, please create a new issue.

**Compiling the kernel**: We already include the compiled binaries of the CUDA kernel, so most users won't need to compile it, but if you are intersted, check `scripts/cheatsheet.txt` for instructions.


### Known issues

Please check the repo [issues](https://github.com/allenai/longformer/issues) for a list of known issues that we are planning to address soon. If your issue is not discussed, please create a new one. 


### Citing

If you use `Longformer` in your research, please cite [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150).
```
@article{Beltagy2020Longformer,
  title={Longformer: The Long-Document Transformer},
  author={Iz Beltagy and Matthew E. Peters and Arman Cohan},
  journal={arXiv:2004.05150},
  year={2020},
}
```

`Longformer` is an open-source project developed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
