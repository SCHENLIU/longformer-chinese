from classification import LongformerClassifier, ClassificationDataset
from collections import namedtuple
from transformers import BertTokenizer
import torch, time, sys
Argument = namedtuple("Argument", ['test_checkpoint', 'num_labels', "test_file", "sequence_length"])
args = Argument(test_checkpoint='models/version_0/checkpoints/ep-epoch=0_acc-acc=0.915.ckpt',
                num_labels=18,
                test_file = "data/train.txt",
                sequence_length = 4096)

class LongformerClassify():
    def __init__(self, mask_padding_with_zero=True, map_generate_from_file=True):
        self.data = []
        self._tokenizer = BertTokenizer.from_pretrained('longformer-chinese-base-4096/')
        self._tokenizer.model_max_length = args.sequence_length
        self.mask_padding_with_zero = mask_padding_with_zero
        self.seqlen = args.sequence_length
        if map_generate_from_file:
            self.produce_label_map()
        self.device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
        self.model = LongformerClassifier.load_from_checkpoint(args.test_checkpoint, num_labels=args.num_labels)
        self.model.to(self.device)

    def produce_label_map(self):
        data = []
        with open(args.test_file, encoding='UTF-8') as fin:
            for i, line in enumerate(fin):
                items = line.strip().split('\tSEP\t')
                if len(items) != 10: continue
                data.append({
                    "text": items[0]+items[1],
                    "label": items[5]
                })
        all_labels = list(set([e["label"] for e in data]))
        self.label_to_idx = {e: i for i, e in enumerate(sorted(all_labels))}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        print(self.label_to_idx)

    def _convert_to_tensors(self, instance):
        def tok(s):
            return self._tokenizer.tokenize(s)
        tokens = [self._tokenizer.cls_token] + tok(instance["text"])
        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        token_ids = token_ids[:self.seqlen-1] +[self._tokenizer.sep_token_id]
        input_len = len(token_ids)
        attention_mask = [1 if self.mask_padding_with_zero else 0] * input_len
        padding_length = self.seqlen - input_len
        token_ids = token_ids + ([self._tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if self.mask_padding_with_zero else 1] * padding_length)
        assert len(token_ids) == self.seqlen, "Error with input length {} vs {}".format(
            len(token_ids), self.seqlen
        )
        assert len(attention_mask) == self.seqlen, "Error with input length {} vs {}".format(
            len(attention_mask), self.seqlen
        )
        label = self.label_to_idx[instance["label"]] if instance["label"] else 0
        return (torch.tensor([token_ids]).to(self.device), torch.tensor([attention_mask]).to(self.device), torch.tensor([label]).to(self.device))

    def predict(self, text):
        instance = {"text": text, "label": None}
        token_ids, attention_mask, label = self._convert_to_tensors(instance=instance)
        logits = self.model(token_ids, attention_mask, label)[0]
        softmax = torch.nn.Softmax(dim=0)
        probabilities = softmax(logits.squeeze())
        res_list = sorted(zip(range(args.num_labels), probabilities.tolist()), key=lambda a: a[1], reverse=True)
        res_list = [(self.idx_to_label[i],str(j)) for i,j in res_list]
        return res_list[:5]





if __name__ == '__main__':
    classifier = LongformerClassify()
    input_file = sys.argv[1]
    fw = open("result.txt", "w")
    with open(input_file, "r") as fr:
        for item in fr:
            item = item.strip().split("\t|SEP|\t")
            if len(item) != 10:
                continue
            res = []
            for r in classifier.predict(item[8]+item[9]):
                res.extend(r)
            fw.write('\t|SEP|\t'.join(item+res)+'\n')
            fw.flush()
    fw.close()
    # for i in range(100):
    #   begin = time.time()
    #   print(classifier.predict("泰安市政协十三届三十四次 主席会议召开"))
    #   print(time.time() - begin)

