import json
import os
import random
from functools import lru_cache

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import Config


# Mini
# zh_tokens: 598762007
# en_tokens: 662435561
# mix_tokens: 0
# total_tokens: 1261197568
# zh_lines: 1969064
# en_lines: 3030936
# mix_lines: 0
# total_lines: 5000000
# zh_avg_tokens_per_line: 304.08
# en_avg_tokens_per_line: 218.56
# avg_tokens_per_line: 252.24
# Full
# zh_tokens: 4046195821
# en_tokens: 4990611528
# mix_tokens: 0
# total_tokens: 9036807349
# zh_lines: 14471506
# en_lines: 22778494
# mix_lines: 0
# total_lines: 37250000
# zh_avg_tokens_per_line: 279.60
# en_avg_tokens_per_line: 219.09
# avg_tokens_per_line: 242.60
class PretrainDataset(Dataset):

    def __init__(self, config: Config, data_path, tokenizer_path, lru_size = 32, reuse_count = 16):
        super().__init__()
        self.config = config
        self.file_paths = [f'{data_path}/{name}' for name in os.listdir(data_path)]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.start_id = self.tokenizer.encode('<im_start>')[0]
        self.end_id = self.tokenizer.encode('<im_end>')[0]
        self.lru_size = lru_size # lru大小

        self.reuse_count = reuse_count # 文件重复使用次数
        self.cur_reuse_count = 0 # 当前文件重复使用次数
        self._cache_data = None # 暂存当前data

        @lru_cache(maxsize = self.lru_size)
        def _load(path):
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    text = json.loads(line.strip())['text']
                    data.append(f'<im_start>{text.strip()}<im_end>')
            return data
        self.load_file = _load

    def _load_new_file(self):
        file_path = random.choice(self.file_paths)
        data = self.load_file(file_path)
        # 更新缓存
        self._cache_data = data
        self.cur_reuse_count = 0
        return data

    def get_loss_mask(self, input_ids, attention_mask):
        b, n = input_ids.shape
        loss_mask = torch.zeros_like(input_ids)

        for i in range(b):
            start_pos = (input_ids[i] == self.start_id).nonzero(as_tuple=True)[0]
            end_pos = (input_ids[i] == self.end_id).nonzero(as_tuple=True)[0]

            if len(start_pos) == 0 or len(end_pos) == 0:
                loss_mask[i] = attention_mask[i]
                continue

            start_pos = start_pos[0].item()
            end_pos = end_pos[0].item()
            loss_mask[i, start_pos: end_pos + 1] = 1
        return loss_mask

    def sample(self, batch_size):
        if self._cache_data is None or self.cur_reuse_count >= self.reuse_count:
            data = self._load_new_file()
        else:
            data = self._cache_data
        self.cur_reuse_count += 1

        idx = torch.randint(0, len(data), (batch_size,)).tolist()
        batch = [data[i] for i in idx]
        # print(batch)
        encoder = self.tokenizer.batch_encode_plus(batch,
                                                   max_length = self.config.max_seq_len,
                                                   truncation = True,
                                                   padding = 'max_length',
                                                   return_tensors = 'pt',
                                                   )
        input_ids = encoder['input_ids']
        attention_mask = encoder['attention_mask']
        loss_mask = self.get_loss_mask(input_ids, attention_mask)

        # print(self.tokenizer.decode(input_ids[0]))
        # print(self.tokenizer.decode(input_ids[0][loss_mask[0] == 1]))
        # print("===========================")
        # print(self.tokenizer.decode(input_ids[1]))
        # print(self.tokenizer.decode(input_ids[1][loss_mask[1] == 1]))
        x = input_ids[:, :-1]
        y = input_ids[:, 1:]
        attn_mask = attention_mask[:, :-1]
        loss_mask = loss_mask[:, 1:]
        # print(self.tokenizer.decode(x[0]))
        # print("=====================")
        # print(self.tokenizer.decode(y[0]))
        # print("=====================")
        # print(self.tokenizer.decode(y[0][loss_mask[0] == True]))
        return x, y, attn_mask, loss_mask


class SFTDataset(Dataset):

    def __init__(self, config: Config, data_path, tokenizer_path, lru_size = 32, reuse_count = 16):
        self.config = config
        self.file_paths = [f'{data_path}/{name}' for name in os.listdir(data_path)]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.start_id = self.tokenizer.encode('<im_start>assistant')
        self.end_id = self.tokenizer.encode('<im_end>')
        self.lru_size = lru_size  # lru大小

        self.reuse_count = reuse_count  # 文件重复使用次数
        self.cur_reuse_count = 0  # 当前文件重复使用次数
        self._cache_data = None  # 暂存当前data

        @lru_cache(maxsize=self.lru_size)
        def _load(path):
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    text = json.loads(line.strip())['conversions']
                    tmp = ''
                    for i, v in enumerate(text):
                        tmp += f'<im_start>{v["from"].strip()}\n{v["content"].strip()}\n<im_end>\n'
                    data.append(tmp)
            return data

        self.load_file = _load

    def _load_new_file(self):
        file_path = random.choice(self.file_paths)
        data = self.load_file(file_path)
        # 更新缓存
        self._cache_data = data
        self.cur_reuse_count = 0
        return data

    def get_loss_mask(self, input_ids, attention_mask):
        b, n = input_ids.shape
        loss_mask = torch.zeros_like(input_ids)

        start_id = self.start_id[0]
        end_id = self.end_id[0]
        for i in range(b):
            start_pos = (input_ids[i] == start_id).nonzero(as_tuple=True)[0]
            end_pos = (input_ids[i] == end_id).nonzero(as_tuple=True)[0]

            if len(start_pos) == 0 or len(end_pos) == 0:
                loss_mask[i] = attention_mask[i]
                continue

            for s, e in zip(start_pos[1::2], end_pos[1::2]):
                s, e = s.item() + len(self.start_id), e.item()
                loss_mask[i, s: e + 1] = 1
            if len(start_pos) != len(end_pos):
                loss_mask[i, start_pos[-1].item() + len(self.start_id):] = 1
        return loss_mask

    def sample(self, batch_size):
        if self._cache_data is None or self.cur_reuse_count >= self.reuse_count:
            data = self._load_new_file()
        else:
            data = self._cache_data

        idx = torch.randint(0, len(data), (batch_size,)).tolist()
        batch = [data[i] for i in idx]
        encoder = self.tokenizer.batch_encode_plus(batch,
                                                   max_length=self.config.max_seq_len,
                                                   truncation=True,
                                                   padding='max_length',
                                                   return_tensors='pt',
                                                   )
        input_ids = encoder['input_ids']
        attention_mask = encoder['attention_mask']
        loss_mask = self.get_loss_mask(input_ids, attention_mask)
        # print(self.tokenizer.decode(input_ids[0]))
        # print(self.tokenizer.decode(input_ids[0][loss_mask[0] == 1]))
        # print("===========================")
        # print(self.tokenizer.decode(input_ids[1]))
        # print(self.tokenizer.decode(input_ids[1][loss_mask[1] == 1]))

        # print(batch[0])
        # print("=====================")
        # print(self.tokenizer.decode(input_ids[0][loss_mask[0] == True]))
        x = input_ids[:, :-1]
        y = input_ids[:, 1:]
        attn_mask = attention_mask[:, :-1]
        loss_mask = loss_mask[:, 1:]
        return x, y, attn_mask, loss_mask


class DPODataset(Dataset):

    def __init__(self, config, data_path, tokenizer_path, lru_size = 32, reuse_count = 128):
        self.config = config
        self.file_paths = [f'{data_path}/{name}' for name in os.listdir(data_path)]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.start_id = self.tokenizer.encode('<im_start>')[0]
        self.end_id = self.tokenizer.encode('<im_end>')[0]
        self.lru_size = lru_size  # lru大小

        self.reuse_count = reuse_count  # 文件重复使用次数
        self.cur_reuse_count = 0  # 当前文件重复使用次数
        self._cache_data = None  # 暂存当前data

        @lru_cache(maxsize=self.lru_size)
        def _load(path):
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    text = json.loads(line.strip())
                    chosen = text['chosen']
                    chosen1 = f'<im_start>{chosen[0]["from"]}\n{chosen[0]["content"]}\n<im_end>\n'
                    chosen2 = f'<im_start>{chosen[1]["from"]}\n{chosen[1]["content"]}\n<im_end>\n'
                    chosen = chosen1 + chosen2

                    rejected = text['rejected']
                    rejected1 = f'<im_start>{rejected[0]["from"]}\n{rejected[0]["content"]}\n<im_end>\n'
                    rejected2 = f'<im_start>{rejected[1]["from"]}\n{rejected[1]["content"]}\n<im_end>\n'
                    rejected = rejected1 + rejected2
                    data.append((chosen, rejected))
            return data

        self.load_file = _load

    def _load_new_file(self):
        file_path = random.choice(self.file_paths)
        data = self.load_file(file_path)
        # 更新缓存
        self._cache_data = data
        self.cur_reuse_count = 0
        return data

    def get_loss_mask(self, input_ids, attention_mask):
        b, n = input_ids.shape
        loss_mask = torch.zeros_like(input_ids)

        for i in range(b):
            start_pos = (input_ids[i] == self.start_id).nonzero(as_tuple=True)[0]
            end_pos = (input_ids[i] == self.end_id).nonzero(as_tuple=True)[0]

            if len(start_pos) == 0 or len(end_pos) == 0:
                loss_mask[i] = attention_mask[i]
                continue

            for s, e in zip(start_pos[1::2], end_pos[1::2]):
                s, e = s.item(), e.item()
                loss_mask[i, s: e + 1] = 1
            if len(start_pos) != len(end_pos):
                loss_mask[i, start_pos[-1].item():] = 1

            # print(self.tokenizer.decode(input_ids[i]))
            # print("===========================================")
            # for s, e in zip(start_pos[1::2], end_pos[1::2]):
            #     s, e = s.item(), e.item()
            #     print(input_ids[i][s: e + 1])
            # if len(start_pos) != len(end_pos):
            #     print(input_ids[i][start_pos[-1].item():])
            # print("===========================================")
            # for s, e in zip(start_pos[1::2], end_pos[1::2]):
            #     s, e = s.item(), e.item()
            #     print(self.tokenizer.decode(input_ids[i][s: e + 1]))
            # if len(start_pos) != len(end_pos):
            #     print(self.tokenizer.decode(input_ids[i][start_pos[-1].item():]))
            # print("===========================================")
            # print(self.tokenizer.decode(input_ids[i][loss_mask[i] == 1]))
            # print("===========================================")
            # tmp = ''
            # for s, e in zip(start_pos[1::2], end_pos[1::2]):
            #     s, e = s.item(), e.item()
            #     tmp += self.tokenizer.decode(input_ids[i][s: e + 1])
            # if len(start_pos) != len(end_pos):
            #     tmp += '\n'
            #     tmp += self.tokenizer.decode(input_ids[i][start_pos[-1].item():])
            # print(tmp == self.tokenizer.decode(input_ids[i][loss_mask[i] == 1]))
            # print("===========================================\n")
        return loss_mask

    def sample(self, batch_size):
        if self._cache_data is None or self.cur_reuse_count >= self.reuse_count:
            data = self._load_new_file()
        else:
            data = self._cache_data

        idx = torch.randint(0, len(data), (batch_size,)).tolist()
        chosen_batch = [data[i][0] for i in idx]
        rejected_batch = [data[i][1] for i in idx]
        chosen_encoder = self.tokenizer.batch_encode_plus(chosen_batch,
                                                          max_length=self.config.max_seq_len,
                                                          truncation=True,
                                                          padding='max_length',
                                                          return_tensors='pt',
                                                          )
        rejected_encoder = self.tokenizer.batch_encode_plus(rejected_batch,
                                                            max_length=self.config.max_seq_len,
                                                            truncation=True,
                                                            padding='max_length',
                                                            return_tensors='pt',
                                                            )
        chosen_input_ids = chosen_encoder['input_ids']
        chosen_attention_mask = chosen_encoder['attention_mask']
        chosen_loss_mask = self.get_loss_mask(chosen_input_ids, chosen_attention_mask)
        rejected_input_ids = rejected_encoder['input_ids']
        rejected_attention_mask = rejected_encoder['attention_mask']
        rejected_loss_mask = self.get_loss_mask(rejected_input_ids, rejected_attention_mask)
        print(self.tokenizer.decode(chosen_input_ids[0]))
        print("==========================================================")
        print(self.tokenizer.decode(chosen_input_ids[0][chosen_loss_mask[0] == 1]))
        print("===========================")
        print(self.tokenizer.decode(rejected_input_ids[1]))
        print("==========================================================")
        print(self.tokenizer.decode(rejected_input_ids[1][rejected_loss_mask[1] == 1]))

        x_chose = chosen_input_ids[:, :-1]
        y_chose = chosen_input_ids[:, 1:]
        x_attn_mask = chosen_attention_mask[:, :-1]
        x_loss_mask = chosen_loss_mask[:, 1:]
        x_reject = rejected_input_ids[:, :-1]
        y_reject = rejected_input_ids[:, 1:]
        x_reject_attn_mask = rejected_attention_mask[:, :-1]
        x_reject_loss_mask = rejected_loss_mask[:, 1:]
        return x_chose, y_chose, x_attn_mask, x_loss_mask, x_reject, y_reject, x_reject_attn_mask, x_reject_loss_mask


if __name__ == '__main__':
    from config import Config
    config = Config()
    tokenizer_path = './model/think_tokenizer'

    data_path = 'D:/think-dataset/think-llm-pretrain/Mini'
    train_loader = PretrainDataset(config, data_path, tokenizer_path, lru_size = 32, reuse_count = 128)
    train_loader.sample(2)
    # x, y, attn_mask, loss_mask = train_loader.sample(1)
    # print(x.shape, y.shape, attn_mask.shape, loss_mask.shape)

    # data_path = 'D:/think-dataset/think-llm-instruct-follow'
    # train_loader = SFTDataset(config, data_path, tokenizer_path, lru_size = 32, reuse_count = 128)
    # train_loader.sample(2)

    # data_path = 'D:/think-dataset/think-llm-dpo'
    # config.max_seq_len = 2048
    # train_loader = DPODataset(config, data_path, tokenizer_path, lru_size = 32, reuse_count = 128)
    # train_loader.sample(32)