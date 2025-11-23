import json
import os
import random
from functools import lru_cache

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import Config


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
                    data.append(f'<im_start>{text}<im_end>')
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
            # print(self.tokenizer.decode(input_ids[i]))
            # print(input_ids[i][start_pos: end_pos + 1])
            # print(self.tokenizer.decode(input_ids[i][start_pos: end_pos + 1]))
            # print(self.tokenizer.decode(input_ids[i][loss_mask[i] == 1]))
            # print(self.tokenizer.decode(input_ids[i][loss_mask[i] == 1]) == self.tokenizer.decode(input_ids[i][start_pos: end_pos + 1]))
        return loss_mask

    def sample(self, batch_size):
        if self._cache_data is None or self.cur_reuse_count >= self.reuse_count:
            data = self._load_new_file()
        else:
            data = self._cache_data
        self.cur_reuse_count += 1

        idx = torch.randint(0, len(data), (batch_size,)).tolist()
        batch = [data[i] for i in idx]
        encoder = self.tokenizer.batch_encode_plus(batch,
                                                   max_length = self.config.max_seq_len,
                                                   truncation = True,
                                                   padding = 'max_length',
                                                   return_tensors = 'pt',
                                                   )
        input_ids = encoder['input_ids']
        attention_mask = encoder['attention_mask']
        loss_mask = self.get_loss_mask(input_ids, attention_mask)

        x = input_ids[:, :-1]
        y = input_ids[:, 1:]
        attn_mask = attention_mask[:, :-1]
        loss_mask = loss_mask[:, 1:]
        return x, y, attn_mask, loss_mask


class SFTDataset(Dataset):

    def __init__(self, config: Config, data_path, tokenizer_path, lru_size = 32, reuse_count = 16):
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
                    text = json.loads(line.strip())['conversions']
                    tmp = ''
                    for i, v in enumerate(text):
                        tmp += f'<im_start>{v["from"]}\n{v["content"]}\n<im_end>\n'
                        # if i != len(text) - 1:
                        #     tmp += f'<im_start>{v["from"]}\n{v["content"]}\n<im_end>\n'
                        # else:
                        #     tmp += f'<im_start>{v["from"]}\n{v["content"]}\n<im_end>'
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
        batch = [data[i] for i in idx]
        encoder = self.tokenizer.batch_encode_plus(batch,
                                                   max_length=self.config.max_seq_len,
                                                   truncation=True,
                                                   padding='max_length',
                                                   return_tensors='pt',
                                                   )
        input_ids = encoder['input_ids']
        attention_mask = encoder['attention_mask']
        # input_ids = torch.tensor([[    1,   499,   283,   202,  1099,  2837,   935,   457,   709,  1188,
        #  1669,  2062,   712,   380,   571,   527,  6919,  1993, 11142,   276,
        #   202,     2,   202,     1,  1357,   760,   831,   202,   709,  1188,
        #  1669,  2062,   712,  8311, 10991,  4366,   554,   447,  2671,   262,
        #   703,  4314,   371,   496,   752,  1705,   889,   527,   607,   580,
        #  7348,  1627,   311,  1209,   889,   447,   420,  6457,  2899,   638,
        #   121,   670,  3753,   276,  1592,  1982,  5279,  4244,   527,   607,
        #   262,  1763,   425,  6028,  1616,  2823,   305,  1256,   862,   253,
        #  2823,   305,  2393,   253,  1616,   872,   249,   585,  5917,   371,
        #  1443,   262,  7347,  2837,  4141,  1985,  1912,  2687,   266,  1780,
        #  9919,  4076,  6358,   305,  7020,  5423,  9289,   393,  5917,   603,
        #  2986,  1599,   266,  7848,  3273,   262,  5636,  1929,   322,  2726,
        #   579,  1455,   266,  1490,  4489,   393,  5920,   270,   125,   276,
        #   224,  1993, 11142,  1986,   262,   507,  2303,   709,  1188,  1669,
        #  2062,   712,  1982,   524,  1100,  2790,  4011,  6783,   393,  1535,
        #   661,  5525,   266, 10991,  3718,   276,  1592,  9476,  1982,   524,
        #  7347,  2513,  7020,  2662,   607,   262,   723,  1982,   524,  5391,
        #  8692,   418,  9809,   266,  1535,  2554,  3753,   276,   324,   527,
        #   607,   345,   262,  6028,  1616,  2823,   305,  1256,   862,   253,
        #  2823,   585,  5917,   661,  4982,  6838,  2129,   305,  2384,  1025,
        #   108,   393,  1681,  1251,   262,  1631,  1315,   457,  9492,  1563,
        #   393,  7020,   266,   963,  1593,   393,  7676,   262,   518,  5817,
        #   463,  5342,  1839,   752,  3276,  1535,   345,  2043,  7472,   305,
        #   529,   800,   305,  2687,   305,  1455,  5682,   266,  1770,   681,
        #  1120,   393,  3879,  1120,   276,  9083,   262,   527,   607,   560,
        #  1345,   457,  5917,  4828,   305,  7433,  2304,   305,  4873,   305,
        #  1373,  3200,   585,  3505,  4101,   266,  7347,  2837,   262,  4496,
        #   322,   752,  3276,  1535,   266,  8384,   648,  4305,  1914,   276,
        #  1345,   457,  6028,  1237,   305,  1914,  1237,   585,  6726,   266,
        #  7347,  2513,   262,  1631,   457,   347,  1120,  1325,   585,  8355,
        #   266,  7347,  2837,   262,  4496,   322,  2726,   579,  1455,  1932,
        #  6726,  1535,   305,  7981,   409,  1535,   393,  4286,  1535,  5682,
        #   266,   940,  4918,  6636,   393,   335,  1321,  3123,   276,  1592,
        #  2176,  1982,   524, 10991,  3718,   262,   723,  1982,   524,  5058,
        #   752,  3276,  1535,  3567,  2358,   650,  2942,   276,  7343,   262,
        #   324,  5645,   709,  1188,  1669,  2062,   712,   397,   262,   467,
        #  3328, 10951,  1592,  8450,  2712,   606,   276,  3227,   262,   527,
        #   607,   342,  1353,  4607,  9159,  1535,   393,  2358,  5841,  5913,
        #   822,  1801,  8193,  2181,   563,  5604,  1565,  8317,   276,  3883,
        #   262,   527,  6919,  3240,   467,  1701,  7848,   262, 10991,  8001,
        #   393,  6535,  4437,   467,   648,  3408,   527,   607,  2208,   342,
        #  2196,   276,  2748,   262,  1470,  3109,  2181,   563,  1986,   262,
        #  5384,  6162,   527,   607,  1526,  1668,  2269,  4160,   393,  1082,
        #   554,   276,   224,   893,   266,  1986,  4282,  1188,  1669,  2062,
        #   712,  1982,   524,  2790,  3619,   412, 10991,  3879,  2662,   607,
        #   276,  1592,  2176,  5700,  1839,   752,  3276,  1535,   266,  1056,
        #  3820,  1480,  5314,   116,   262,  1299,   457,  2726,   579,  1455,
        #   345,   335,   811,   305,  7020,   305, 11665,  5682,   266,  5817,
        #  1423,  1663,   262,  2790,  1100,  4853,  1535,   393,  4011,  4867,
        #   276,   202,     333,     333,     333,     333,     333,     333,     333,     333,
        #     333,     333,     333,     333,     333,     333,     333,     333,     333,     333,
        #     333,     333]])
        loss_mask = self.get_loss_mask(input_ids, attention_mask)
        # print(self.tokenizer.decode(input_ids[0]))
        # print()
        # print(self.tokenizer.decode(input_ids[0][loss_mask[0] == 1]))
        # print("===========================")
        # print(self.tokenizer.decode(input_ids[1]))
        # print()
        # print(self.tokenizer.decode(input_ids[1][loss_mask[1] == 1]))

        x = input_ids[:, :-1]
        y = input_ids[:, 1:]
        attn_mask = attention_mask[:, :-1]
        loss_mask = loss_mask[:, 1:]
        return x, y, attn_mask, loss_mask


if __name__ == '__main__':
    from config import Config
    config = Config()
    # data_path = 'D:/think-dataset/think-llm-pretrain'
    tokenizer_path = './model/think_tokenizer'
    # train_loader = PretrainDataset(config, data_path, tokenizer_path, lru_size = 32, reuse_count = 128)
    #
    # x, y, attn_mask, loss_mask = train_loader.sample(1)
    # print(x.shape, y.shape, attn_mask.shape, loss_mask.shape)

    data_path = 'D:/think-dataset/think-llm-instruct-follow'
    train_loader = SFTDataset(config, data_path, tokenizer_path, lru_size = 32, reuse_count = 128)
    train_loader.sample(32)