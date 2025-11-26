import torch
from transformers import AutoTokenizer

from config import Config
from gpt import ThinkChat


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = Config()
    model = ThinkChat(config).to(device=device)
    model.load_state_dict(torch.load('pretrain_checkpoint.bin')['model'])
    # model.load_state_dict(torch.load('sft_checkpoint.bin')['model'])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('./model/think_tokenizer')

    text = f'<im_start>MildMonopoly is a website intended to deliver news related to'
    # text = f'<im_start>user\n为什么在催化剂的作用下，不同分子间的反应速率可以变得相同？\n<im_end>\n<im_start>assistant\n'

    encoder = tokenizer.batch_encode_plus([text])
    input_ids = torch.tensor(encoder['input_ids'], dtype = torch.long, device = device)
    attention_mask = torch.tensor(encoder['attention_mask'], dtype = torch.long, device = device)
    # print(input_ids)
    # print(attention_mask)
    kv_cache = {}
    start_pos = 0
    # n = input_ids.shape[1]
    # for i in range(config.max_seq_len):
    out = model.generate(input_ids, start_pos, attention_mask, kv_cache, temperature = 0.7, top_p = 0.85, rp = 1.02, eos_token_id = 2)
    print(out)
    print(out.shape)
    pred = tokenizer.decode(out[0, :])
    print(text + ''.join(pred))

    #     with torch.no_grad():
    #         out = model(input_ids, start_pos, attention_mask, kv_cache)
    #     pred.append(tokenizer.decode(out[0:, -1].argmax(dim=-1)[0].item()))
    #     if pred[-1] == '<im_end>':
    #         break
    #
    #     input_ids = out[:, -1:].argmax(dim=-1)
    #     attention_mask = None
    #     if i == 0:
    #         start_pos = n
    #     else:
    #         start_pos += 1
    #     # print(start_pos)
    #     # print(kv_cache[0]['k'].shape, kv_cache[0]['v'].shape)
    # print(text + ''.join(pred))