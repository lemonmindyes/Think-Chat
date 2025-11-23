import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn

from config import Config
from gpt import ThinkChat
from dataset import SFTDataset


if __name__ == '__main__':
    # param
    dtype = torch.bfloat16
    batch_size = 8
    base_lr = 5e-5
    warmup_ratio = 0.02
    grad_clip = 1.0
    accumulation_steps = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = nullcontext() if device.type == 'cpu' else torch.amp.autocast(device_type = device.type, dtype = dtype)

    config = Config()
    model = ThinkChat(config).to(device = device)
    model.load_state_dict(torch.load('pretrain_checkpoint.bin')['model'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total params:{total_params}')

    data_path = 'D:/think-dataset/think-llm-instruct-follow' # 5546514 lines
    tokenizer_path = './model/think_tokenizer'
    train_loader = SFTDataset(config, data_path, tokenizer_path, lru_size = 32, reuse_count = 128)

    loss_func = nn.CrossEntropyLoss(reduction = 'none')
    scaler = torch.amp.GradScaler(enabled = (device.type == 'cuda'))
    opt = torch.optim.AdamW(model.parameters(), lr = base_lr, betas = (0.9, 0.95), weight_decay = 0.0)

    total_step = 30000
    warmup_step = int(warmup_ratio * total_step)
    if not os.path.exists('sft_train_log.txt'):
        with open('sft_train_log.txt', 'a', encoding='utf-8') as f:
            f.write(f'Step, Loss, Lr, Time\n')
    try:
        checkpoint = torch.load('sft_checkpoint.bin')
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        scaler.load_state_dict(checkpoint['scaler'])
    except:
        checkpoint = {'current_step': 0}
    current_step = checkpoint['current_step']

    start_time = time.time()
    for step in range(checkpoint['current_step'], total_step):
        model.train()

        if step < warmup_step:
            lr = base_lr * current_step / warmup_step
        else:
            progress = (current_step - warmup_step) / (total_step - warmup_step)
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

        # update lr
        for param_group in opt.param_groups:
            param_group['lr'] = lr

        x, y, attn_mask, loss_mask = train_loader.sample(batch_size)
        x, y, attn_mask, loss_mask = x.to(device), y.to(device), attn_mask.to(device), loss_mask.to(device)

        with ctx:
            out = model(x, start_pos = 0, attn_mask = attn_mask, kv_cache = None)
            loss = loss_func(
                out.view(-1, out.shape[-1]),
                y.view(-1)
            ).view(y.shape)
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss /= accumulation_steps

        scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(opt)
            scaler.update()

            opt.zero_grad(set_to_none=True)

        current_step += 1

        if step % 20 == 0:
            print(f'Step:{step + 1}/{total_step}, Loss:{loss.item() * accumulation_steps:.4f}, '
                  f'Lr:{opt.param_groups[0]["lr"]:.6f}, Time:{time.time() - start_time:.2f}')
            with open('sft_train_log.txt', 'a', encoding ='utf-8') as f:
                f.write(f'{step + 1}, {loss.item() * accumulation_steps:.4f}, {opt.param_groups[0]["lr"]:.6f}, '
                        f'{time.time() - start_time:.2f}\n')

        if step % 100 == 0:
            from transformers import AutoTokenizer
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained('./model/think_tokenizer')

            text = f'<im_start>user\n写一篇关于人工智能的未来发展的文章。\n<im_end>\n<im_start>assistant\n'

            encoder = tokenizer.batch_encode_plus([text])
            input_ids = torch.tensor(encoder['input_ids'], dtype = torch.long, device = device)
            attention_mask = torch.tensor(encoder['attention_mask'], dtype = torch.long, device = device)
            # print(input_ids)
            # print(attention_mask)
            kv_cache = {}
            out = None
            pred = []
            start_pos = 0
            n = input_ids.shape[1]
            for i in range(config.max_seq_len):
                with torch.no_grad():
                    out = model(input_ids, start_pos, attention_mask, kv_cache)
                pred.append(tokenizer.decode(out[0:, -1].argmax(dim=-1)[0].item()))
                if pred[-1] == '<im_end>':
                    break

                input_ids = out[:, -1:].argmax(dim=-1)
                attention_mask = None
                if i == 0:
                    start_pos = n
                else:
                    start_pos += 1
                # print(start_pos)
                # print(kv_cache[0]['k'].shape, kv_cache[0]['v'].shape)
            print(text + ''.join(pred))

        if step % 1000 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'scaler': scaler.state_dict(),
                'current_step': current_step
            }
            torch.save(checkpoint, 'sft_checkpoint.bin.pt')
            os.replace('sft_checkpoint.bin.pt', 'sft_checkpoint.bin')

    checkpoint = {
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'current_step': current_step
    }
    torch.save(checkpoint, 'sft_checkpoint.bin.pt')
    os.replace('sft_checkpoint.bin.pt', 'sft_checkpoint.bin')



