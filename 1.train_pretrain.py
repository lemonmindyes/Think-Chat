import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn

from config import Config
from dataset import PretrainDataset
from gpt import ThinkChat


if __name__ == '__main__':
    # param
    dtype = torch.bfloat16
    batch_size = 32
    base_lr = 3e-4
    weight_decay = 0.1
    warmup_ratio = 0.03
    grad_clip = 1.0
    accumulation_steps = 8
    model_save_step = 200
    log_print_save_step = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = nullcontext() if device.type == 'cpu' else torch.amp.autocast(device_type = device.type, dtype = dtype)

    config = Config()
    model = ThinkChat(config).to(device = device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total params:{total_params}')

    data_path = 'D:/think-dataset/think-llm-pretrain/Mini'
    tokenizer_path = './model/think_tokenizer'
    train_loader = PretrainDataset(config, data_path, tokenizer_path, lru_size = 64, reuse_count = 64)

    loss_func = nn.CrossEntropyLoss(reduction = 'none')
    scaler = torch.amp.GradScaler(enabled = (device.type == 'cuda'))
    opt = torch.optim.AdamW(model.parameters(), lr = base_lr, betas = (0.9, 0.95), weight_decay = weight_decay)

    update_step = 20000 # update iter
    total_step = update_step * accumulation_steps
    warmup_step = int(warmup_ratio * total_step)
    if not os.path.exists('pretrain_train_log.txt'):
        with open('pretrain_train_log.txt', 'a', encoding='utf-8') as f:
            f.write(f'Step, Loss, Lr, Time\n')
    try:
        checkpoint = torch.load('pretrain_checkpoint.bin')
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

        if (step + 1) % log_print_save_step == 0:
            print(f'Step:{step + 1}/{total_step}, Loss:{loss.item() * accumulation_steps:.4f}, '
                  f'Lr:{opt.param_groups[0]["lr"]:.6f}, Time:{time.time() - start_time:.2f}')
            with open('pretrain_train_log.txt', 'a', encoding ='utf-8') as f:
                f.write(f'{step + 1}, {loss.item() * accumulation_steps:.4f}, {opt.param_groups[0]["lr"]:.6f}, '
                        f'{time.time() - start_time:.2f}\n')

        if (step + 1) % log_print_save_step == 0:
            from transformers import AutoTokenizer
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained('./model/think_tokenizer')

            # 尿路感染的病因是什么??
            # 非复杂性尿路感染80%由大肠杆菌引起,10~15%由葡萄球菌和克雷白氏杆菌引起,仅2~5%是由变性杆菌所致。
            # 而复杂性尿路感染的细菌谱则要广的多,大肠杆菌仍为主要致病菌,
            # 但许多其它的革兰氏阴性细菌如变性杆菌、沙雷菌属、克雷白菌及假单孢菌属等,均可导致复杂性尿路感染。
            # 在糖尿病患者或免疫力低下的患者中,霉菌的感染日益增多。
            text = f'<im_start>尿路感染的病因是什么?'

            encoder = tokenizer.batch_encode_plus([text])
            input_ids = torch.tensor(encoder['input_ids'], dtype = torch.long, device = device)
            attention_mask = torch.tensor(encoder['attention_mask'], dtype = torch.long, device = device)
            # print(input_ids)
            # print(attention_mask)
            kv_cache = {}
            start_pos = 0
            out = model.generate(input_ids, start_pos, attention_mask, kv_cache, temperature = 0.7, top_p = 0.95,
                                 rp = 1.05, eos_token_id = 2)
            pred = tokenizer.decode(out[0, :])
            print(text + '\n' + ''.join(pred))

        if (step + 1) % model_save_step == 0:
            checkpoint = {
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'scaler': scaler.state_dict(),
                'current_step': current_step
            }
            torch.save(checkpoint, 'pretrain_checkpoint.bin.tmp')
            os.replace('pretrain_checkpoint.bin.tmp', 'pretrain_checkpoint.bin')

    checkpoint = {
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'current_step': current_step
    }
    torch.save(checkpoint, 'pretrain_checkpoint.bin.tmp')
    os.replace('pretrain_checkpoint.bin.tmp', 'pretrain_checkpoint.bin')



