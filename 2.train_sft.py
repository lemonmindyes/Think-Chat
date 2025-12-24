import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn

from config import Config
from dataset import SFTDataset
from gpt import ThinkChat


if __name__ == '__main__':
    # param
    dtype = torch.bfloat16
    batch_size = 32
    base_lr = 1e-5
    weight_decay = 0.1
    warmup_ratio = 0.005
    grad_clip = 1.0
    accumulation_steps = 1
    model_save_step = 200
    log_print_save_step = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = nullcontext() if device.type == 'cpu' else torch.amp.autocast(device_type = device.type, dtype = dtype)

    config = Config()
    config.use_yarn = True
    config.max_seq_len = 512
    model = ThinkChat(config).to(device = device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total params:{total_params}')

    data_path = 'D:/think-dataset/think-llm-instruct-follow/Mini'
    tokenizer_path = './model/think_tokenizer'
    train_loader = SFTDataset(config, data_path, tokenizer_path, lru_size = 128, reuse_count = 1)

    loss_func = nn.CrossEntropyLoss(reduction = 'none')
    scaler = torch.amp.GradScaler(enabled = (device.type == 'cuda'))
    opt = torch.optim.AdamW(model.parameters(), lr = base_lr, betas = (0.9, 0.95), weight_decay = weight_decay)

    update_step = 200000 # update iter
    total_step = update_step * accumulation_steps
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
        model.load_state_dict(torch.load('model/Think-Chat-Mini/Mini/pretrain_checkpoint.bin')['model'])
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
                  f'Lr:{opt.param_groups[0]["lr"]:.7f}, Time:{time.time() - start_time:.2f}')
            with open('sft_train_log.txt', 'a', encoding ='utf-8') as f:
                f.write(f'{step + 1}, {loss.item() * accumulation_steps:.4f}, {opt.param_groups[0]["lr"]:.7f}, '
                        f'{time.time() - start_time:.2f}\n')

        if (step + 1) % log_print_save_step == 0:
            from transformers import AutoTokenizer

            model.eval()
            tokenizer = AutoTokenizer.from_pretrained('./model/think_tokenizer')

            # {
            #     "role":
            #         "user", "content": "法国的首都是什么？"
            # }
            # {
            #     "role": "assistant",
            #     "content": "法国的首都是巴黎。"
            # }
            user_content = '法国的首都是什么？'
            text = f'<im_start>user\n{user_content}\n<im_end>\n<im_start>assistant\n'

            encoder = tokenizer.batch_encode_plus([text])
            input_ids = torch.tensor(encoder['input_ids'], dtype=torch.long, device=device)
            attention_mask = torch.tensor(encoder['attention_mask'], dtype=torch.long, device=device)
            kv_cache = {}
            start_pos = 0
            out = model.generate(input_ids, start_pos, attention_mask, kv_cache, temperature=0.6, top_p=0.95,
                                 rp=1.05, eos_token_id=2)
            pred = tokenizer.decode(out[0, :])
            print(text + ''.join(pred))

            # {
            #     "role": "user",
            #     "content": "列出三种可能由于不良卫生习惯引起的疾病。"
            # }
            # {
            #     "role": "assistant",
            #     "content": "1. 感染性腹泻疾病：不洗手或食用未经清洗的食物，可能会导致细菌或病毒感染，引发腹泻等肠道疾病。\n\n"
            #                "2. 皮肤感染：不勤洗澡或不洗手可能导致皮肤感染，引发脓包、疥疮、毛囊炎等疾病。\n\n"
            #                "3. 结膜炎：不勤洗手并经常摩擦眼睛，可能会导致细菌感染眼睛，引发结膜炎，出现眼睛红肿、分泌物增多、瘙痒等症状。"
            # }
            user_content = '列出三种可能由于不良卫生习惯引起的疾病。'
            text = f'<im_start>user\n{user_content}\n<im_end>\n<im_start>assistant\n'

            encoder = tokenizer.batch_encode_plus([text])
            input_ids = torch.tensor(encoder['input_ids'], dtype=torch.long, device=device)
            attention_mask = torch.tensor(encoder['attention_mask'], dtype=torch.long, device=device)
            kv_cache = {}
            start_pos = 0
            out = model.generate(input_ids, start_pos, attention_mask, kv_cache, temperature=0.7, top_p=0.95,
                                 rp=1.05, eos_token_id=2)
            pred = tokenizer.decode(out[0, :])
            print(text + ''.join(pred))

        if (step + 1) % model_save_step == 0:
            checkpoint = {
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'scaler': scaler.state_dict(),
                'current_step': current_step
            }
            torch.save(checkpoint, 'sft_checkpoint.bin.tmp')
            os.replace('sft_checkpoint.bin.tmp', 'sft_checkpoint.bin')

    checkpoint = {
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'current_step': current_step
    }
    torch.save(checkpoint, 'sft_checkpoint.bin.tmp')
    os.replace('sft_checkpoint.bin.tmp', 'sft_checkpoint.bin')



