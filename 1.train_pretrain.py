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
    batch_size = 8
    base_lr = 5e-4
    warmup_ratio = 0.05
    grad_clip = 1.0
    accumulation_steps = 8
    model_save_step = 100
    log_print_save_step = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = nullcontext() if device.type == 'cpu' else torch.amp.autocast(device_type = device.type, dtype = dtype)

    config = Config()
    model = ThinkChat(config).to(device = device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total params:{total_params}')

    data_path = 'D:/think-dataset/think-llm-pretrain' # 50835675 lines
    tokenizer_path = './model/think_tokenizer'
    train_loader = PretrainDataset(config, data_path, tokenizer_path, lru_size = 32, reuse_count = 128)

    loss_func = nn.CrossEntropyLoss(reduction = 'none')
    scaler = torch.amp.GradScaler(enabled = (device.type == 'cuda'))
    opt = torch.optim.AdamW(model.parameters(), lr = base_lr, betas = (0.9, 0.95), weight_decay = 0.0)

    update_step = 100000 # update iter
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

            # 8月29日凌晨,浙江大学一名刚报到的大一女生从寝室上铺摔下,至发稿时已被送往重症监护室,尚未脱离生命危险。
            # 据学校老师说,女孩姓徐,江苏人,是从上下铺的上层摔下来,床沿有栏杆,但具体经过他不清楚。据介绍,4点40多分女孩被送到市二医院,当时处于昏迷状态。
            # 上午10点10分左右,杭州市二医院手术室的门打开,医生向学校和家属介绍,女孩颅骨骨折,有脑水肿、脑挫伤。
            # 10点30分,女孩手术完毕,被推了出来,医生称手术顺利。
            # 女孩母亲告诉记者,她们夫妻26日陪女儿到校报到,27日回家,\"今天早上4点多接到学校的电话,就连忙赶来。\"经ct检查,颅内有出血和水肿,
            # 医院进行了开颅手术,清除血肿,术后已送往重症监护室,还未脱离生命危险,后续要进一步观察。据了解,这名女生徐,今年18岁,是江苏人,
            # 这个月26号才来校报到,还没有参加新生军训。今天凌晨,她从寝室床的上铺意外摔了下来,随后同寝室的室友赶紧拨打了120,老师也赶紧赶来,一起赶到医院。
            # 参与抢救的医生表示,女孩送到医院的时候处于昏迷状态,她头部的伤口在右侧耳廓等处,有明显的外伤。手术进行了三个小时,主要是脑挫伤,左侧的硬膜下血肿,右侧也有骨折。
            # 医生表示,仅就手术来说,抢救是顺利的,但目前还没有脱离生命危险。上午10点40分,女生的父母赶到了医院,一直眼泪汪汪,至始至终一言不发。
            text = f'<im_start>8月29日凌晨,浙江大学一名刚报到的大一女生从寝室上铺摔下,至发稿时已被送往重症监护室,尚未脱离生命危险。据学校老师说,'

            encoder = tokenizer.batch_encode_plus([text])
            input_ids = torch.tensor(encoder['input_ids'], dtype = torch.long, device = device)
            attention_mask = torch.tensor(encoder['attention_mask'], dtype = torch.long, device = device)
            # print(input_ids)
            # print(attention_mask)
            kv_cache = {}
            start_pos = 0
            out = model.generate(input_ids, start_pos, attention_mask, kv_cache, temperature = 0.85, top_p = 0.85,
                                 rp = 1.05, eos_token_id = 2)
            pred = tokenizer.decode(out[0, :])
            print(text + ''.join(pred))

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



