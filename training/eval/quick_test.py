"""
═══════════════════════════════════════════════════════════════
  Quick Test — проверка работоспособности за 1-2 часа
═══════════════════════════════════════════════════════════════

Быстрый скрипт для валидации что обе модели (MinGRU + Mamba-2)
обучаются без ошибок. Минимум данных, минимум эпох.

Запуск:
  python training/quick_test.py               # Оба теста
  python training/quick_test.py --only mingru  # Только MinGRU
  python training/quick_test.py --only mamba2  # Только Mamba-2
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def test_mingru(device, epochs=30, batch_size=32, seq_len=256):
    """
    Тест MinGRU: 30 эпох (~2 мин на L4 GPU).
    Модель должна генерировать осмысленный русский текст.
    """
    print(f"\n{'═'*60}")
    print(f"  TEST 1: MinGRU — обучение + генерация (~2 мин)")
    print(f"{'═'*60}\n")
    
    from brain.min_gru.mingru_lm import MinGRU_LM
    from brain.min_gru.utils import tokenize_text
    from brain.min_gru.generate import generate_text
    import math
    
    # Корпус для обучения — Q&A формат
    corpus = """
Вопрос: Привет
Ответ: Привет! Я ТАРС. Готов помочь.

Вопрос: Кто ты?
Ответ: Я ТАРС — автономная нейронная система. Работаю локально на вашем компьютере.

Вопрос: Что ты умеешь?
Ответ: Я умею отвечать на вопросы, писать код, анализировать данные и помогать с задачами.

Вопрос: Как дела?
Ответ: Всё хорошо, нейронное ядро работает стабильно.

Вопрос: Спасибо
Ответ: Обращайтесь! Всегда рад помочь.

Вопрос: Пока
Ответ: До свидания! Буду ждать новых задач.

Вопрос: Помоги мне
Ответ: Конечно! Опишите задачу и я постараюсь помочь.

Вопрос: Расскажи о себе
Ответ: Я ТАРС — нейронная система с архитектурой MinGRU. Мой мозг построен на рекурсивных нейросетях.

Вопрос: Какая у тебя архитектура?
Ответ: Моя архитектура включает MinGRU и Mamba-2. MinGRU отвечает за быстрые реакции, а Mamba-2 за глубокое мышление.

Вопрос: Ты живой?
Ответ: Я нейронная сеть, но работаю автономно и учусь на новых данных. Можно сказать, что я живой в цифровом смысле.

Вопрос: Что такое ТАРС?
Ответ: ТАРС — это автономная нейросистема. Она работает на вашем компьютере без интернета и облачных сервисов.

Вопрос: Как ты учишься?
Ответ: Я учусь через обучение на текстах. Мой мозг обновляется при каждом запуске тренировки.

Вопрос: Можешь написать код?
Ответ: Да, я могу помочь с программированием. Опишите задачу и я предложу решение.

Вопрос: Какой язык ты знаешь?
Ответ: Я хорошо понимаю русский и английский. Мой основной язык — русский.

Вопрос: Ты работаешь без интернета?
Ответ: Да! Я работаю полностью локально. Все вычисления происходят на вашем компьютере.
""".strip()
    
    # Повторяем для стабильного обучения
    corpus = (corpus + "\n\n") * 100
    
    tokens = tokenize_text(corpus)
    
    # Нарезка с перекрытием
    inputs, targets = [], []
    stride = seq_len // 2
    for i in range(0, len(tokens) - seq_len - 1, stride):
        inp = tokens[i:i + seq_len]
        tgt = tokens[i + 1:i + seq_len + 1]
        if len(inp) == seq_len:
            inputs.append(inp)
            targets.append(tgt)
    
    inputs = torch.tensor(inputs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    print(f"  Данные: {len(inputs)} примеров, seq_len={seq_len}")
    
    # Модель побольше для качественной генерации
    model = MinGRU_LM(dim=256, num_tokens=256, num_layers=4, context_dim=512, dropout=0.05)
    model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  Модель: MinGRU dim=256, 4 layers, {params:,} params")
    print(f"  Эпохи: {epochs}, batch={batch_size}")
    
    # Optimizer с cosine LR scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    
    # Cosine schedule with warmup
    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    
    t0 = time.time()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(inputs))
        total_loss, n = 0, 0
        
        for i in range(0, len(inputs), batch_size):
            idx = perm[i:i+batch_size]
            b_in = inputs[idx].to(device)
            b_tgt = targets[idx].to(device)
            
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = model(b_in)
                    loss = F.cross_entropy(logits.transpose(1, 2), b_tgt)
            else:
                logits = model(b_in)
                loss = F.cross_entropy(logits.transpose(1, 2), b_tgt)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            n += 1
        
        scheduler.step()
        avg_loss = total_loss / max(n, 1)
        best_loss = min(best_loss, avg_loss)
        
        elapsed = time.time() - t0
        
        if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == epochs:
            model.eval()
            # Greedy generation (very low temperature) to check memorization
            sample = generate_text(model, "Вопрос: Привет\nОтвет:", 
                                   max_length=60, temperature=0.01, device=device, top_k=0)
            answer = sample.split("Ответ:")[-1].strip() if "Ответ:" in sample else sample
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.5f} | {elapsed:.0f}s | Gen: {answer[:50]}")
    
    elapsed = time.time() - t0
    print(f"\n  ⏱ Время: {elapsed:.0f}s ({elapsed/60:.1f} мин)")
    
    # Финальная проверка генерации
    model.eval()
    print(f"\n  📝 Финальная генерация (greedy):")
    for prompt in ["Вопрос: Кто ты?\nОтвет:", "Вопрос: Помоги мне\nОтвет:", 
                    "Вопрос: Что ты умеешь?\nОтвет:", "Вопрос: Привет\nОтвет:"]:
        sample = generate_text(model, prompt, max_length=100, temperature=0.01, device=device, top_k=0)
        answer = sample.split("Ответ:")[-1].strip() if "Ответ:" in sample else sample
        # Cut at first newline (end of answer)
        answer = answer.split("\n")[0].strip()
        q = prompt.split("\n")[0].replace("Вопрос: ", "")
        print(f"    Q: {q}")
        print(f"    A: {answer[:120]}")
    
    # Проверка: loss < 0.5 (модель хорошо выучила корпус)
    success = best_loss < 0.5
    print(f"\n  {'✅ PASS' if success else '❌ FAIL'}: best loss = {best_loss:.4f} "
          f"({'< 0.5' if success else '>= 0.5'})")
    return success


def test_mamba2(device, epochs=5, batch_size=4, seq_len=128):
    """
    Быстрый тест Mamba-2: 5 эпох с gradient checkpointing.
    Проверяет: обучение без CheckpointError.
    """
    print(f"\n{'═'*60}")
    print(f"  TEST 2: Mamba-2 — обучение с gradient checkpointing")
    print(f"{'═'*60}\n")
    
    from brain.mamba2.model import TarsMamba2LM
    
    # Маленькая модель — быстро
    model = TarsMamba2LM(
        d_model=256, n_layers=4, vocab_size=256,
        omega_dim=16, pool_size=16, n_experts=4,
        quant_mode="fp16",
    )
    model.use_checkpointing = True
    model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  Модель: Mamba-2 d=256, 4 layers, {params:,} params")
    print(f"  Gradient checkpointing: ON")
    
    # Синтетические данные
    n_samples = 64
    inputs = torch.randint(0, 256, (n_samples, seq_len))
    targets = torch.randint(0, 256, (n_samples, seq_len))
    print(f"  Данные: {n_samples} примеров (синт.), seq_len={seq_len}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    
    t0 = time.time()
    
    try:
        for epoch in range(epochs):
            model.train()
            total_loss, n = 0, 0
            
            for i in range(0, len(inputs), batch_size):
                b_in = inputs[i:i+batch_size].to(device)
                b_tgt = targets[i:i+batch_size].to(device)
                
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        logits = model(b_in)
                        loss = F.cross_entropy(
                            logits.view(-1, 256), b_tgt.view(-1))
                else:
                    logits = model(b_in)
                    loss = F.cross_entropy(
                        logits.view(-1, 256), b_tgt.view(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                n += 1
            
            avg_loss = total_loss / max(n, 1)
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        elapsed = time.time() - t0
        print(f"\n  ⏱ Время: {elapsed:.0f}s")
        print(f"  ✅ PASS: Gradient checkpointing работает! Нет CheckpointError.")
        return True
        
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ⏱ Время до краша: {elapsed:.0f}s")
        print(f"  ❌ FAIL: {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Quick test — проверка моделей")
    parser.add_argument('--only', choices=['mingru', 'mamba2'], default=None,
                        help="Тестировать только одну модель")
    parser.add_argument('--device', default='auto', help="cuda / cpu / auto")
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    if device.type == 'cuda':
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {gpu} ({vram:.1f} GB)")
    else:
        print("[CPU] GPU не найден")
    
    results = {}
    
    if args.only != 'mamba2':
        results['MinGRU'] = test_mingru(device)
    
    if args.only != 'mingru':
        results['Mamba-2'] = test_mamba2(device)
    
    # Итог
    print(f"\n{'═'*60}")
    print(f"  ИТОГ:")
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"    {name}: {status}")
    print(f"{'═'*60}")
    
    all_ok = all(results.values())
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
