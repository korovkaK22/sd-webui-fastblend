# FastBlend - Інструкція з використання

## Вимоги

- Windows 10/11
- Python 3.11 (рекомендовано)
- NVIDIA GPU з підтримкою CUDA 12.x
- ~8 GB RAM мінімум
- ~4 GB VRAM мінімум

## Встановлення

### 1. Клонування репозиторію

```bash
git clone https://github.com/YOUR_USERNAME/sd-webui-fastblend.git
cd sd-webui-fastblend
```

### 2. Створення віртуального середовища

```bash
# Windows
py -3.11 -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Встановлення залежностей

```bash
pip install -r requirements.txt
```

## Використання

### Швидка обробка (одне відео, ~15-20 хв)

```bash
python process_fast.py
```

Параметри в `process_fast.py`:
- `VIDEO_PATH` - шлях до вхідного відео
- `OUTPUT_DIR` - папка для результату

### Якісна обробка з checkpoints (~45 хв)

```bash
python process_quality.py
```

Параметри в `process_quality.py`:
- `SOURCE_DIR` - папка з вхідними відео
- `OUTPUT_DIR` - папка для результатів
- Підтримує відновлення після збою

### Відновлення після збою

Якщо обробка впала, просто запустіть скрипт знову - він продовжить з останнього checkpoint:

```bash
python process_quality.py
```

Checkpoints зберігаються в папці `checkpoints/`.

## Параметри обробки

### Режими (mode)

| Режим | Час | Пам'ять | Якість |
|-------|-----|---------|--------|
| Fast | Швидко | Багато RAM | Добра |
| Balanced | Середньо | Мало RAM | Добра |
| Accurate | Довго | Мало RAM | Найкраща |

### Основні параметри

- `window_size` - розмір вікна згладжування (5-30). Більше = плавніше, але може бути розмито
- `batch_size` - розмір батчу (1-16). Більше = швидше, але більше VRAM
- `num_iter` - кількість ітерацій (3-10). Більше = якісніше, але довше
- `minimum_patch_size` - мінімальний розмір патчу (5-15). Для високої роздільності використовуйте більше значення

### Рекомендації по роздільності

| Роздільність | batch_size | minimum_patch_size |
|--------------|------------|-------------------|
| 512x512 | 8-16 | 5 |
| 720p | 4-8 | 7 |
| 1080p | 2-4 | 9 |
| 1280x720+ | 2 | 9-11 |

## Структура проекту

```
sd-webui-fastblend/
├── FastBlend/           # Основний код алгоритму
├── checkpoints/         # Checkpoints для відновлення (створюється автоматично)
├── process_fast.py      # Швидка обробка
├── process_quality.py   # Якісна обробка з checkpoints
├── requirements.txt     # Залежності
└── USAGE.md            # Ця інструкція
```

## Вирішення проблем

### "CuPy failed to load nvrtc64_120_0.dll"
PyTorch не встановлений або встановлений без CUDA. Перевстановіть:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### "Unable to allocate X MiB for an array"
Недостатньо RAM. Зменшіть `batch_size` до 2 або 1.

### GPU не використовується
Перевірте, що CUDA доступна:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Відео виходить розмитим
Зменшіть `window_size` (наприклад, з 15 до 7-10).
