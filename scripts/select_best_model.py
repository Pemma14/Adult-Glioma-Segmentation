import argparse
import pandas as pd
import logging
import re
from clearml import Task

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_best_metrics(task):
    """Извлекает лучший Dice и номер эпохи из задачи ClearML."""
    scalars = task.get_reported_scalars()
    best_dice = None
    best_epoch = None
    
    # 1. Пробуем найти в single values (то, что train.py пишет через report_single_value)
    for title in scalars:
        if 'best_val_dice' in scalars[title]:
            best_dice = scalars[title]['best_val_dice']['y'][-1]
        if 'best_epoch' in scalars[title]:
            best_epoch = int(scalars[title]['best_epoch']['y'][-1])
            
    # 2. Fallback на обычные метрики (Val Dice / mean_dice)
    if best_dice is None:
        metrics = task.get_last_scalar_metrics()
        if 'Val Dice' in metrics and 'mean_dice' in metrics['Val Dice']:
            best_dice = metrics['Val Dice']['mean_dice']['last']
        elif 'Dice' in metrics and 'val' in metrics['Dice']:
            best_dice = metrics['Dice']['val']['last']
            
    return best_dice, best_epoch

def parse_task_info(task):
    """Извлекает метаданные задачи из тегов и названия."""
    tags = task.get_tags()
    info = {
        'stage': next((t.split(':')[1] for t in tags if t.startswith('stage:')), None),
        'model': next((t.split(':')[1] for t in tags if t.startswith('model:')), None),
        'fold': next((t.split(':')[1] for t in tags if t.startswith('fold:')), None),
    }
    
    # Извлекаем run_id из названия (паттерн _rN_)
    run_match = re.search(r'_r(\d+)', task.name)
    info['run'] = int(run_match.group(1)) if run_match else 0
    
    # Если тегов нет, пробуем распарсить название: {stage}_{model}_r{run}_f{fold}
    if not info['stage'] or not info['model']:
        parts = task.name.split('_')
        if len(parts) >= 2:
            info['stage'] = info['stage'] or parts[0]
            info['model'] = info['model'] or parts[1]
            
    return info

def run_analysis(project, stage=None, model=None, run_id=None, tag_production=False):
    tasks = Task.get_tasks(
        project_name=project,
        task_filter={'status': ['completed', 'published', 'closed']}
    )
    
    results = []
    for task in tasks:
        info = parse_task_info(task)
        
        # Применяем фильтры, если они заданы
        if stage and info['stage'] != stage: continue
        if model and info['model'] != model: continue
        if run_id is not None and info['run'] != run_id: continue
        
        dice, epoch = get_best_metrics(task)
        if dice is None: continue
        
        results.append({
            'task_id': task.id,
            'task_name': task.name,
            'stage': info['stage'],
            'model': info['model'],
            'run': info['run'],
            'fold': info['fold'],
            'dice': dice,
            'epoch': epoch,
            'task_obj': task
        })
        
    if not results:
        logger.warning("Задачи, соответствующие фильтрам, не найдены.")
        return

    df = pd.DataFrame(results)
    
    # 1. Групповой отчет по экспериментам (Mean ± Std)
    summary = df.groupby(['stage', 'model', 'run']).agg(
        mean_dice=('dice', 'mean'),
        std_dice=('dice', 'std'),
        folds=('fold', 'count'),
        max_dice=('dice', 'max')
    ).reset_index().sort_values(['stage', 'mean_dice'], ascending=[True, False])
    
    print("\n" + "="*90)
    print("ОБЩАЯ СВОДКА ПО ЭКСПЕРИМЕНТАМ")
    print("="*90)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    print("="*90)

    # 2. Детальный отчет по фолдам (если выбран конкретный запуск или их мало)
    if run_id is not None or len(summary) == 1:
        print("\nДЕТАЛИЗАЦИЯ ПО ФОЛДАМ:")
        cols = ['fold', 'dice', 'epoch', 'task_id']
        print(df.sort_values('fold')[cols].to_string(index=False))

    # 3. Поиск и тегирование лучшей модели
    best_idx = df['dice'].idxmax()
    best_info = df.loc[best_idx]
    
    print(f"\nАбсолютно лучший результат: {best_info['task_name']} (Dice: {best_info['dice']:.4f})")

    if tag_production:
        best_task = best_info['task_obj']
        models = best_task.get_models()
        if models:
            best_model = models[-1]
            best_model.add_tags(['production'])
            logger.info(f"Модель из задачи {best_task.id} помечена тегом 'production'")
        else:
            logger.warning("У лучшей задачи не найдены артефакты моделей.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Анализ результатов экспериментов")
    parser.add_argument("--project", default="AdultGliomaSegmentation", help="Имя проекта")
    parser.add_argument("--stage", help="Фильтр по стадии (например, hpo или final)")
    parser.add_argument("--model", help="Фильтр по имени модели")
    parser.add_argument("--run_id", type=int, help="Фильтр по ID запуска")
    parser.add_argument("--tag_best", action="store_true", help="Пометить лучшую модель тегом production")
    
    args = parser.parse_args()
    run_analysis(args.project, args.stage, args.model, args.run_id, args.tag_best)
