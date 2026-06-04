import logging
from clearml import Task, Model

logger = logging.getLogger(__name__)

def select_best_model(project_name='AdultGliomaSegmentation'):
    # Получаем все завершенные задачи в проекте
    tasks = Task.get_tasks(
        project_name=project_name,
        task_filter={'status': ['completed']}
    )
    
    best_task = None
    max_dice = -1
    
    for task in tasks:
        # Получаем последнюю метрику Dice
        metrics = task.get_last_scalar_metrics()
        if 'Dice' in metrics and 'val' in metrics['Dice']:
            dice = metrics['Dice']['val']['last']
            if dice > max_dice:
                max_dice = dice
                best_task = task
                
    if best_task:
        logger.info(f"Лучшая модель найдена в задаче: {best_task.name} (ID: {best_task.id})")
        logger.info(f"Dice: {max_dice}")
        
        # Получаем артефакт модели
        models = best_task.get_models()
        if models:
            best_model_artifact = models[-1] # Берем последнюю сохраненную
            logger.info(f"Путь к весам: {best_model_artifact.url}")
            
            # Добавляем тег 'production'
            best_model_artifact.add_tags(['production'])
            logger.info("Модель помечена тегом 'production' для деплоя.")
            return best_model_artifact
    else:
        logger.warning("Завершенные задачи с метрикой Dice не найдены.")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    select_best_model()
