import logging
import sys
from pathlib import Path

# Добавляем корень проекта в путь для импорта, чтобы скрипт мог найти модули внутри scripts.data_prep
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.utils.import_msd_to_processed import import_msd
from scripts.utils.prepare_upenn_4d import prepare_upenn_dataset
from scripts.utils.collect_metadata import collect_metadata
from scripts.utils.validate_dataset import (
    validate_integrity,
    validate_msd_consistency,
    validate_upenn_consistency,
    validate_duplicates,
    validate_processed_dataset
)

logger = logging.getLogger(__name__)

def main():
    # Настройка базового логирования
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("=== Starting Data Preparation Master Script ===")
    
    try:
        # 1. Импорт MSD (копирование и исправление dataset.json)
        logger.info("\n--- STEP 1: Importing MSD dataset ---")
        import_msd()
        
        # 2. Подготовка UPENN (объединение модальностей в 4D и переориентация в RAS)
        logger.info("\n--- STEP 2: Preparing UPENN dataset (stacking and reorienting) ---")
        prepare_upenn_dataset()
        
        # 3. Сбор метаданных (создание metadata.csv с хешами и статистикой меток)
        logger.info("\n--- STEP 3: Collecting metadata, hashes and label stats ---")
        collect_metadata()
        
        # 4. Валидация (запуск всех проверок качества)
        logger.info("\n--- STEP 4: Running all validation checks ---")
        success = True
        
        # Проверка целостности NIfTI файлов (читаемость и отсутствие NaNs)
        if not validate_integrity(['data/raw/MSD_BrainTumour', 'data/raw/UPENN-GBM']):
            success = False
            
        # Проверка соответствия папок в сырых данных MSD
        if not validate_msd_consistency('data/raw/MSD_BrainTumour'):
            success = False
            
        # Проверка полноты модальностей в сырых данных UPENN
        if not validate_upenn_consistency('data/raw/UPENN-GBM'):
            success = False
            
        # Проверка на наличие дубликатов пациентов между датасетами
        if not validate_duplicates('data/processed/metadata.csv'):
            success = False
            
        # Проверка корректности обработанных данных (4D, RAS, Label values)
        for ds in ['UPENN-GBM', 'MSD_BrainTumour']:
            if not validate_processed_dataset(ds):
                success = False
                
        if not success:
            logger.error("\n[!] Data preparation finished with VALIDATION ERRORS.")
            sys.exit(1)
        else:
            logger.info("\n[v] Data preparation finished SUCCESSFULLY.")
            
    except Exception as e:
        logger.error(f"\n[!] Critical error during data preparation: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
