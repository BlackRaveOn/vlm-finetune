import logging


def set_logger(name: str):
    """
    Создаёт и настраивает логгер с консольным выводом.

    Функция инициализирует объект логгера с указанным именем, устанавливает уровень логирования
    `INFO`, добавляет консольный обработчик (`StreamHandler`) и форматирует сообщения в виде:

        YYYY-MM-DD HH:MM:SS [УРОВЕНЬ] имя_логгера: сообщение

    Пример форматированного вывода:
        2025-10-25 21:30:12 [INFO] llavamodel: Модель успешно загружена.

    Параметры:
        name: str
            Имя логгера (обычно соответствует имени модуля, например `__name__`).

    Возвращает:
        logging.Logger:
            Настроенный экземпляр логгера, готовый к использованию.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
