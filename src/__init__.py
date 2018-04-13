import logging
import os

def setup_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger

logger = setup_logger("zero")

repository_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) # fuck yeah python
results_dir = os.environ.get('RESULTS_DIR', os.path.join(repository_dir, 'results'))