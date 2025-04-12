import os
import sys
import logging

logging_str = "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s"
logging_dir = os.path.join(os.path.dirname("/".join(os.path.abspath(__file__).split('/')[:-2])), "logs")
loging_path = os.path.join(logging_dir, "youtubesummarizerlogger.log")
if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
if not os.path.exists(loging_path):
    with open(loging_path, "w") as f:
        pass
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(loging_path),
        logging.StreamHandler(sys.stdout)
    ]
)

logging = logging.getLogger('youtubesummarizer')