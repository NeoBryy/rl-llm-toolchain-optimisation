print("IMPORTS START")
from src.rl.model_wrapper import ReActLlamaModel
from src.utils.logger import get_logger
import logging
print("IMPORTS DONE")

# Configure logging to see output
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

print("Initializing model...")
logger.info("Initializing model...")
model = ReActLlamaModel("Qwen/Qwen2.5-1.5B-Instruct")
print("Running episode...")
logger.info("Running episode...")
result = model.run_episode("What is the average consumption for residential customers?")
print(result)
