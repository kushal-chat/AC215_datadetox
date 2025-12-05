from .agent import hf_search_agent, neo4j_search_agent, compiler_agent
import logging
from termcolor_dg import logging_basic_color_config

logging_basic_color_config()
logging.log(logging.INFO, "🙅🙅🙅 logger successful 🙅🙅🙅")

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

logging_basic_color_config()
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)

__all__ = ["hf_search_agent", "neo4j_search_agent", "compiler_agent"]
