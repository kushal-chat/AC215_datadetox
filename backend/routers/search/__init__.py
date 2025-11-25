from .agent import search_agent
import logging
from termcolor_dg import logging_basic_color_config

logging_basic_color_config()
logging.log(logging.INFO, '🙅🙅🙅 logger successful 🙅🙅🙅')

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

logging_basic_color_config()
logging.basicConfig(
    level=logging.INFO,          
    format=LOG_FORMAT,
)
logging.getLogger("httpcore.http11").setLevel(logging.DEBUG)
logging.getLogger("openai.agents").setLevel(logging.INFO)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("neo4j.io").setLevel(logging.DEBUG)

__all__ = ["search_agent"]