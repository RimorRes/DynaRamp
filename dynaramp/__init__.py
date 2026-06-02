import logging

# Configure parent logger for the package
logger = logging.getLogger(__name__)  # This will be 'dynaramp'
logger.setLevel(logging.DEBUG)

# Optional: Add a handler if no handler is configured at the parent/root level
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Prevent propagation if you don't want messages going up the hierarchy
# logger.propagate = False
