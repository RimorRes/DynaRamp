import logging

# Configure parent logger for the package
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
