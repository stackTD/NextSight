"""Main entry point for NextSight application."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from config.settings import LOG_LEVEL, LOG_FORMAT, LOG_FILE


def setup_logging():
    """Configure logging for the application."""
    logger.remove()  # Remove default logger
    logger.add(sys.stderr, level=LOG_LEVEL, format=LOG_FORMAT)
    logger.add(LOG_FILE, level=LOG_LEVEL, format=LOG_FORMAT, rotation="1 day")


def main():
    """Main application entry point."""
    setup_logging()
    logger.info("Starting NextSight application...")
    
    try:
        # TODO: Initialize camera manager
        # TODO: Initialize hand detector
        # TODO: Initialize object detector
        # TODO: Initialize display manager
        # TODO: Start main processing loop
        
        logger.info("NextSight application initialized successfully")
        logger.info("Press 'q' to quit")
        
        # Placeholder for main loop
        print("NextSight is ready! (Implementation coming soon)")
        print("Press Enter to exit...")
        input()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise
    finally:
        logger.info("NextSight application shutting down...")


if __name__ == "__main__":
    main()