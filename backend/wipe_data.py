"""Utility script to wipe all persistent data for CognitiveAI.

Removes SQLite DB files, cache files, clears in-memory stores, and clears Pinecone LTM index
if configured. Run this from the project root or the backend folder with the virtualenv active.
"""
import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DB_PATHS = [ROOT.parent / "cognitiveai.db", ROOT / "cognitiveai.db"]
CANDIDATE_CACHE = [ROOT / "cognitiveai_cache.db", ROOT.parent / "cognitiveai_cache.db", Path("cognitiveai_cache.db")]


def remove_file(p: Path):
    try:
        if p.exists():
            p.unlink()
            logger.info(f"Removed file: {p}")
        else:
            logger.info(f"File not found (skipping): {p}")
    except Exception as e:
        logger.warning(f"Failed to remove {p}: {e}")


def main():
    logger.info("Starting wipe of persistent data...")

    
    for p in DB_PATHS:
        remove_file(p)

    
    for p in CANDIDATE_CACHE:
        remove_file(p)

    
    try:
        
        project_root = ROOT.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        import backend.main as mainmod

        
        try:
            if hasattr(mainmod, "pdf_text_store"):
                mainmod.pdf_text_store.clear()
                logger.info("Cleared in-memory pdf_text_store")
        except Exception as e:
            logger.warning(f"Failed to clear pdf_text_store: {e}")

        
        try:
            if hasattr(mainmod, "stm_manager") and mainmod.stm_manager:
                
                for uid in list(getattr(mainmod.stm_manager, "user_memories", {}).keys()):
                    try:
                        mainmod.stm_manager.clear_memories(uid)
                    except Exception:
                        pass
                logger.info("Cleared in-memory STM memories")
        except Exception as e:
            logger.warning(f"Failed to clear STM memories: {e}")

        
        try:
            if hasattr(mainmod, "ltm_manager") and mainmod.ltm_manager:
                mainmod.ltm_manager.clear_all_memories()
                logger.info("Cleared LTM index via LTMManager.clear_all_memories()")
        except Exception as e:
            logger.warning(f"Failed to clear LTM via LTMManager: {e}")

    except Exception as e:
        logger.warning(f"Could not import backend.main; skipping in-memory and LTM clear: {e}")

    logger.info("Wipe complete.")


if __name__ == "__main__":
    main()
