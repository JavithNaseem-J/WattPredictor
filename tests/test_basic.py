"""Basic sanity tests for project structure."""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestProjectStructure:
    """Test that project structure is correct."""
    
    def test_src_directory_exists(self):
        """Test that src directory exists."""
        src_dir = Path(__file__).parent.parent / "src"
        assert src_dir.exists()
        assert src_dir.is_dir()
    
    def test_config_module_exists(self):
        """Test that config module can be imported."""
        try:
            from WattPredictor.config.config import get_config
            assert get_config is not None
        except ImportError as e:
            pytest.fail(f"Failed to import config: {e}")
    
    def test_wattpredictor_package_exists(self):
        """Test that WattPredictor package exists."""
        wattpredictor_dir = Path(__file__).parent.parent / "src" / "WattPredictor"
        assert wattpredictor_dir.exists()
        assert (wattpredictor_dir / "__init__.py").exists()
    
    def test_required_directories_exist(self):
        """Test that required directories exist."""
        project_root = Path(__file__).parent.parent
        
        required_dirs = [
            project_root / "data",
            project_root / "artifacts",
            project_root / "src",
            project_root / "tests",
        ]
        
        for directory in required_dirs:
            assert directory.exists(), f"Directory missing: {directory}"
    
    def test_training_pipeline_exists(self):
        """Test that training pipeline exists."""
        pipeline = Path(__file__).parent.parent / "src" / "WattPredictor" / "pipeline" / "training_pipeline.py"
        assert pipeline.exists()
    
    def test_app_exists(self):
        """Test that Streamlit app exists."""
        app_script = Path(__file__).parent.parent / "app.py"
        assert app_script.exists()
