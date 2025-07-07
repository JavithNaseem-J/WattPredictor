import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

project_name ="WattPredictor"

list_of_files = [
    "config_file/config.yaml",
    "config_file/params.yaml",
    "config_file/schema.yaml",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/config/config.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/helpers.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/config_entity/__init__.py",
    f"src/{project_name}/config_entity/config_entity.py",
    f"src/{project_name}/__init__.py",
    "notebooks/eda.ipynb",
    "main.py",
    "app.py",
    "pyproject.toml",
    "setup.py",
    "Dockerfile",
    "tests/__init__.py",
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir,file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Created directory: {file_dir}")

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            with open(file_path, 'w') as f:
                pass
            logging.info(f"Created file: {file_path}")
    else:
        logging.info(f"File already exists: {file_path}")