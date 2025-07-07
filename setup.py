import setuptools

__version__ = "0.0.0"

REPO_NAME = "WattPredictor"
AUTHOR_USER_NAME = "JavithNaseem-J"
SRC_REPO = "WattPredictor"
AUTHOR_EMAIL = "Javithnaseem.j@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for WattPredictor",
    long_description='Hourly Watt Demand Prediction using Machine Learning',
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)