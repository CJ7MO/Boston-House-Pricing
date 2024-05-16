import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "v0.0.0"

REPO_NAME = "Boston-House-Pricing"
AUTHOR_USER_NAME = "cj7mo"
SRC_REPO = "BOSTON-HOUSE-PRICING"
AUTHOR_EMAIL = "jcobando@uninorte.edu.co"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for ml app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
)