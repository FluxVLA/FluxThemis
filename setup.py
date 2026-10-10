from pathlib import Path

from setuptools import setup


def read_requirements():
    lines = Path('requirements.txt').read_text(encoding='utf-8').splitlines()
    return [line for line in lines if line and not line.startswith('#')]


if __name__ == '__main__':
    setup(install_requires=read_requirements())
