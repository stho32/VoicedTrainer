from setuptools import setup, find_packages

setup(
    name="voiced_trainer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add dependencies here
    ],
    entry_points={
        'console_scripts': [
            'voiced-trainer=voiced_trainer.main:main',
        ],
    },
    author="",
    author_email="",
    description="VoicedTrainer application",
    keywords="voice, trainer",
    url="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
)
