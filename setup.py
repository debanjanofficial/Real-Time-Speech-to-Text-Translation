from setuptools import setup, find_packages

setup(
    name="real_time_speech_translation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    author="Debanjan Chakraborty",
    author_email="dbjn.ckbrty99@gmail.com",
    description="A real-time speech-to-text translation system",
    keywords="speech recognition, translation, real-time",
    python_requires="=3.11",
)
