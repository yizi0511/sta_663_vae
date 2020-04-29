import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sta-663-vae", # Replace with your own username
    version="1.0.0",
    author="Chenxi Wu, Yizi Zhang",
    author_email="chenxi.wu@duke.edu, yizi.zhang@duke.edu",
    description="Final project of STA 663: Implementation of Variational Autoencoder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yizi0511/sta_663_vae",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
	"Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
	'cupy',
	'tensorflow',
	'tensorflow-gpu',
    ],
)
