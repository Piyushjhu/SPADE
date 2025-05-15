from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = os.path.join(lib_folder, 'requirements.txt')
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='spall_analysis',
    version='0.1.0', # Match __init__.py
    author='[Your Name or Organization Name]', # CHANGE THIS
    author_email='[Your Email]', # CHANGE THIS
    description='A Python toolkit for analyzing spall experiments from velocity data.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='[Optional: URL to GitHub repo or project page]', # Optional
    license='MIT', # Match LICENSE file
    packages=find_packages(), # Automatically find the 'spall_analysis' package
    install_requires=install_requires, # Read from requirements.txt
    python_requires='>=3.7', # Specify minimum Python version
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords='spall strength, shock physics, materials science, data analysis, velocity interferometry',
    # Include example data files if they are in data/ and needed
    include_package_data=True, # If you have non-code files inside the package dir
    package_data={
        # If data files are inside the spall_analysis module itself
        # 'spall_analysis': ['data/*.csv'],
    },
    # If data files are outside the module (e.g., in root data/ dir), use data_files
    # data_files=[('spall_analysis_data', ['data/combined_lit_table.csv', 'data/combined_lit_table_only_poly.csv'])], # Installs data to specified system dir
)