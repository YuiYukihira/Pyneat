from setuptools import setup

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

with open('README.rst', 'r') as f:
    readme = f.read()

setup(name="pyneat",
      version="0.1.3",
      description="An implementation of Kenneth O. Stanley's NeuroEvolution of Augmenting Topologies",
      long_description=readme,
      url="https://gitlab.com/NeatJumper/Neat",
      author="Yui Yukihira",
      author_email="yui.yukihira42@gmail.com",
      license="MIT",
      packages=['pyneat'],
      install_requires=requirements,
      python_requires=">=3.6.4",
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      zip_safe=False,
      download_url="https://gitlab.com/NeatJumper/Neat/repository/0.1.3/archive.tar.gz",
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3 :: Only',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ])
