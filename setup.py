from setuptools import setup

setup(name="pyneat",
      version="0.1",
      description="An implementation of Kenneth O. Stanley's NeuroEvolution of Augmenting Topologies",
      url="https://gitlab.com/NeatJumper/Neat",
      author="Yui Yukihira",
      author_email="yui.yukihira42@gmail.com",
      license="MIT",
      packages=['pyneat'],
      install_requires=[
          'tensorflow'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      zip_safe=False)
