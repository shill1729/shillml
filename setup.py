from setuptools import setup


def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()


setup(
    name='shillml',
    version='0.0.47',
    packages=['shillml'],
    url='https://github.com/shill1729/shillml',
    install_requires=parse_requirements("requirements.txt"),
    license='MIT',
    author='Sean Hill',
    author_email='52792611+shill1729@users.noreply.github.com',
    description='Various machine learning algorithms'
)
