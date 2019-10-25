from setuptools import setup
from distutils.extension import Extension
import re
import os
import codecs
here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        version_file,
        re.M,
    )
    if version_match:
        return version_match.group(1)

    raise RuntimeError("Unable to find version string.")

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }
ext_modules = [ ]

#print(find_version("deepgmap", "version.py"))
setup(
    name='easy_heatmapper',
    #version=VERSION,
    version=find_version( "version.py"),
    description='Drawing clustered heatmap.',
    author='Koh Onimaru',
    author_email='koh.onimaru@gmail.com',
    url='',
    py_modules=['easy_heatmapper'],
    #packages=find_packages('deepgmap'),
    #packages=['deepgmap.'],
    provides=['easy_heatmapper'],
    #package_data = {
    #     '': ['enhancer_prediction/*', '*.pyx', '*.pxd', '*.c', '*.h'],
    #},
    #packages=find_packages(),
    cmdclass = cmdclass,
    ext_modules=ext_modules,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: Apache Software License ',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        
    
    ],
    install_requires=[ 'numpy', 'matplotlib', 'sklearn', 'fastcluster', 'umap-learn', 'scipy', 'MulticoreTSNE','tornado'],
    long_description=open('README.md').read(),
)