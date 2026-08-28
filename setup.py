from setuptools import find_packages, setup

package_name = 'traffic_light'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hyeonjun',
    maintainer_email='hyeonjun@todo.todo',
    description='Traffic light detection package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector = traffic_light.traffic_light_detector:main',
        ],
    },
)
