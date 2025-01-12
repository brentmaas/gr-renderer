project('gr-renderer', 'cpp', 'c')

compiler_c = meson.get_compiler('c')
compiler_cpp = meson.get_compiler('cpp')
if compiler_c.get_id() != 'clang' or compiler_cpp.get_id() != 'clang'
    error('Please use clang/clang++ as your compiler as it contains a required c++23 feature')
endif

source = [
    'src/glad.c',
    'src/main.cpp'
]
cpp_args = [
    '-Wall',
    '-Wextra',
    '-std=c++23'
]
dependencies = [
    dependency('glfw3'),
    dependency('libjpeg')
]
executable('gr-renderer', source, include_directories: 'include', cpp_args: cpp_args, dependencies: dependencies)
