cmake_minimum_required(VERSION 3.26)
project(CppSelfPlay)


find_package(Torch REQUIRED)

find_package(Python3 3.11 COMPONENTS Interpreter Development REQUIRED)
find_package(PythonLibs 3.11 REQUIRED)
find_package(pybind11 REQUIRED)
add_library(CppSelfPlay main.cpp
        MCTS/Node.cpp
        MCTS/Node.h
        MCTS/MuzeroSearchTree.cpp
        MCTS/MuzeroSearchTree.h
        Buffers/MuZeroFrameBuffer.cpp
        Buffers/MuZeroFrameBuffer.h
        utils.cpp
        utils.h
        Network/MuZeroDefaultNet.cpp
        Network/MuZeroDefaultNet.h
        CppSelfPlay.cpp
)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(CppSelfPlay PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(CppSelfPlay PRIVATE "${PYTHON_LIBRARIES}")
target_link_libraries(CppSelfPlay PRIVATE ${MY_LIBRARIES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -fopenmp /std:c++17")
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET CppSelfPlay
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:CppSelfPlay>)
    
endif (MSVC)

