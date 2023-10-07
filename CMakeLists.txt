# CMakeList.txt: CMakeProject1에 대한 CMake 프로젝트, 여기에 소스를 포함하고
# 프로젝트 특정 논리를 정의합니다.
#
cmake_minimum_required (VERSION 3.8)

set(CMAKE_PREFIX_PATH "C:\\Users\\user\\libtorch")
find_package(Torch REQUIRED)

# 이 프로젝트의 실행 파일에 소스를 추가합니다.
add_executable (CMakeProject1 "CMakeProject1.cpp" "train.cpp" "memory.cpp" "mcts.cpp" "PolicyValue.cpp" "GameManager.cpp" "CMakeProject1.h" "train.h" "memory.h" "mcts.h" "PolicyValue.h" "GameManager.h" "randm.h" "randm.cpp")
target_link_libraries(CMakeProject1 "${TORCH_LIBRARIES}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10000000")

if(MSVC)
	file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	add_custom_command(TARGET CMakeProject1 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}
	$<TARGET_FILE_DIR:CMakeProject1>)
endif(MSVC)

#if (CMAKE_VERSION VERSION_GREATER 3.12)
#  set_property(TARGET CMakeProject1 PROPERTY CXX_STANDARD 20)
#endif()
# TODO: 필요한 경우 테스트를 추가하고 대상을 설치합니다.
