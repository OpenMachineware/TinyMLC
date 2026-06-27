cmake_minimum_required(VERSION 3.16)
project(TinyGUI VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

add_executable(${PROJECT_NAME}
        src/main.cpp
        src/MainWindow.cpp
        src/MainWindow.h
        src/ConfigPanel.cpp
        src/ConfigPanel.h
        src/ConsolePanel.cpp
        src/ConsolePanel.h
        src/GraphPanel.cpp
        src/GraphPanel.h
)

target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Widgets)

set_target_properties(${PROJECT_NAME} PROPERTIES
        AUTOMOC ON
        AUTOUIC ON
)
