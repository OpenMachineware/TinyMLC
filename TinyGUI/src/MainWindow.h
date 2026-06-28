// TinyMLC - Tiny Machine Learning Compiler
//
// Copyright (c) 2026 Jia Liu & TinyMLC Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file is part of TinyMLC.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TINY_GUI_MAIN_WINDOW_H
#define TINY_GUI_MAIN_WINDOW_H

#include <QMainWindow>
#include <QSplitter>
#include <QProcess>
#include <QProgressBar>
#include <QLabel>
#include <QDragEnterEvent>
#include <QDropEvent>

enum class ProcessMode {
    None,
    Generate,
    Convert,
    Build
};

class ConsolePanel;
class ConfigPanel;
class GraphPanel;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dropEvent(QDropEvent *event) override;

private slots:
    void onGenerate();
    void onClear();
    void onAbout();
    void onStop();
    void loadConfig();
    void onExportLog();
    void onReadyReadStandardOutput();
    void onReadyReadStandardError();
    void onProcessFinished(int exitCode, QProcess::ExitStatus status);
    void setStatus(const QString& text, int progress);

private:
    void appendAnsiText(const QString& line);
    void createMenuBar();
    void createToolBar();
    void createCentralWidget();
    void runConvert(const QString& filePath);
    void runBuild();

    ConsolePanel* m_console;
    ConfigPanel*  m_config;
    GraphPanel*   m_graph;
    QSplitter*    m_mainSplitter;
    QSplitter*    m_rightSplitter;
    QProcess*     m_process;
    QProgressBar* m_progressBar;
    QLabel*       m_statusLabel;
    QString       m_outputBuffer;
    QString       m_currentColor;
    ProcessMode   m_currentMode;
    QString       m_modelInfoPath;
    QString       m_pythonPath;
    QString       m_scriptPath;
    QString       m_defaultTarget;
    QString       m_defaultMode;
    QString       m_defaultAccel;
    QString       m_gccArm;
    QString       m_gccRiscv;
    QString       m_qemuArm;
    QString       m_qemuRiscv;
    QString       m_cmsisPath;
    QString       m_nmsisPath;
};

class GraphPanel;

#endif
