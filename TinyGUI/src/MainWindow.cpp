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

#include "MainWindow.h"
#include "ConsolePanel.h"
#include "ConfigPanel.h"
#include "GraphPanel.h"
#include "ConfigDialog.h"

#include <QMenuBar>
#include <QToolBar>
#include <QMessageBox>
#include <QAction>
#include <QSplitter>
#include <QVBoxLayout>
#include <QRegularExpression>
#include <QProgressBar>
#include <QStyle>
#include <QDir>
#include <QMimeData>
#include <QJsonDocument>
#include <QJsonObject>
#include <QFileDialog>
#include <QTextStream>
#include <QDateTime>
#include <QStatusBar>


static const QString CONFIG_PATH = QDir::homePath() + "/.tinymlc/config.json";


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_process(nullptr)
    , m_currentMode(ProcessMode::None)
    , m_modelInfoPath(QDir::currentPath() + "/model_info.json")
    , m_statusLabel(nullptr)
    , m_progressBar(nullptr) {
    setWindowTitle(tr("TinyMLC"));
    resize(1024, 768);

    createMenuBar();
    createToolBar();
    createCentralWidget();

    setStatus(tr("Ready"), 0);
    statusBar()->showMessage(tr("Ready"));

    setAcceptDrops(true);
}

MainWindow::~MainWindow() {
    if (m_process && m_process->state() != QProcess::NotRunning) {
        m_process->terminate();
        m_process->waitForFinished(1000);
    }
}

void MainWindow::createMenuBar() {
    QMenu* fileMenu = menuBar()->addMenu(tr("&File"));

    QAction* generateAction = new QAction(tr("&Generate"), this);
    generateAction->setShortcut(QKeySequence("Ctrl+G"));
    connect(generateAction, &QAction::triggered, this,
        &MainWindow::onGenerate);
    fileMenu->addAction(generateAction);

    QAction* stopAction = new QAction(tr("&Stop"), this);
    stopAction->setShortcut(QKeySequence("Ctrl+S"));
    connect(stopAction, &QAction::triggered, this, &MainWindow::onStop);
    fileMenu->addAction(stopAction);

    QAction* clearAction = new QAction(tr("&Clear"), this);
    clearAction->setShortcut(QKeySequence("Ctrl+L"));
    connect(clearAction, &QAction::triggered, this, &MainWindow::onClear);
    fileMenu->addAction(clearAction);

    fileMenu->addSeparator();

    // Export Log
    QAction* exportAction = new QAction(tr("&Export Log..."), this);
    exportAction->setShortcut(QKeySequence("Ctrl+E"));
    connect(exportAction, &QAction::triggered, this, &MainWindow::onExportLog);
    fileMenu->addAction(exportAction);

    fileMenu->addSeparator();

    QAction* exitAction = new QAction(tr("E&xit"), this);
    exitAction->setShortcut(QKeySequence("Ctrl+Q"));
    connect(exitAction, &QAction::triggered, this, &QWidget::close);
    fileMenu->addAction(exitAction);

    QMenu* helpMenu = menuBar()->addMenu(tr("&Help"));
    QAction* aboutAction = new QAction(tr("&About"), this);
    connect(aboutAction, &QAction::triggered, this, &MainWindow::onAbout);
    helpMenu->addAction(aboutAction);

    QMenu* settingsMenu = menuBar()->addMenu(tr("&Settings"));
    QAction* settingsAction = new QAction(tr("&Preferences..."), this);
    connect(settingsAction, &QAction::triggered, this, [this]() {
        ConfigDialog dialog(this);
        dialog.exec();
    });
    settingsMenu->addAction(settingsAction);
}

void MainWindow::createToolBar() {
    QToolBar* toolbar = addToolBar(tr("Main"));
    toolbar->setMovable(false);
    toolbar->setIconSize(QSize(20, 20));

    // ---- Left side: actions ----
    QAction* generateAction = new QAction(
        QIcon::fromTheme("media-playback-start",
            style()->standardIcon(QStyle::SP_MediaPlay)),
        tr("Generate"), this
    );
    connect(generateAction, &QAction::triggered, this,
        &MainWindow::onGenerate);
    toolbar->addAction(generateAction);

    QAction* stopAction = new QAction(
        QIcon::fromTheme("media-playback-stop",
            style()->standardIcon(QStyle::SP_MediaStop)),
        tr("Stop"), this
    );
    connect(stopAction, &QAction::triggered, this, &MainWindow::onStop);
    toolbar->addAction(stopAction);

    QAction* clearAction = new QAction(
        QIcon::fromTheme("edit-clear",
            style()->standardIcon(QStyle::SP_DialogResetButton)),
        tr("Clear"), this
    );
    connect(clearAction, &QAction::triggered, this, &MainWindow::onClear);
    toolbar->addAction(clearAction);

    // ---- Spacer to push settings to the right ----
    QWidget* spacer = new QWidget();
    spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    toolbar->addWidget(spacer);

    // ---- Export Log ----
    QAction* exportAction = new QAction(
        QIcon::fromTheme("document-save",
            style()->standardIcon(QStyle::SP_DialogSaveButton)),
        tr("Export Log"), this);
    connect(exportAction, &QAction::triggered, this, &MainWindow::onExportLog);
    toolbar->addAction(exportAction);

    // ---- Right side: Settings ----
    QAction* settingsAction = new QAction(
        QIcon::fromTheme("preferences-system",
            style()->standardIcon(QStyle::SP_ComputerIcon)),
        tr("Settings"), this);
    connect(settingsAction, &QAction::triggered, this, [this]() {
        ConfigDialog dialog(this);
        dialog.exec();
    });
    toolbar->addAction(settingsAction);

    // ---- Title label (optional) ----
    QLabel* titleLabel = new QLabel(tr(" TinyMLC Config "), this);
    titleLabel->setStyleSheet("color: #aaaaaa; font-size: 13px;");
    toolbar->addWidget(titleLabel);
}

void MainWindow::createCentralWidget() {
    // Main splitter: Config(left) | Console + Progress + Graph(right)
    m_mainSplitter = new QSplitter(Qt::Horizontal, this);
    m_mainSplitter->setChildrenCollapsible(false);  // make it draggable
    m_mainSplitter->setHandleWidth(6);

    m_config = new ConfigPanel(m_mainSplitter);
    m_mainSplitter->addWidget(m_config);

    // Right splitter, Console(top) | Progress + Graph(bottom)
    m_rightSplitter = new QSplitter(Qt::Vertical, m_mainSplitter);
    m_rightSplitter->setChildrenCollapsible(false);  // make it draggable
    m_mainSplitter->setHandleWidth(6);

    // Console
    m_console = new ConsolePanel(m_rightSplitter);
    m_rightSplitter->addWidget(m_console);

    // Bottom：(Progress + Status) + Graph
    QWidget* bottomWidget = new QWidget(m_rightSplitter);
    QVBoxLayout* bottomLayout = new QVBoxLayout(bottomWidget);
    bottomLayout->setSpacing(4);
    bottomLayout->setContentsMargins(0, 0, 0, 0);

    // Progress + Status
    QHBoxLayout* statusLayout = new QHBoxLayout();
    m_statusLabel = new QLabel(tr("Ready"), bottomWidget);
    m_statusLabel->setFixedWidth(120);
    m_statusLabel->setStyleSheet("color: #aaaaaa; font-size: 11px;");
    statusLayout->addWidget(m_statusLabel);

    m_progressBar = new QProgressBar(bottomWidget);
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);
    m_progressBar->setTextVisible(true);
    m_progressBar->setFormat(tr("%p%"));
    m_progressBar->setFixedHeight(20);
    statusLayout->addWidget(m_progressBar, 1);

    bottomLayout->addLayout(statusLayout);

    // Graph
    m_graph = new GraphPanel(bottomWidget);
    bottomLayout->addWidget(m_graph, 1);

    m_rightSplitter->addWidget(bottomWidget);

    // Add to main splitter
    m_mainSplitter->addWidget(m_rightSplitter);

    // init size
    m_mainSplitter->setSizes({200, 744});  // left: 280px, right: left
    m_rightSplitter->setSizes({300, 444}); // top: 300px, bottom: 444px

    setCentralWidget(m_mainSplitter);
}

void MainWindow::appendAnsiText(const QString& line) {
    QString text = line;

    // Update color when it has ANSI set color codes
    if (text.contains("\u001B[36m")) {
        m_currentColor = "<font color=\"#50ffff\">";
    }
    // Clear color when it has ANSI reset color codes
    if (text.contains("\u001B[0m")) {
        // No clear, waiting for next line
        // Waiting for next color
    }

    // Replace ANSI codes with HTML tags
    text.replace("\u001B[36m", "<font color=\"#50ffff\">");
    text.replace("\u001B[34m", "<font color=\"#50a0ff\">");
    text.replace("\u001B[32m", "<font color=\"#50ff50\">");
    text.replace("\u001B[33m", "<font color=\"#ffff50\">");
    text.replace("\u001B[31m", "<font color=\"#ff5050\">");
    text.replace("\u001B[0m", "</font>");
    text.replace("\u001B[1m", "<b>");
    text.replace("\u001B[22m", "</b>");

    // Add the last color to a line, when it have no <font>,
    // but m_currentColor do have a value
    if (!m_currentColor.isEmpty() && !text.contains("<font")) {
        text = m_currentColor + text;
    }

    // Remove any leftover ANSI codes
    QRegularExpression ansi("\u001B\\[[0-9;]*m");
    text.replace(ansi, "");

    // Add missing </font>
    if (text.contains("<font") && text.count("<font")
        > text.count("</font>")) {
        text += "</font>";
    }

    // Directly append HTML without extra escaping
    // QTextEdit::appendHtml handles HTML tags correctly
    // Append <br>
    m_console->appendHtml(text + "<br>");
}

void MainWindow::onGenerate() {
    if (m_process && m_process->state() != QProcess::NotRunning) {
        m_console->appendPlainText(tr("⚠️ Process already running"));
        return;
    }

    m_currentMode = ProcessMode::Generate;
    setStatus(tr("Generating..."), 0);
    statusBar()->showMessage(tr("Generating..."));

    int maxMacs = m_config->getMaxMacs();
    int maxRam = m_config->getMaxRam();
    int maxFlash = m_config->getMaxFlash();
    int gens = m_config->getGenerations();
    int pop = m_config->getPopulation();

    m_console->appendPlainText(tr("▶ Starting network generation...\n"));
    setStatus(tr("Generating network..."), 20);

    QString pythonPath = "/Users/jia/Desktop/TinyMLC/.venv/bin/python";
    QString scriptPath = "/Users/jia/Desktop/TinyMLC/main.py";

    // FIXME: Current dir as output_dir
    // It should be configured or specified by file dialog
    QString outputDir = QDir::currentPath();
    QString modelInfoPath = outputDir + "/model_info.json";

    QStringList args;
    args << scriptPath
         << "generate"
         << "--task-type" << "classification"
         << "--input-shape" << "1,28,28,1"
         << "--output-shape" << "1,10"
         << "--max-macs" << QString::number(maxMacs)
         << "--max-ram" << QString::number(maxRam)
         << "--max-flash" << QString::number(maxFlash)
         << "--estimator" << "software"
         << "--generate-mode" << "genetic"
         << "--generations" << QString::number(gens)
         << "--population" << QString::number(pop)
         << "--dump-model" << modelInfoPath
         << "--mode" << "debug";

    m_process = new QProcess(this);

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("FORCE_COLOR", "1");
    env.insert("CLICOLOR_FORCE", "1");
    env.insert("TERM", "xterm-256color");
    m_process->setProcessEnvironment(env);

    m_process->setProgram(pythonPath);
    m_process->setArguments(args);

    connect(m_process, &QProcess::readyReadStandardOutput,
            this, &MainWindow::onReadyReadStandardOutput);
    connect(m_process, &QProcess::readyReadStandardError,
            this, &MainWindow::onReadyReadStandardError);
    connect(m_process, &QProcess::finished,
            this, &MainWindow::onProcessFinished);

    m_process->start();

    if (!m_process->waitForStarted(1000)) {
        m_console->appendPlainText(tr("❌ Failed to start TinyMLC"));
        setStatus(tr("Error"), 0);
        statusBar()->showMessage(tr("Error"));
        delete m_process;
        m_process = nullptr;
        return;
    }
    statusBar()->showMessage(tr("Running (PID: %1)")
        .arg(m_process->processId()));
}

void MainWindow::onReadyReadStandardOutput() {
    if (!m_process) return;
    QByteArray data = m_process->readAllStandardOutput();
    if (data.isEmpty()) return;

    QString text = QString::fromUtf8(data);
    m_outputBuffer += text;

    int pos;
    while ((pos = m_outputBuffer.indexOf('\n')) != -1) {
        QString line = m_outputBuffer.left(pos);
        m_outputBuffer = m_outputBuffer.mid(pos + 1);

        // Check if it is a model_info line
        if (line.startsWith("MODEL_INFO: ")) {
            QString jsonStr = line.mid(12);  // Strip "MODEL_INFO: "
            // Update GraphPanel
            m_graph->loadModelInfoFromJson(jsonStr);
            continue;
        }
        if (line.startsWith("OPTIMIZED_MODEL: ")) {
            QString jsonStr = line.mid(17);
            m_graph->loadModelInfoFromJson(jsonStr);
            continue;
        }

        // normal log line
        appendAnsiText(line);
    }
}


void MainWindow::onReadyReadStandardError() {
    if (!m_process) return;
    QByteArray data = m_process->readAllStandardError();
    if (data.isEmpty()) return;

    QString text = QString::fromUtf8(data);
    m_console->appendHtml(QString("<font color=\"#ff8888\">%1</font>")
                          .arg(text.toHtmlEscaped()));
}

void MainWindow::onProcessFinished(int exitCode, QProcess::ExitStatus status) {
    // Display any remaining buffered text
    if (!m_outputBuffer.isEmpty()) {
        appendAnsiText(m_outputBuffer);
        m_outputBuffer.clear();
    }

    if (status == QProcess::NormalExit && exitCode == 0) {
        // FIXME: the patch should be calced.
        QString modelInfoPath = QDir::currentPath() + "/model_info.json";
        m_graph->loadModelInfo(modelInfoPath);
        m_graph->setReady(true);
        m_config->setModelInfo(modelInfoPath);

        if (m_currentMode == ProcessMode::Generate) {
            m_console->appendPlainText(tr("✅ Generation complete!\n"));
            runBuild();
        } else if (m_currentMode == ProcessMode::Convert) {
            m_console->appendPlainText(tr("✅ Conversion complete!\n"));
        }

        setStatus(tr("Ready"), 100);
        statusBar()->showMessage(tr("Ready"));
    } else {
        m_console->appendPlainText(tr("❌ Process failed (code: %1)\n")
            .arg(exitCode));
        setStatus(tr("Error"), 0);
        statusBar()->showMessage(tr("Error"));
    }

    m_process = nullptr;
    m_currentMode = ProcessMode::None;
}

void MainWindow::runBuild() {
    m_console->appendPlainText(tr("🔨 Building and running...\n"));

    QProcess* buildProcess = new QProcess(this);
    buildProcess->setWorkingDirectory(QDir::currentPath());

    // FIXME: config
    QString buildScript = "./build_riscv_pure_c_debug.sh";

    connect(buildProcess, &QProcess::readyReadStandardOutput,
            this, [this, buildProcess]() {
        QString output = buildProcess->readAllStandardOutput();
        m_console->appendPlainText(output);
    });
    connect(buildProcess, &QProcess::readyReadStandardError,
            this, [this, buildProcess]() {
        QString output = buildProcess->readAllStandardError();
        m_console->appendPlainText("⚠️ " + output);
    });
    connect(buildProcess, &QProcess::finished, this,
            [this, buildProcess](int code, QProcess::ExitStatus status) {
        if (status == QProcess::NormalExit && code == 0) {
            m_console->appendPlainText(tr("✅ Build and run complete!\n"));
        } else {
            m_console->appendPlainText(tr("❌ Build failed (code: %1)\n")
                .arg(code));
        }
        buildProcess->deleteLater();
    });

    buildProcess->start("bash", {buildScript});
    if (!buildProcess->waitForStarted(3000)) {
        m_console->appendPlainText(tr("❌ Failed to start build script\n"));
        buildProcess->deleteLater();
    }
}

void MainWindow::runConvert(const QString& filePath) {
    if (m_process && m_process->state() != QProcess::NotRunning) {
        m_console->appendPlainText(tr("⚠️ Process already running"));
        return;
    }

    m_currentMode = ProcessMode::Convert;
    setStatus(tr("Converting..."), 20);

    QString pythonPath = "/Users/jia/Desktop/TinyMLC/.venv/bin/python";
    QString scriptPath = "/Users/jia/Desktop/TinyMLC/main.py";

    QStringList args;
    args << scriptPath
         << "convert"
         << "--model" << filePath
         << "--dump-model" << m_modelInfoPath;

    m_process = new QProcess(this);
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("FORCE_COLOR", "1");
    env.insert("CLICOLOR_FORCE", "1");
    env.insert("TERM", "xterm-256color");
    m_process->setProcessEnvironment(env);

    connect(m_process, &QProcess::readyReadStandardOutput,
            this, &MainWindow::onReadyReadStandardOutput);
    connect(m_process, &QProcess::readyReadStandardError,
            this, &MainWindow::onReadyReadStandardError);
    connect(m_process, &QProcess::finished,
            this, &MainWindow::onProcessFinished);

    m_process->setProgram(pythonPath);
    m_process->setArguments(args);
    m_process->start();

    if (!m_process->waitForStarted(3000)) {
        m_console->appendPlainText(tr("❌ Failed to start converter\n"));
        setStatus(tr("Error"), 0);
        delete m_process;
        m_process = nullptr;
        return;
    }
}

void MainWindow::onExportLog() {
    // Get current console context
    QString logContent = m_console->getPlainText();

    if (logContent.isEmpty() || logContent == "Ready...") {
        QMessageBox::information(this, tr("Export Log"),
                                 tr("Console is empty. Nothing to export."));
        return;
    }

    // Default filename：tinymlc_log_YYYYMMDD_HHMMSS.log
    QString defaultName = QString("tinymlc_log_%1.log")
        .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));

    QString filePath = QFileDialog::getSaveFileName(
        this,
        tr("Export Log"),
        defaultName,
        tr("Log Files (*.log);;Text Files (*.txt);;All Files (*)")
    );

    if (filePath.isEmpty()) return;

    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, tr("Export Log"),
                             tr("Failed to write file: %1").arg(filePath));
        return;
    }

    QTextStream stream(&file);
    stream << logContent;
    file.close();

    statusBar()->showMessage(tr("Log exported to: %1").arg(filePath), 3000);
}

void MainWindow::dragEnterEvent(QDragEnterEvent *event) {
    if (event->mimeData()->hasUrls()) {
        QList<QUrl> urls = event->mimeData()->urls();
        if (!urls.isEmpty()) {
            QString path = urls.first().toLocalFile();
            if (path.endsWith(".onnx") || path.endsWith(".tflite")) {
                event->acceptProposedAction();
                return;
            }
        }
    }
    event->ignore();
}

void MainWindow::dropEvent(QDropEvent *event) {
    QList<QUrl> urls = event->mimeData()->urls();
    if (urls.isEmpty()) return;

    QString path = urls.first().toLocalFile();
    if (path.endsWith(".onnx") || path.endsWith(".tflite")) {
        m_console->appendPlainText(tr("📥 Dropped: %1").arg(path));
        runConvert(path);
        event->acceptProposedAction();
    }
}

void MainWindow::loadConfig() {
    QFile file(CONFIG_PATH);
    if (!file.exists()) return;

    if (!file.open(QIODevice::ReadOnly)) return;
    QByteArray data = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (doc.isNull()) return;

    QJsonObject obj = doc.object();

    m_pythonPath = obj["python_path"].toString();
    m_scriptPath = obj["script_path"].toString();
    m_defaultTarget = obj["target"].toString("riscv");
    m_defaultMode = obj["mode"].toString("debug");
    m_defaultAccel = obj["accel"].toString("pure-c");
    m_gccArm = obj["gcc_arm"].toString("arm-none-eabi-gcc");
    m_gccRiscv = obj["gcc_riscv"].toString("riscv-none-elf-gcc");
    m_qemuArm = obj["qemu_arm"].toString("qemu-system-arm");
    m_qemuRiscv = obj["qemu_riscv"].toString("qemu-system-riscv32");
    m_cmsisPath = obj["cmsis_nn_path"].toString();
    m_nmsisPath = obj["nmsis_nn_path"].toString();
}

void MainWindow::onStop() {
    if (!m_process) {
        m_console->appendPlainText(tr("⚠️ No process running"));
        return;
    }

    if (m_process->state() == QProcess::NotRunning) {
        m_console->appendPlainText(tr("⚠️ Process already finished"));
        m_process = nullptr;
        return;
    }

    m_console->appendPlainText(tr("⏹ Stopping process..."));
    m_process->terminate();

    if (!m_process->waitForFinished(2000)) {
        m_process->kill();
        m_process->waitForFinished(1000);
    }

    m_console->appendPlainText(tr("✅ Process stopped\n"));
    setStatus(tr("Stopped"), 0);
    m_process = nullptr;
}

void MainWindow::onClear() {
    m_console->clear();
    m_console->appendPlainText(tr("Ready...\n"));
    m_outputBuffer.clear();
}

void MainWindow::onAbout() {
    QMessageBox::about(this, tr("About TinyMLC"),
        tr("TinyMLC\n"
           "Version 0.1.0\n\n"
           "Automatic Network Generator + Code Generator for MCU"));
}

void MainWindow::setStatus(const QString& text, int progress) {
    m_statusLabel->setText(text);
    m_progressBar->setValue(progress);
}
