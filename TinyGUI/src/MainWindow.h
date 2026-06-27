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
};

class GraphPanel;

#endif
