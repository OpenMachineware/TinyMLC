#ifndef TINY_GUI_CONSOLE_PANEL_H
#define TINY_GUI_CONSOLE_PANEL_H

#include <QWidget>
#include <QTextEdit>

class ConsolePanel : public QWidget {
    Q_OBJECT

public:
    explicit ConsolePanel(QWidget *parent = nullptr);

    void appendPlainText(const QString& text);
    void appendHtml(const QString& html);
    void clear();

private:
    QTextEdit* m_textEdit;
};

#endif
