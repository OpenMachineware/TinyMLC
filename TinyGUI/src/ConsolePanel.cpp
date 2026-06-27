#include "ConsolePanel.h"

#include <QVBoxLayout>
#include <QLabel>

ConsolePanel::ConsolePanel(QWidget *parent)
    : QWidget(parent) {
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setSpacing(4);

    QLabel* label = new QLabel(tr("Console Output"), this);
    QFont font = label->font();
    font.setBold(true);
    label->setFont(font);
    layout->addWidget(label);

    m_textEdit = new QTextEdit(this);
    m_textEdit->setReadOnly(true);
    m_textEdit->setFontFamily("Courier New, monospace");
    m_textEdit->setFontPointSize(11);
    m_textEdit->setStyleSheet(
        "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; }"
    );

    layout->addWidget(m_textEdit);
    appendPlainText(tr("Ready..."));
}

void ConsolePanel::appendPlainText(const QString& text) {
    m_textEdit->append(text);
}

void ConsolePanel::appendHtml(const QString& html) {
    QString text = html;
    text.replace("\r\n", "<br>");
    text.replace("\n", "<br>");
    text.replace("\r", "<br>");

    // Remove <br> when the line is start with it
    if (text.startsWith("<br>")) {
        text = text.mid(4);
    }

    m_textEdit->moveCursor(QTextCursor::End);
    m_textEdit->insertHtml(text);
    m_textEdit->moveCursor(QTextCursor::End);
}

void ConsolePanel::clear() {
    m_textEdit->clear();
}
