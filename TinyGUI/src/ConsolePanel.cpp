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
