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

#ifndef TINY_GUI_CONFIG_DIALOG_H
#define TINY_GUI_CONFIG_DIALOG_H

#include <QDialog>
#include <QLineEdit>
#include <QComboBox>
#include <QPushButton>

class ConfigDialog : public QDialog {
    Q_OBJECT

public:
    explicit ConfigDialog(QWidget *parent = nullptr);

    void loadConfig();
    void saveConfig();

private slots:
    void onAccept();
    void onBrowsePython();
    void onBrowseScript();
    void onBrowseCmsis();
    void onBrowseNmsis();

private:
    void setupUI();
    QString browseFile(const QString& title, const QString& filter = "");

    QLineEdit* m_pythonPath;
    QLineEdit* m_scriptPath;
    QComboBox* m_targetCombo;
    QComboBox* m_modeCombo;
    QComboBox* m_accelCombo;
    QLineEdit* m_gccArm;
    QLineEdit* m_gccRiscv;
    QLineEdit* m_qemuArm;
    QLineEdit* m_qemuRiscv;
    QLineEdit* m_cmsisPath;
    QLineEdit* m_nmsisPath;
};

#endif
