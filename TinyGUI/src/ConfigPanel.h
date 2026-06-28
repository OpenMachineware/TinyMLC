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

#ifndef TINY_GUI_CONFIG_PANEL_H
#define TINY_GUI_CONFIG_PANEL_H

#include <QWidget>
#include <QSpinBox>
#include <QLabel>
#include <QGroupBox>

class ConfigPanel : public QWidget {
    Q_OBJECT

public:
    explicit ConfigPanel(QWidget *parent = nullptr);

    int getMaxMacs() const;
    int getMaxRam() const;
    int getMaxFlash() const;
    int getGenerations() const;
    int getPopulation() const;

    void setModelInfo(const QString& jsonPath);

private:
    QSpinBox* m_macsSpin;
    QSpinBox* m_ramSpin;
    QSpinBox* m_flashSpin;
    QSpinBox* m_gensSpin;
    QSpinBox* m_popSpin;

    // Model info display
    QGroupBox* m_infoGroup;
    QLabel* m_inputShapeLabel;
    QLabel* m_outputShapeLabel;
    QLabel* m_layersLabel;
    QLabel* m_macsInfoLabel;
    QLabel* m_paramsLabel;
    QLabel* m_ramInfoLabel;
    QLabel* m_flashInfoLabel;
};

#endif
