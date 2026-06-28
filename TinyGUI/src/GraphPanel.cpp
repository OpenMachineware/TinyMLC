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

#include "GraphPanel.h"

#include <QPainter>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>
#include <QDebug>

GraphPanel::GraphPanel(QWidget *parent)
    : QWidget(parent), m_ready(false), m_pendingRefresh(false)
    , m_animating(false) {
    setMinimumHeight(150);
    setStyleSheet("background-color: #252525;");

    m_refreshTimer.setSingleShot(true);
    m_refreshTimer.setInterval(50);  // 50ms
    connect(&m_refreshTimer, &QTimer::timeout, this,
        &GraphPanel::onRefreshTimer);

    m_animTimer.setInterval(16);  // ~60fps
    connect(&m_animTimer, &QTimer::timeout, this, &GraphPanel::onAnimTimer);
}

void GraphPanel::loadModelInfo(const QString& jsonPath) {
    QFile file(jsonPath);
    if (!file.open(QIODevice::ReadOnly)) {
        qDebug() << "Cannot open" << jsonPath;
        return;
    }

    QByteArray data = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (doc.isNull()) {
        qDebug() << "Invalid JSON";
        return;
    }

    QJsonObject root = doc.object();
    QJsonArray ops = root["ops"].toArray();
    parseOps(ops);
    layoutNodes();
    update();
}

void GraphPanel::loadModelInfoFromJson(const QString& jsonStr) {
    QJsonDocument doc = QJsonDocument::fromJson(jsonStr.toUtf8());
    if (doc.isNull()) {
        qDebug() << "Invalid JSON for model_info";
        return;
    }

    QJsonObject root = doc.object();
    QJsonArray ops = root["ops"].toArray();

    bool isOptimized = root.contains("optimized") && root["optimized"].toBool();

    parseOps(ops);

    if (isOptimized) {
        for (auto& node : m_nodes) {
            node.color = QColor(0xff, 0x66, 0x00);
        }
    } else {
        for (auto& node : m_nodes) {
            node.color = QColor(0x00, 0x99, 0xff);
        }
    }

    layoutNodes();
    requestRefresh();
}

void GraphPanel::parseOps(const QJsonArray& ops) {
    m_nodes.clear();
    m_edges.clear();

    int centerX = width() / 2;
    int centerY = height() / 2;

    // ---- Create nodes ----
    for (int i = 0; i < ops.size(); ++i) {
        QJsonObject op = ops[i].toObject();
        QString opName = op["op_name"].toString();
        GraphNode node;
        node.label = opName;
        node.radius = 30;
        node.color = QColor(0x00, 0x99, 0xff);
        node.ready = false;
        node.pos = QPoint(centerX + (i - ops.size()/2) * 20, centerY);
        node.targetPos = node.pos;
        m_nodes.append(node);
    }

    // ---- Create edges ----
    for (int i = 0; i < ops.size(); ++i) {
        QJsonObject op = ops[i].toObject();
        QJsonArray inputs = op["input_indices"].toArray();
        QJsonArray outputs = op["output_indices"].toArray();

        if (outputs.isEmpty()) continue;
        int outputIdx = outputs[0].toInt();

        // Find output node index
        int toIdx = -1;
        for (int j = 0; j < ops.size(); ++j) {
            QJsonObject other = ops[j].toObject();
            QJsonArray otherOutputs = other["output_indices"].toArray();
            if (!otherOutputs.isEmpty()
                && otherOutputs[0].toInt() == outputIdx) {
                toIdx = j;
                break;
            }
        }

        // Find input node index
        for (const auto& inputVal : inputs) {
            int inputIdx = inputVal.toInt();
            int fromIdx = -1;
            for (int j = 0; j < ops.size(); ++j) {
                QJsonObject other = ops[j].toObject();
                QJsonArray otherOutputs = other["output_indices"].toArray();
                if (!otherOutputs.isEmpty()
                    && otherOutputs[0].toInt() == inputIdx) {
                    fromIdx = j;
                    break;
                }
            }
            if (fromIdx != -1 && toIdx != -1 && fromIdx != toIdx) {
                GraphEdge edge;
                edge.from = fromIdx;
                edge.to = toIdx;
                m_edges.append(edge);
            }
        }
    }
}

void GraphPanel::onAnimTimer() {
    bool allDone = true;
    for (auto& node : m_nodes) {
        QPoint delta = node.targetPos - node.pos;
        if (abs(delta.x()) > 1 || abs(delta.y()) > 1) {
            node.pos += delta / 4;
            allDone = false;
        } else {
            node.pos = node.targetPos;
        }
    }
    update();

    if (allDone) {
        m_animTimer.stop();
        m_animating = false;
    }
}

void GraphPanel::layoutNodes() {
    if (m_nodes.isEmpty()) return;

    int width = this->width();
    int height = this->height();
    int nodeCount = m_nodes.size();

    int nodesPerRow = 8;
    int rows = (nodeCount + nodesPerRow - 1) / nodesPerRow;
    int cols = qMin(nodeCount, nodesPerRow);

    // Dynamic calc spacing, 100px at least. Scale when nodes too mcuh.
    int spacing = qMax(80, qMin(120, (width - 100) / cols));
    int totalWidth = (cols - 1) * spacing;
    int startX = (width - totalWidth) / 2;
    int rowHeight = 120;

    for (int i = 0; i < nodeCount; ++i) {
        int row = i / nodesPerRow;
        int col = i % nodesPerRow;
        int x = startX + col * spacing;
        int y = (height / 2) + (row - (rows - 1) / 2.0) * rowHeight;
        m_nodes[i].targetPos = QPoint(x, y);
    }

    if (!m_animating) {
        m_animating = true;
        m_animTimer.start();
    }
}

void GraphPanel::setReady(bool ready) {
    m_ready = ready;
    update();
}

void GraphPanel::paintEvent(QPaintEvent *event) {
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // Draw edges
    for (const auto& edge : m_edges) {
        drawEdge(painter, edge);
    }

    // Draw nodes
    for (const auto& node : m_nodes) {
        drawNode(painter, node);
    }
}

void GraphPanel::drawNode(QPainter& painter, const GraphNode& node) {
    QColor color = node.color;
    if (m_ready) {
        // Make it brighter when model is ready.
        painter.setPen(QPen(Qt::darkYellow, 3));
    } else {
        painter.setPen(QPen(QColor(0x33, 0x66, 0x99), 2));
    }

    painter.setBrush(color);
    painter.drawEllipse(node.pos, node.radius, node.radius);

    painter.setPen(QPen(Qt::darkCyan));
    painter.setFont(QFont("Arial", 9));

    QFontMetrics fm(painter.font());
    int textWidth = fm.horizontalAdvance(node.label) + 10;
    QRect textRect(node.pos.x() - textWidth/2,
                   node.pos.y() + node.radius + 4,
                   textWidth, 20);
    painter.drawText(textRect, Qt::AlignCenter, node.label);
}

void GraphPanel::drawEdge(QPainter& painter, const GraphEdge& edge) {
    if (edge.from >= m_nodes.size() || edge.to >= m_nodes.size()) return;

    QPoint fromPos = m_nodes[edge.from].pos + QPoint(30, 0);
    QPoint toPos = m_nodes[edge.to].pos + QPoint(-30, 0);

    if (m_ready) {
        // Make it brighter when model is ready.
        painter.setPen(QPen(Qt::darkYellow, 3));
    } else {
        painter.setPen(QPen(QColor(0x44, 0x66, 0x88), 2));
    }

    painter.drawLine(fromPos, toPos);
}

void GraphPanel::onRefreshTimer() {
    m_pendingRefresh = false;
    update();  // ReDraw
}

void GraphPanel::requestRefresh() {
    if (!m_pendingRefresh) {
        m_pendingRefresh = true;
        m_refreshTimer.start();
    }
}
