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

#ifndef TINY_GUI_GRAPH_PANEL_H
#define TINY_GUI_GRAPH_PANEL_H

#include <QWidget>
#include <QVector>
#include <QString>
#include <QPoint>
#include <QColor>
#include <QTimer>


struct GraphNode {
    QString label;
    QPoint pos;
    QPoint targetPos;
    int radius;
    QColor color;
    bool ready;
};

struct GraphEdge {
    int from;
    int to;
};

class GraphPanel : public QWidget {
    Q_OBJECT

public:
    explicit GraphPanel(QWidget *parent = nullptr);

    void setReady(bool ready);
    void loadModelInfo(const QString& jsonPath);
    void loadModelInfoFromJson(const QString& jsonStr);

protected:
    void paintEvent(QPaintEvent *event) override;

private slots:
    void onRefreshTimer();
    void onAnimTimer();

private:
    void layoutNodes();
    void drawNode(QPainter& painter, const GraphNode& node);
    void drawEdge(QPainter& painter, const GraphEdge& edge);
    void parseOps(const QJsonArray& ops);
    void requestRefresh();

    QVector<GraphNode> m_nodes;
    QVector<GraphEdge> m_edges;
    bool m_ready;
    QTimer m_refreshTimer;
    bool m_pendingRefresh;
    QTimer m_animTimer;
    bool m_animating;
};

#endif  // TINY_GUI_GRAPH_PANEL_H
