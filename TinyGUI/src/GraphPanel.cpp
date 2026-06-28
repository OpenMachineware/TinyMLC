#include "GraphPanel.h"

#include <QPainter>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>
#include <QDebug>

GraphPanel::GraphPanel(QWidget *parent)
    : QWidget(parent), m_ready(false) {
    setMinimumHeight(150);
    setStyleSheet("background-color: #252525;");
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
    parseOps(ops);
    layoutNodes();
    update();
}

void GraphPanel::parseOps(const QJsonArray& ops) {
    m_nodes.clear();
    m_edges.clear();

    // ---- Create nodes ----
    for (int i = 0; i < ops.size(); ++i) {
        QJsonObject op = ops[i].toObject();
        QString opName = op["op_name"].toString();
        GraphNode node;
        node.label = opName;
        node.radius = 30;
        node.color = QColor(0x00, 0x99, 0xff);
        node.ready = false;
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

void GraphPanel::layoutNodes() {
    if (m_nodes.isEmpty()) return;

    int width = this->width();
    int height = this->height();
    int spacing = 120;
    int totalWidth = (m_nodes.size() - 1) * spacing;
    int startX = (width - totalWidth) / 2;
    int y = height / 2;

    for (int i = 0; i < m_nodes.size(); ++i) {
        m_nodes[i].pos = QPoint(startX + i * spacing, y);
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
