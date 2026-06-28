#ifndef TINY_GUI_GRAPH_PANEL_H
#define TINY_GUI_GRAPH_PANEL_H

#include <QWidget>
#include <QVector>
#include <QString>
#include <QPoint>
#include <QColor>


struct GraphNode {
    QString label;
    QPoint pos;
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
    void parseOps(const QJsonArray& ops);

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    void layoutNodes();
    void drawNode(QPainter& painter, const GraphNode& node);
    void drawEdge(QPainter& painter, const GraphEdge& edge);

    QVector<GraphNode> m_nodes;
    QVector<GraphEdge> m_edges;
    bool m_ready;
};

#endif  // TINY_GUI_GRAPH_PANEL_H
