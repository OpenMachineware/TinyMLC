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
