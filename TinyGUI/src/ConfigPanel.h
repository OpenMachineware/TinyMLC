#ifndef TINY_GUI_CONFIG_PANEL_H
#define TINY_GUI_CONFIG_PANEL_H

#include <QWidget>
#include <QSpinBox>

class ConfigPanel : public QWidget {
    Q_OBJECT

public:
    explicit ConfigPanel(QWidget *parent = nullptr);

    int getMaxMacs() const;
    int getMaxRam() const;
    int getMaxFlash() const;
    int getGenerations() const;
    int getPopulation() const;

private:
    QSpinBox* m_macsSpin;
    QSpinBox* m_ramSpin;
    QSpinBox* m_flashSpin;
    QSpinBox* m_gensSpin;
    QSpinBox* m_popSpin;
};

#endif
