#include "ConfigPanel.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QSpacerItem>

ConfigPanel::ConfigPanel(QWidget *parent)
    : QWidget(parent) {
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(8);

    // ---- Hardware Constraints ----
    QGroupBox* hwGroup = new QGroupBox(tr("Hardware Constraints"), this);
    QVBoxLayout* hwLayout = new QVBoxLayout(hwGroup);

    // MACs
    QHBoxLayout* macsLayout = new QHBoxLayout();
    macsLayout->addWidget(new QLabel(tr("MACs:"), this));
    m_macsSpin = new QSpinBox(this);
    m_macsSpin->setRange(1000, 10000000);
    m_macsSpin->setValue(100000);
    m_macsSpin->setSingleStep(1000);
    macsLayout->addWidget(m_macsSpin);
    macsLayout->addStretch();
    hwLayout->addLayout(macsLayout);

    // RAM
    QHBoxLayout* ramLayout = new QHBoxLayout();
    ramLayout->addWidget(new QLabel(tr("RAM (KB):"), this));
    m_ramSpin = new QSpinBox(this);
    m_ramSpin->setRange(1, 1024);
    m_ramSpin->setValue(30);
    ramLayout->addWidget(m_ramSpin);
    ramLayout->addStretch();
    hwLayout->addLayout(ramLayout);

    // Flash
    QHBoxLayout* flashLayout = new QHBoxLayout();
    flashLayout->addWidget(new QLabel(tr("Flash (KB):"), this));
    m_flashSpin = new QSpinBox(this);
    m_flashSpin->setRange(1, 4096);
    m_flashSpin->setValue(64);
    flashLayout->addWidget(m_flashSpin);
    flashLayout->addStretch();
    hwLayout->addLayout(flashLayout);

    mainLayout->addWidget(hwGroup);

    // ---- Generate Options ----
    QGroupBox* generateGroup = new QGroupBox(tr("Generate Options"), this);
    QVBoxLayout* generateLayout = new QVBoxLayout(generateGroup);

    // Generations
    QHBoxLayout* gensLayout = new QHBoxLayout();
    gensLayout->addWidget(new QLabel(tr("Generations:"), this));
    m_gensSpin = new QSpinBox(this);
    m_gensSpin->setRange(1, 1000);
    m_gensSpin->setValue(50);
    gensLayout->addWidget(m_gensSpin);
    gensLayout->addStretch();
    generateLayout->addLayout(gensLayout);

    // Population
    QHBoxLayout* popLayout = new QHBoxLayout();
    popLayout->addWidget(new QLabel(tr("Population:"), this));
    m_popSpin = new QSpinBox(this);
    m_popSpin->setRange(1, 500);
    m_popSpin->setValue(50);
    popLayout->addWidget(m_popSpin);
    popLayout->addStretch();
    generateLayout->addLayout(popLayout);

    mainLayout->addWidget(generateGroup);
    mainLayout->addStretch();

    setMinimumWidth(200);
}

int ConfigPanel::getMaxMacs() const { return m_macsSpin->value(); }
int ConfigPanel::getMaxRam() const { return m_ramSpin->value(); }
int ConfigPanel::getMaxFlash() const { return m_flashSpin->value(); }
int ConfigPanel::getGenerations() const { return m_gensSpin->value(); }
int ConfigPanel::getPopulation() const { return m_popSpin->value(); }
