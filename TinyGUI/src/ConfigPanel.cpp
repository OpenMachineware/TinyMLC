#include "ConfigPanel.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QSpacerItem>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>

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

    // ---- Model Info ----
    m_infoGroup = new QGroupBox(tr("Model Information"), this);
    QVBoxLayout* infoLayout = new QVBoxLayout(m_infoGroup);
    m_inputShapeLabel = new QLabel(tr("Input: -"), this);
    infoLayout->addWidget(m_inputShapeLabel);
    m_outputShapeLabel = new QLabel(tr("Output: -"), this);
    infoLayout->addWidget(m_outputShapeLabel);
    m_layersLabel = new QLabel(tr("Layers: -"), this);
    infoLayout->addWidget(m_layersLabel);
    m_macsInfoLabel = new QLabel(tr("MACs: -"), this);
    infoLayout->addWidget(m_macsInfoLabel);
    m_paramsLabel = new QLabel(tr("Params: -"), this);
    infoLayout->addWidget(m_paramsLabel);
    m_ramInfoLabel = new QLabel(tr("Peak RAM: -"), this);
    infoLayout->addWidget(m_ramInfoLabel);
    m_flashInfoLabel = new QLabel(tr("Flash: -"), this);
    infoLayout->addWidget(m_flashInfoLabel);

    mainLayout->addWidget(hwGroup);
    mainLayout->addWidget(m_infoGroup);
    mainLayout->addWidget(generateGroup);
    mainLayout->addStretch();

    setMinimumWidth(200);
}

void ConfigPanel::setModelInfo(const QString& jsonPath) {
    QFile file(jsonPath);
    if (!file.open(QIODevice::ReadOnly)) {
        return;
    }

    QByteArray data = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (doc.isNull()) return;

    QJsonObject root = doc.object();

    // Input/Output
    QJsonArray inputs = root["input"].toArray();
    QJsonArray outputs = root["output"].toArray();

    if (!inputs.isEmpty()) {
        QJsonObject inp = inputs[0].toObject();
        QJsonArray shape = inp["shape"].toArray();
        QString shapeStr;
        for (const auto& v : shape) {
            shapeStr += QString::number(v.toInt()) + ",";
        }
        shapeStr.chop(1);
        m_inputShapeLabel->setText(tr("Input: [%1]").arg(shapeStr));
    }

    if (!outputs.isEmpty()) {
        QJsonObject out = outputs[0].toObject();
        QJsonArray shape = out["shape"].toArray();
        QString shapeStr;
        for (const auto& v : shape) {
            shapeStr += QString::number(v.toInt()) + ",";
        }
        shapeStr.chop(1);
        m_outputShapeLabel->setText(tr("Output: [%1]").arg(shapeStr));
    }

    // Layers
    QJsonArray ops = root["ops"].toArray();
    m_layersLabel->setText(tr("Layers: %1").arg(ops.size()));

    // MACs, Params, RAM, Flash (from quant_scales or metadata)
    QJsonObject quantScales = root["quant_scales"].toObject();
    if (quantScales.contains("macs")) {
        m_macsInfoLabel->setText(tr("MACs: %1")
            .arg(quantScales["macs"].toInt()));
    }
    if (quantScales.contains("params")) {
        m_paramsLabel->setText(tr("Params: %1")
            .arg(quantScales["params"].toInt()));
    }
    if (quantScales.contains("peak_ram")) {
        m_ramInfoLabel->setText(tr("Peak RAM: %1 B")
            .arg(quantScales["peak_ram"].toInt()));
    }
    if (quantScales.contains("flash")) {
        m_flashInfoLabel->setText(tr("Flash: %1 B")
            .arg(quantScales["flash"].toInt()));
    }
}

int ConfigPanel::getMaxMacs() const { return m_macsSpin->value(); }
int ConfigPanel::getMaxRam() const { return m_ramSpin->value(); }
int ConfigPanel::getMaxFlash() const { return m_flashSpin->value(); }
int ConfigPanel::getGenerations() const { return m_gensSpin->value(); }
int ConfigPanel::getPopulation() const { return m_popSpin->value(); }
