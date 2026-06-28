#include "ConfigDialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QJsonDocument>
#include <QJsonObject>
#include <QFile>
#include <QDir>

static const QString CONFIG_PATH = QDir::homePath() + "/.tinymlc/config.json";

ConfigDialog::ConfigDialog(QWidget *parent)
    : QDialog(parent) {
    setupUI();
    loadConfig();
}

void ConfigDialog::setupUI() {
    setWindowTitle(tr("Settings"));
    setMinimumWidth(500);

    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    QFormLayout* formLayout = new QFormLayout();

    // ---- Python ----
    QHBoxLayout* pythonLayout = new QHBoxLayout();
    m_pythonPath = new QLineEdit(this);
    QPushButton* pythonBrowse = new QPushButton(tr("Browse..."), this);
    connect(pythonBrowse, &QPushButton::clicked, this, &ConfigDialog::onBrowsePython);
    pythonLayout->addWidget(m_pythonPath);
    pythonLayout->addWidget(pythonBrowse);
    formLayout->addRow(tr("Python Path:"), pythonLayout);

    // ---- Script ----
    QHBoxLayout* scriptLayout = new QHBoxLayout();
    m_scriptPath = new QLineEdit(this);
    QPushButton* scriptBrowse = new QPushButton(tr("Browse..."), this);
    connect(scriptBrowse, &QPushButton::clicked, this, &ConfigDialog::onBrowseScript);
    scriptLayout->addWidget(m_scriptPath);
    scriptLayout->addWidget(scriptBrowse);
    formLayout->addRow(tr("main.py Path:"), scriptLayout);

    // ---- Target ----
    m_targetCombo = new QComboBox(this);
    m_targetCombo->addItems({"riscv", "arm", "host"});
    formLayout->addRow(tr("Default Target:"), m_targetCombo);

    // ---- Mode ----
    m_modeCombo = new QComboBox(this);
    m_modeCombo->addItems({"debug", "release"});
    formLayout->addRow(tr("Default Mode:"), m_modeCombo);

    // ---- Accel ----
    m_accelCombo = new QComboBox(this);
    m_accelCombo->addItems({"pure-c", "cmsis-nn", "nmsis-nn"});
    formLayout->addRow(tr("Default Accel:"), m_accelCombo);

    // ---- GCC ARM ----
    m_gccArm = new QLineEdit(this);
    m_gccArm->setText("arm-none-eabi-gcc");
    formLayout->addRow(tr("ARM GCC:"), m_gccArm);

    // ---- GCC RISC-V ----
    m_gccRiscv = new QLineEdit(this);
    m_gccRiscv->setText("riscv-none-elf-gcc");
    formLayout->addRow(tr("RISC-V GCC:"), m_gccRiscv);

    // ---- QEMU ARM ----
    m_qemuArm = new QLineEdit(this);
    m_qemuArm->setText("qemu-system-arm");
    formLayout->addRow(tr("QEMU ARM:"), m_qemuArm);

    // ---- QEMU RISC-V ----
    m_qemuRiscv = new QLineEdit(this);
    m_qemuRiscv->setText("qemu-system-riscv32");
    formLayout->addRow(tr("QEMU RISC-V:"), m_qemuRiscv);

    // ---- CMSIS-NN ----
    QHBoxLayout* cmsisLayout = new QHBoxLayout();
    m_cmsisPath = new QLineEdit(this);
    QPushButton* cmsisBrowse = new QPushButton(tr("Browse..."), this);
    connect(cmsisBrowse, &QPushButton::clicked, this, &ConfigDialog::onBrowseCmsis);
    cmsisLayout->addWidget(m_cmsisPath);
    cmsisLayout->addWidget(cmsisBrowse);
    formLayout->addRow(tr("CMSIS-NN Path:"), cmsisLayout);

    // ---- NMSIS ----
    QHBoxLayout* nmsisLayout = new QHBoxLayout();
    m_nmsisPath = new QLineEdit(this);
    QPushButton* nmsisBrowse = new QPushButton(tr("Browse..."), this);
    connect(nmsisBrowse, &QPushButton::clicked, this, &ConfigDialog::onBrowseNmsis);
    nmsisLayout->addWidget(m_nmsisPath);
    nmsisLayout->addWidget(nmsisBrowse);
    formLayout->addRow(tr("NMSIS Path:"), nmsisLayout);

    mainLayout->addLayout(formLayout);

    // ---- Buttons ----
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    QPushButton* okButton = new QPushButton(tr("OK"), this);
    QPushButton* cancelButton = new QPushButton(tr("Cancel"), this);
    connect(okButton, &QPushButton::clicked, this, &ConfigDialog::onAccept);
    connect(cancelButton, &QPushButton::clicked, this, &QDialog::reject);
    buttonLayout->addStretch();
    buttonLayout->addWidget(okButton);
    buttonLayout->addWidget(cancelButton);
    mainLayout->addLayout(buttonLayout);
}

void ConfigDialog::loadConfig() {
    QFile file(CONFIG_PATH);
    if (!file.exists()) return;

    if (!file.open(QIODevice::ReadOnly)) return;

    QByteArray data = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (doc.isNull()) return;

    QJsonObject obj = doc.object();
    m_pythonPath->setText(obj["python_path"].toString());
    m_scriptPath->setText(obj["script_path"].toString());
    m_targetCombo->setCurrentText(obj["target"].toString("riscv"));
    m_modeCombo->setCurrentText(obj["mode"].toString("debug"));
    m_accelCombo->setCurrentText(obj["accel"].toString("pure-c"));
    m_gccArm->setText(obj["gcc_arm"].toString("arm-none-eabi-gcc"));
    m_gccRiscv->setText(obj["gcc_riscv"].toString("riscv-none-elf-gcc"));
    m_qemuArm->setText(obj["qemu_arm"].toString("qemu-system-arm"));
    m_qemuRiscv->setText(obj["qemu_riscv"].toString("qemu-system-riscv32"));
    m_cmsisPath->setText(obj["cmsis_nn_path"].toString());
    m_nmsisPath->setText(obj["nmsis_nn_path"].toString());
}

void ConfigDialog::saveConfig() {
    QJsonObject obj;
    obj["python_path"] = m_pythonPath->text();
    obj["script_path"] = m_scriptPath->text();
    obj["target"] = m_targetCombo->currentText();
    obj["mode"] = m_modeCombo->currentText();
    obj["accel"] = m_accelCombo->currentText();
    obj["gcc_arm"] = m_gccArm->text();
    obj["gcc_riscv"] = m_gccRiscv->text();
    obj["qemu_arm"] = m_qemuArm->text();
    obj["qemu_riscv"] = m_qemuRiscv->text();
    obj["cmsis_nn_path"] = m_cmsisPath->text();
    obj["nmsis_nn_path"] = m_nmsisPath->text();

    QJsonDocument doc(obj);

    QDir dir(QDir::homePath() + "/.tinymlc");
    if (!dir.exists()) {
        dir.mkpath(".");
    }

    QFile file(CONFIG_PATH);
    if (!file.open(QIODevice::WriteOnly)) {
        QMessageBox::warning(this, tr("Error"), tr("Failed to save config"));
        return;
    }

    file.write(doc.toJson());
    file.close();
}

void ConfigDialog::onAccept() {
    saveConfig();
    accept();
}

void ConfigDialog::onBrowsePython() {
    QString path = browseFile(tr("Select Python Interpreter"), "python*");
    if (!path.isEmpty()) m_pythonPath->setText(path);
}

void ConfigDialog::onBrowseScript() {
    QString path = browseFile(tr("Select main.py"), "*.py");
    if (!path.isEmpty()) m_scriptPath->setText(path);
}

void ConfigDialog::onBrowseCmsis() {
    QString path = browseFile(tr("Select CMSIS-NN directory"), "");
    if (!path.isEmpty()) m_cmsisPath->setText(path);
}

void ConfigDialog::onBrowseNmsis() {
    QString path = browseFile(tr("Select NMSIS directory"), "");
    if (!path.isEmpty()) m_nmsisPath->setText(path);
}

QString ConfigDialog::browseFile(const QString& title, const QString& filter) {
    return QFileDialog::getExistingDirectory(this, title);
}
