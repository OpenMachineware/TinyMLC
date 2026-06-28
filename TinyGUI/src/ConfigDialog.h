#ifndef TINY_GUI_CONFIG_DIALOG_H
#define TINY_GUI_CONFIG_DIALOG_H

#include <QDialog>
#include <QLineEdit>
#include <QComboBox>
#include <QPushButton>

class ConfigDialog : public QDialog {
    Q_OBJECT

public:
    explicit ConfigDialog(QWidget *parent = nullptr);

    void loadConfig();
    void saveConfig();

private slots:
    void onAccept();
    void onBrowsePython();
    void onBrowseScript();
    void onBrowseCmsis();
    void onBrowseNmsis();

private:
    void setupUI();
    QString browseFile(const QString& title, const QString& filter = "");

    QLineEdit* m_pythonPath;
    QLineEdit* m_scriptPath;
    QComboBox* m_targetCombo;
    QComboBox* m_modeCombo;
    QComboBox* m_accelCombo;
    QLineEdit* m_gccArm;
    QLineEdit* m_gccRiscv;
    QLineEdit* m_qemuArm;
    QLineEdit* m_qemuRiscv;
    QLineEdit* m_cmsisPath;
    QLineEdit* m_nmsisPath;
};

#endif
