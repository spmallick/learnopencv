#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "camviewer.h"
#include <opencv2/highgui/highgui.hpp>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    bool isCameraOn;
    ~MainWindow();

private slots:
    void on_pushButton_clicked();
    void updateView(void);

private:
    Ui::MainWindow *ui;
    QGraphicsScene scene;
    cv::VideoCapture video;
    cv::Mat frame;

};

#endif // MAINWINDOW_H
