#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QTimer>
#include <QDebug>

#define CAM_ID 0

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QTimer* timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this,  SLOT(updateView()));
    timer->start(30);

    ui->camViewer->setScene(&scene);

}

MainWindow::~MainWindow()
{
    delete ui;
    if(video.isOpened())
    {
         video.release();
    }
}

void MainWindow::on_pushButton_clicked()
{
    if(video.isOpened())
    {
         video.release();
         ui->pushButton->setText("Start");

    }
    else
    {
        video.open(CAM_ID);
        ui->pushButton->setText("Stop");


    }
}




void MainWindow::updateView()
{
    if(!video.isOpened()) return;
    cv::Mat frame;
    while(1)
    {
        video >> frame;
        if(!frame.empty()) break;
    }
    if(frame.empty()) return;
    ui->camViewer->setImage(QImage((const unsigned char*)(frame.data), frame.cols,frame.rows,QImage::Format_RGB888).rgbSwapped());


}
