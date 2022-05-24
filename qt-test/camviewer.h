#ifndef CAMVIEWER_H
#define CAMVIEWER_H

#include <QGraphicsView>


class CamViewer : public QGraphicsView
{

public :
    CamViewer(QWidget* centralWidget) : QGraphicsView(centralWidget) {}
    CamViewer(QGraphicsScene * scene, QWidget * parent = 0) : QGraphicsView(scene, parent){}
    ~CamViewer() {}
    void setImage(QImage im);

protected :
    void paintEvent(QPaintEvent* paintEventInfo);

private :
    QImage image;
};

#endif // CAMVIEWER_H
