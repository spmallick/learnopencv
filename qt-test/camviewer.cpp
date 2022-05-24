#include "camviewer.h"

void CamViewer::setImage(QImage im)
{
    image = im;
    this->scene()->update();

}

void CamViewer::paintEvent(QPaintEvent* evnt)
{
     QPainter painter(viewport());
     painter.drawImage(QRect(0,0,viewport()->width(),viewport()->height()), image, QRect(0,0, image.width(), image.height()) );
     QGraphicsView::paintEvent(evnt);
}
