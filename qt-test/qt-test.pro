#-------------------------------------------------
#
# Project created by QtCreator 2016-06-07T18:10:12
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = qt-test
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    camviewer.cpp

HEADERS  += mainwindow.h \
    camviewer.h

FORMS    += mainwindow.ui

LIBS += -L/usr/local/lib


QT_CONFIG -= no-pkg-config
CONFIG  += link_pkgconfig
PKGCONFIG += opencv
