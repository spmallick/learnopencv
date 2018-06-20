#include <QtGui>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <list>
#include "../../svm.h"
using namespace std;

#define DEFAULT_PARAM "-t 2 -c 100"
#define XLEN 500
#define YLEN 500

QRgb colors[] =
{
	qRgb(0,0,0),
	qRgb(0,120,120),
	qRgb(120,120,0),
	qRgb(120,0,120),
	qRgb(0,200,200),
	qRgb(200,200,0),
	qRgb(200,0,200)
};

class SvmToyWindow : public QWidget
{

Q_OBJECT

public:
	SvmToyWindow();
	~SvmToyWindow();
protected:
	virtual void mousePressEvent( QMouseEvent* );
	virtual void paintEvent( QPaintEvent* );

private:
	QPixmap buffer;
	QPixmap icon1;
	QPixmap icon2;
	QPixmap icon3;
	QPushButton button_change_icon;
	QPushButton button_run;
	QPushButton button_clear;
	QPushButton button_save;
	QPushButton button_load;
	QLineEdit input_line;
	QPainter buffer_painter;
	struct point {
		double x, y;
		signed char value;
	};
	list<point> point_list;
	int current_value;
	const QPixmap& choose_icon(int v)
	{
		if(v==1) return icon1;
		else if(v==2) return icon2;
		else return icon3;
	}
	void clear_all()
	{
		point_list.clear();
		buffer.fill(Qt::black);
		repaint();
	}
	void draw_point(const point& p)
	{
		const QPixmap& icon = choose_icon(p.value);
		buffer_painter.drawPixmap((int)(p.x*XLEN),(int)(p.y*YLEN),icon);
		repaint();
	}
	void draw_all_points()
	{
		for(list<point>::iterator p = point_list.begin(); p != point_list.end();p++)
			draw_point(*p);	
	}
private slots: 
	void button_change_icon_clicked()
	{
		++current_value;
		if(current_value > 3) current_value = 1;
		button_change_icon.setIcon(choose_icon(current_value));
	}
	void button_run_clicked()
	{
		// guard
		if(point_list.empty()) return;

		svm_parameter param;
		int i,j;	

		// default values
		param.svm_type = C_SVC;
		param.kernel_type = RBF;
		param.degree = 3;
		param.gamma = 0;
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 100;
		param.C = 1;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		param.probability = 0;
		param.nr_weight = 0;
		param.weight_label = NULL;
		param.weight = NULL;

		// parse options
		const char *p = input_line.text().toAscii().constData();

		while (1) {
			while (*p && *p != '-')
				p++;

			if (*p == '\0')
				break;

			p++;
			switch (*p++) {
				case 's':
					param.svm_type = atoi(p);
					break;
				case 't':
					param.kernel_type = atoi(p);
					break;
				case 'd':
					param.degree = atoi(p);
					break;
				case 'g':
					param.gamma = atof(p);
					break;
				case 'r':
					param.coef0 = atof(p);
					break;
				case 'n':
					param.nu = atof(p);
					break;
				case 'm':
					param.cache_size = atof(p);
					break;
				case 'c':
					param.C = atof(p);
					break;
				case 'e':
					param.eps = atof(p);
					break;
				case 'p':
					param.p = atof(p);
					break;
				case 'h':
					param.shrinking = atoi(p);
					break;
			        case 'b':
					param.probability = atoi(p);
					break;
				case 'w':
					++param.nr_weight;
					param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
					param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
					param.weight_label[param.nr_weight-1] = atoi(p);
					while(*p && !isspace(*p)) ++p;
					param.weight[param.nr_weight-1] = atof(p);
					break;
			}
		}
	
		// build problem
		svm_problem prob;

		prob.l = point_list.size();
		prob.y = new double[prob.l];

		if(param.kernel_type == PRECOMPUTED)
		{
		}
		else if(param.svm_type == EPSILON_SVR ||
			param.svm_type == NU_SVR)
		{
			if(param.gamma == 0) param.gamma = 1;
			svm_node *x_space = new svm_node[2 * prob.l];
			prob.x = new svm_node *[prob.l];

			i = 0;
			for (list <point>::iterator q = point_list.begin(); q != point_list.end(); q++, i++)
			{
				x_space[2 * i].index = 1;
				x_space[2 * i].value = q->x;
				x_space[2 * i + 1].index = -1;
				prob.x[i] = &x_space[2 * i];
				prob.y[i] = q->y;
			}

			// build model & classify
			svm_model *model = svm_train(&prob, &param);
			svm_node x[2];
			x[0].index = 1;
			x[1].index = -1;
			int *j = new int[XLEN];

			for (i = 0; i < XLEN; i++)
			{
				x[0].value = (double) i / XLEN;
				j[i] = (int)(YLEN*svm_predict(model, x));
			}
			
			buffer_painter.setPen(colors[0]);
			buffer_painter.drawLine(0,0,0,YLEN-1);

			int p = (int)(param.p * YLEN);
			for(i = 1; i < XLEN; i++)
			{
				buffer_painter.setPen(colors[0]);
				buffer_painter.drawLine(i,0,i,YLEN-1);
			
				buffer_painter.setPen(colors[5]);
				buffer_painter.drawLine(i-1,j[i-1],i,j[i]);
				
				if(param.svm_type == EPSILON_SVR)
				{
					buffer_painter.setPen(colors[2]);
					buffer_painter.drawLine(i-1,j[i-1]+p,i,j[i]+p);

					buffer_painter.setPen(colors[2]);
					buffer_painter.drawLine(i-1,j[i-1]-p,i,j[i]-p);
				}
			}

			svm_free_and_destroy_model(&model);
			delete[] j;
			delete[] x_space;
			delete[] prob.x;
			delete[] prob.y;
		}
		else
		{
			if(param.gamma == 0) param.gamma = 0.5;
			svm_node *x_space = new svm_node[3 * prob.l];
			prob.x = new svm_node *[prob.l];

			i = 0;
			for (list <point>::iterator q = point_list.begin(); q != point_list.end(); q++, i++)
			{
				x_space[3 * i].index = 1;
				x_space[3 * i].value = q->x;
				x_space[3 * i + 1].index = 2;
				x_space[3 * i + 1].value = q->y;
				x_space[3 * i + 2].index = -1;
				prob.x[i] = &x_space[3 * i];
				prob.y[i] = q->value;
			}

			// build model & classify
			svm_model *model = svm_train(&prob, &param);
			svm_node x[3];
			x[0].index = 1;
			x[1].index = 2;
			x[2].index = -1;

			for (i = 0; i < XLEN; i++)
				for (j = 0; j < YLEN ; j++) {
					x[0].value = (double) i / XLEN;
					x[1].value = (double) j / YLEN;
					double d = svm_predict(model, x);
					if (param.svm_type == ONE_CLASS && d<0) d=2;
					buffer_painter.setPen(colors[(int)d]);
					buffer_painter.drawPoint(i,j);
			}

			svm_free_and_destroy_model(&model);
			delete[] x_space;
			delete[] prob.x;
			delete[] prob.y;
		}
		free(param.weight_label);
		free(param.weight);
		draw_all_points();
	}
	void button_clear_clicked()
	{
		clear_all();
	}
	void button_save_clicked()
	{
		QString filename = QFileDialog::getSaveFileName();
		if(!filename.isNull())
		{
			FILE *fp = fopen(filename.toAscii().constData(),"w");
			
			const char *p = input_line.text().toAscii().constData();
			const char* svm_type_str = strstr(p, "-s ");
			int svm_type = C_SVC;
			if(svm_type_str != NULL)
				sscanf(svm_type_str, "-s %d", &svm_type);
		
			if(fp)
			{
				if(svm_type == EPSILON_SVR || svm_type == NU_SVR)
				{
					for(list<point>::iterator p = point_list.begin(); p != point_list.end();p++)
						fprintf(fp,"%f 1:%f\n", p->y, p->x);
				}
				else
				{
					for(list<point>::iterator p = point_list.begin(); p != point_list.end();p++)
						fprintf(fp,"%d 1:%f 2:%f\n", p->value, p->x, p->y);
				}
				fclose(fp);
			}
		}
	}
	void button_load_clicked()
	{
		QString filename = QFileDialog::getOpenFileName();
		if(!filename.isNull())
		{
			FILE *fp = fopen(filename.toAscii().constData(),"r");
			if(fp)
			{
				clear_all();
				char buf[4096];
				while(fgets(buf,sizeof(buf),fp))
				{
					int v;
					double x,y;
					if(sscanf(buf,"%d%*d:%lf%*d:%lf",&v,&x,&y)==3)
					{
						point p = {x,y,v};
						point_list.push_back(p);
					}
					else if(sscanf(buf,"%lf%*d:%lf",&y,&x)==2)
					{
						point p = {x,y,current_value};
						point_list.push_back(p);
					}
					else
						break;
				}
				fclose(fp);
				draw_all_points();
			}				
		}
		
	}
};

#include "svm-toy.moc"

SvmToyWindow::SvmToyWindow()
:button_change_icon(this)
,button_run("Run",this)
,button_clear("Clear",this)
,button_save("Save",this)
,button_load("Load",this)
,input_line(this)
,current_value(1)
{
	buffer = QPixmap(XLEN,YLEN);
	buffer.fill(Qt::black);

	buffer_painter.begin(&buffer);

	QObject::connect(&button_change_icon, SIGNAL(clicked()), this,
			 SLOT(button_change_icon_clicked()));
	QObject::connect(&button_run, SIGNAL(clicked()), this,
			 SLOT(button_run_clicked()));
	QObject::connect(&button_clear, SIGNAL(clicked()), this,
			 SLOT(button_clear_clicked()));
	QObject::connect(&button_save, SIGNAL(clicked()), this,
			 SLOT(button_save_clicked()));
	QObject::connect(&button_load, SIGNAL(clicked()), this,
			 SLOT(button_load_clicked()));
	QObject::connect(&input_line, SIGNAL(returnPressed()), this,
			 SLOT(button_run_clicked()));

  	// don't blank the window before repainting
	setAttribute(Qt::WA_NoBackground);
  
	icon1 = QPixmap(4,4);
	icon2 = QPixmap(4,4);
	icon3 = QPixmap(4,4);
	
	
	QPainter painter;
	painter.begin(&icon1);
	painter.fillRect(0,0,4,4,QBrush(colors[4]));
	painter.end();

	painter.begin(&icon2);
	painter.fillRect(0,0,4,4,QBrush(colors[5]));
	painter.end();

	painter.begin(&icon3);
	painter.fillRect(0,0,4,4,QBrush(colors[6]));
	painter.end();

	button_change_icon.setGeometry( 0, YLEN, 50, 25 );
	button_run.setGeometry( 50, YLEN, 50, 25 );
	button_clear.setGeometry( 100, YLEN, 50, 25 );
	button_save.setGeometry( 150, YLEN, 50, 25);
	button_load.setGeometry( 200, YLEN, 50, 25);
	input_line.setGeometry( 250, YLEN, 250, 25);
	
	input_line.setText(DEFAULT_PARAM);
	button_change_icon.setIcon(icon1);
}

SvmToyWindow::~SvmToyWindow()
{
	buffer_painter.end();
}

void SvmToyWindow::mousePressEvent( QMouseEvent* event )
{
	point p = {(double)event->x()/XLEN, (double)event->y()/YLEN, current_value};
	point_list.push_back(p);
	draw_point(p);
}

void SvmToyWindow::paintEvent( QPaintEvent* )
{
	// copy the image from the buffer pixmap to the window
	QPainter p(this);
	p.drawPixmap(0, 0, buffer);
}

int main( int argc, char* argv[] )
{
	QApplication myapp( argc, argv );

	SvmToyWindow* mywidget = new SvmToyWindow();
	mywidget->setGeometry( 100, 100, XLEN, YLEN+25 );

	mywidget->show();
	return myapp.exec();
}
