#include <windows.h>
#include <windowsx.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <list>
#include "../../svm.h"
using namespace std;

#define DEFAULT_PARAM "-t 2 -c 100"
#define XLEN 500
#define YLEN 500
#define DrawLine(dc,x1,y1,x2,y2,c) \
	do { \
		HPEN hpen = CreatePen(PS_SOLID,0,c); \
		HPEN horig = SelectPen(dc,hpen); \
		MoveToEx(dc,x1,y1,NULL); \
		LineTo(dc,x2,y2); \
		SelectPen(dc,horig); \
		DeletePen(hpen); \
	} while(0)

using namespace std;

COLORREF colors[] =
{
	RGB(0,0,0),
	RGB(0,120,120),
	RGB(120,120,0),
	RGB(120,0,120),
	RGB(0,200,200),
	RGB(200,200,0),
	RGB(200,0,200)
};

HWND main_window;
HBITMAP buffer;
HDC window_dc;
HDC buffer_dc;
HBRUSH brush1, brush2, brush3;
HWND edit;

enum {
	ID_BUTTON_CHANGE, ID_BUTTON_RUN, ID_BUTTON_CLEAR,
	ID_BUTTON_LOAD, ID_BUTTON_SAVE, ID_EDIT
};

struct point {
	double x, y;
	signed char value;
};

list<point> point_list;
int current_value = 1;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
		   PSTR szCmdLine, int iCmdShow)
{
	static char szAppName[] = "SvmToy";
	MSG msg;
	WNDCLASSEX wndclass;

	wndclass.cbSize = sizeof(wndclass);
	wndclass.style = CS_HREDRAW | CS_VREDRAW;
	wndclass.lpfnWndProc = WndProc;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH) GetStockObject(BLACK_BRUSH);
	wndclass.lpszMenuName = NULL;
	wndclass.lpszClassName = szAppName;
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

	RegisterClassEx(&wndclass);

	main_window = CreateWindow(szAppName,	// window class name
				    "SVM Toy",	// window caption
				    WS_OVERLAPPEDWINDOW,// window style
				    CW_USEDEFAULT,	// initial x position
				    CW_USEDEFAULT,	// initial y position
				    XLEN,	// initial x size
				    YLEN+52,	// initial y size
				    NULL,	// parent window handle
				    NULL,	// window menu handle
				    hInstance,	// program instance handle
				    NULL);	// creation parameters

	ShowWindow(main_window, iCmdShow);
	UpdateWindow(main_window);

	CreateWindow("button", "Change", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
		     0, YLEN, 50, 25, main_window, (HMENU) ID_BUTTON_CHANGE, hInstance, NULL);
	CreateWindow("button", "Run", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
		     50, YLEN, 50, 25, main_window, (HMENU) ID_BUTTON_RUN, hInstance, NULL);
	CreateWindow("button", "Clear", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
		     100, YLEN, 50, 25, main_window, (HMENU) ID_BUTTON_CLEAR, hInstance, NULL);
	CreateWindow("button", "Save", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
		     150, YLEN, 50, 25, main_window, (HMENU) ID_BUTTON_SAVE, hInstance, NULL);
	CreateWindow("button", "Load", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
		     200, YLEN, 50, 25, main_window, (HMENU) ID_BUTTON_LOAD, hInstance, NULL);

	edit = CreateWindow("edit", NULL, WS_CHILD | WS_VISIBLE,
		            250, YLEN, 250, 25, main_window, (HMENU) ID_EDIT, hInstance, NULL);

	Edit_SetText(edit,DEFAULT_PARAM);

	brush1 = CreateSolidBrush(colors[4]);
	brush2 = CreateSolidBrush(colors[5]);
	brush3 = CreateSolidBrush(colors[6]);

	window_dc = GetDC(main_window);
	buffer = CreateCompatibleBitmap(window_dc, XLEN, YLEN);
	buffer_dc = CreateCompatibleDC(window_dc);
	SelectObject(buffer_dc, buffer);
	PatBlt(buffer_dc, 0, 0, XLEN, YLEN, BLACKNESS);

	while (GetMessage(&msg, NULL, 0, 0)) {
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
	return msg.wParam;
}

int getfilename( HWND hWnd , char *filename, int len, int save) 
{ 	
	OPENFILENAME OpenFileName; 	
	memset(&OpenFileName,0,sizeof(OpenFileName));
	filename[0]='\0';
 	
	OpenFileName.lStructSize       = sizeof(OPENFILENAME); 
	OpenFileName.hwndOwner         = hWnd; 	
	OpenFileName.lpstrFile         = filename; 
	OpenFileName.nMaxFile          = len; 
	OpenFileName.Flags             = 0;
 
	return save?GetSaveFileName(&OpenFileName):GetOpenFileName(&OpenFileName);		
}

void clear_all()
{
	point_list.clear();
	PatBlt(buffer_dc, 0, 0, XLEN, YLEN, BLACKNESS);
	InvalidateRect(main_window, 0, 0);
}

HBRUSH choose_brush(int v)
{
	if(v==1) return brush1;
	else if(v==2) return brush2;
	else return brush3;
}

void draw_point(const point & p)
{
	RECT rect;
	rect.left = int(p.x*XLEN);
	rect.top = int(p.y*YLEN);
	rect.right = int(p.x*XLEN) + 3;
	rect.bottom = int(p.y*YLEN) + 3;
	FillRect(window_dc, &rect, choose_brush(p.value));
	FillRect(buffer_dc, &rect, choose_brush(p.value));
}

void draw_all_points()
{
	for(list<point>::iterator p = point_list.begin(); p != point_list.end(); p++)
		draw_point(*p);
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
	char str[1024];
	Edit_GetLine(edit, 0, str, sizeof(str));
	const char *p = str;

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
		for (list<point>::iterator q = point_list.begin(); q != point_list.end(); q++, i++)
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
		
		DrawLine(buffer_dc,0,0,0,YLEN,colors[0]);
		DrawLine(window_dc,0,0,0,YLEN,colors[0]);
		
		int p = (int)(param.p * YLEN);
		for(int i=1; i < XLEN; i++)
		{
			DrawLine(buffer_dc,i,0,i,YLEN,colors[0]);
			DrawLine(window_dc,i,0,i,YLEN,colors[0]);
			
			DrawLine(buffer_dc,i-1,j[i-1],i,j[i],colors[5]);
			DrawLine(window_dc,i-1,j[i-1],i,j[i],colors[5]);

			if(param.svm_type == EPSILON_SVR)
			{			
				DrawLine(buffer_dc,i-1,j[i-1]+p,i,j[i]+p,colors[2]);
				DrawLine(window_dc,i-1,j[i-1]+p,i,j[i]+p,colors[2]);

				DrawLine(buffer_dc,i-1,j[i-1]-p,i,j[i]-p,colors[2]);
				DrawLine(window_dc,i-1,j[i-1]-p,i,j[i]-p,colors[2]);
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
		for (list<point>::iterator q = point_list.begin(); q != point_list.end(); q++, i++)
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
			for (j = 0; j < YLEN; j++) {
				x[0].value = (double) i / XLEN;
				x[1].value = (double) j / YLEN;
				double d = svm_predict(model, x);
				if (param.svm_type == ONE_CLASS && d<0) d=2;
				SetPixel(window_dc, i, j, colors[(int)d]);
				SetPixel(buffer_dc, i, j, colors[(int)d]);
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

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	HDC hdc;
	PAINTSTRUCT ps;

	switch (iMsg) {
	case WM_LBUTTONDOWN:
		{
			int x = LOWORD(lParam);
			int y = HIWORD(lParam);
			point p = {(double)x/XLEN, (double)y/YLEN, current_value};
			point_list.push_back(p);
			draw_point(p);
		}
		return 0;
	case WM_PAINT:
		{
			hdc = BeginPaint(hwnd, &ps);
			BitBlt(hdc, 0, 0, XLEN, YLEN, buffer_dc, 0, 0, SRCCOPY);
			EndPaint(hwnd, &ps);
		}
		return 0;
	case WM_COMMAND:
		{
			int id = LOWORD(wParam);
			switch (id) {
			case ID_BUTTON_CHANGE:
				++current_value;
				if(current_value > 3) current_value = 1;
				break;
			case ID_BUTTON_RUN:
				button_run_clicked();
				break;
			case ID_BUTTON_CLEAR:
				clear_all();				
				break;
			case ID_BUTTON_SAVE:
				{
					char filename[1024];
					if(getfilename(hwnd,filename,1024,1))
					{
						FILE *fp = fopen(filename,"w");

						char str[1024];
						Edit_GetLine(edit, 0, str, sizeof(str));
						const char *p = str;
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
				break;
			case ID_BUTTON_LOAD:
				{
					char filename[1024];
					if(getfilename(hwnd,filename,1024,0))					
					{
						FILE *fp = fopen(filename,"r");
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
				break;
			}
		}
		return 0;
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	}

	return DefWindowProc(hwnd, iMsg, wParam, lParam);
}
