#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>

#include "mex.h"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif 
#endif 
#ifndef max
#define max(x,y) (((x)>(y))?(x):(y))
#endif
#ifndef min
#define min(x,y) (((x)<(y))?(x):(y))
#endif

void exit_with_help()
{
	mexPrintf(
	"Usage: [label_vector, instance_matrix] = libsvmread('filename');\n"
	);
}

static void fake_answer(int nlhs, mxArray *plhs[])
{
	int i;
	for(i=0;i<nlhs;i++)
		plhs[i] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

static char *line;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line, max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

// read in a problem (in libsvm format)
void read_problem(const char *filename, int nlhs, mxArray *plhs[])
{
	int max_index, min_index, inst_max_index;
	size_t elements, k, i, l=0;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	mwIndex *ir, *jc;
	double *labels, *samples;

	if(fp == NULL)
	{
		mexPrintf("can't open input file %s\n",filename);
		fake_answer(nlhs, plhs);
		return;
	}

	max_line_len = 1024;
	line = (char *) malloc(max_line_len*sizeof(char));

	max_index = 0;
	min_index = 1; // our index starts from 1
	elements = 0;
	while(readline(fp) != NULL)
	{
		char *idx, *val;
		// features
		int index = 0;

		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		strtok(line," \t"); // label
		while (1)
		{
			idx = strtok(NULL,":"); // index:value
			val = strtok(NULL," \t");
			if(val == NULL)
				break;

			errno = 0;
			index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || index <= inst_max_index)
			{
				mexPrintf("Wrong input format at line %d\n",l+1);
				fake_answer(nlhs, plhs);
				return;
			}
			else
				inst_max_index = index;

			min_index = min(min_index, index);
			elements++;
		}
		max_index = max(max_index, inst_max_index);
		l++;
	}
	rewind(fp);

	// y
	plhs[0] = mxCreateDoubleMatrix(l, 1, mxREAL);
	// x^T
	if (min_index <= 0)
		plhs[1] = mxCreateSparse(max_index-min_index+1, l, elements, mxREAL);
	else
		plhs[1] = mxCreateSparse(max_index, l, elements, mxREAL);

	labels = mxGetPr(plhs[0]);
	samples = mxGetPr(plhs[1]);
	ir = mxGetIr(plhs[1]);
	jc = mxGetJc(plhs[1]);

	k=0;
	for(i=0;i<l;i++)
	{
		char *idx, *val, *label;
		jc[i] = k;

		readline(fp);

		label = strtok(line," \t\n");
		if(label == NULL)
		{
			mexPrintf("Empty line at line %d\n",i+1);
			fake_answer(nlhs, plhs);
			return;
		}
		labels[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
		{
			mexPrintf("Wrong input format at line %d\n",i+1);
			fake_answer(nlhs, plhs);
			return;
		}

		// features
		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");
			if(val == NULL)
				break;

			ir[k] = (mwIndex) (strtol(idx,&endptr,10) - min_index); // precomputed kernel has <index> start from 0

			errno = 0;
			samples[k] = strtod(val,&endptr);
			if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
			{
				mexPrintf("Wrong input format at line %d\n",i+1);
				fake_answer(nlhs, plhs);
				return;
			}
			++k;
		}
	}
	jc[l] = k;

	fclose(fp);
	free(line);

	{
		mxArray *rhs[1], *lhs[1];
		rhs[0] = plhs[1];
		if(mexCallMATLAB(1, lhs, 1, rhs, "transpose"))
		{
			mexPrintf("Error: cannot transpose problem\n");
			fake_answer(nlhs, plhs);
			return;
		}
		plhs[1] = lhs[0];
	}
}

void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
	char filename[256];

	if(nrhs != 1 || nlhs != 2)
	{
		exit_with_help();
		fake_answer(nlhs, plhs);
		return;
	}

	mxGetString(prhs[0], filename, mxGetN(prhs[0]) + 1);

	if(filename == NULL)
	{
		mexPrintf("Error: filename is NULL\n");
		return;
	}

	read_problem(filename, nlhs, plhs);

	return;
}

