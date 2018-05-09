#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace bv
{

	CV_EXPORTS_W void fillHoles(Mat &mat);

	class CV_EXPORTS_W Filters 
	{
	public:
		CV_WRAP Filters();
		CV_WRAP void edge(InputArray im, OutputArray imedge);
	};
}
