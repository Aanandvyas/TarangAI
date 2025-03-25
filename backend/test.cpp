#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    Mat a = imread("E:\\Testing\\sample.jpg");

    if (a.empty()) {
        cerr << "Error: Could not open or find the image!" << endl;
        return -1;
    }

    imshow("Sample Image", a);

    waitKey(0);

    return 0;
}