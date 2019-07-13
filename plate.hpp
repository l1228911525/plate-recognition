#ifndef PLATE_H
#define PLATE_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

enum Color{BLUE, YELLOW, UNKNOW};
class plate
{
private:
    Mat PlateMat;
    Color PlateColor;
    int PlateCols;
    int PlateRows;
    int PlateArea;
protected:
public:
    plate(){}
    plate(Mat src, Color color = UNKNOW, int cols = 0, int rows = 0, int area = 0)
    {
        this->PlateMat = src;
        this->PlateColor = color;
        this->PlateCols = cols;
        this->PlateRows = rows;
        this->PlateArea = area;
    }
    inline Mat GetPlateMat(){return this->PlateMat;}
    inline Color GetPlateColor(){return this->PlateColor;}
    inline int GetPlateCols(){return this->PlateCols;}
    inline int GetPlateRows(){return this->PlateRows;}
    inline int GetPlateArea(){return this->PlateArea;}
    inline void SetPlate(Mat src, Color color, int cols, int rows , int area)
    {
        SetPlateMat(src);
        SetPlateColor(color);
        SetPlateCols(cols);
        SetPlateRows(rows);
        SetPlateArea(area);
    }
    inline void SetPlateMat(Mat src){src.copyTo(this->PlateMat);}
    inline void SetPlateColor(Color color){this->PlateColor = color;}
    inline void SetPlateCols(int cols){this->PlateCols = cols;}
    inline void SetPlateRows(int rows){this->PlateRows = rows;}
    inline void SetPlateArea(int area){this->PlateArea = area;}
    Color GetUNKNOWPlateColor(Mat &src)
    {
        Mat inputImg = src.clone();
        if(inputImg.channels() == 3)
            cvtColor(inputImg, inputImg, CV_RGB2GRAY);
        cv::threshold(inputImg, inputImg, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);

    }
};
#endif // PLATE_H
