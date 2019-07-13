#include "plate.hpp"
#include "platelocate.hpp"
#include "svm.hpp"
#include "anncharrecognise.hpp"
#include "chars_segment.h"
#include <stdio.h>

using namespace std;
using namespace cv;

bool plateColorJudge(plate &p, const Color r, const bool adaptive_minsv,
                       float &percent);

int main()
{
    vector<plate> SobelPlate, ColorPlate, svmColorPlate, svmSobelPlate;
    Mat src = imread("plate2.jpg");
    platedetect detect;
    detect.BluePlusYellowLocate(src);
    ColorPlate = detect.GetColorDetectPlate();
    platesvm svm_("hcsvm.xml");
    svm_.PlateSvmJudgeVecPlate(ColorPlate, svmColorPlate);
    vector<Mat> SingleChar[svmColorPlate.size()];
    CCharsSegment cchar;
    cchar.charsSegmentPlate(svmColorPlate[0], SingleChar[0]);
    annrecognise ann_;
    string result = ann_.charsIdentify(SingleChar[0][0], true);
    for(int i = 1; i < SingleChar[0].size(); i++)
    {
        result += ann_.charsIdentify(SingleChar[0][i], false);
    }
    cout << result;
    return 0;
}
/*

Color getPlateType( plate &p, const bool adaptive_minsv) {
    float max_percent = 0;
    Color max_color = UNKNOW;

    float blue_percent = 0;
    float yellow_percent = 0;
    float white_percent = 0;

    if (plateColorJudge(p, BLUE, adaptive_minsv, blue_percent) == true) {
      // cout << "BLUE" << endl;
      return BLUE;
    } else if (plateColorJudge(p, YELLOW, adaptive_minsv, yellow_percent) ==
               true) {
      // cout << "YELLOW" << endl;
      return YELLOW;
    }
    else {
      //std::cout << "OTHER" << std::endl;

      /*max_percent = blue_percent > yellow_percent ? blue_percent : yellow_percent;
      max_color = blue_percent > yellow_percent ? BLUE : YELLOW;
      max_color = max_percent > white_percent ? max_color : WHITE;

      /// always return blue
      return BLUE;
    }
  }


 bool plateColorJudge(plate &p, const Color r, const bool adaptive_minsv,
                       float &percent) {
    Mat src = p.GetPlateMat();
    const float thresh = 0.45f;

    Mat src_gray;
    platedetect::colorMatch(src, src_gray, r, adaptive_minsv);

    percent =
        float(countNonZero(src_gray)) / float(src_gray.rows * src_gray.cols);
    // cout << "percent:" << percent << endl;

    if (percent > thresh)
      return true;
    else
      return false;
  }
*/
