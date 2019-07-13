#ifndef SVMFEATURE_H
#define SVMFEATURE_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include <stdio.h>
#include <iostream>


using namespace std;
using namespace cv;

 float countOfBigValue(Mat &mat, int iValue) {
    float iCount = 0.0;
    if (mat.rows > 1) {
      for (int i = 0; i < mat.rows; ++i) {
        if (mat.data[i * mat.step[0]] > iValue) {
          iCount += 1.0;
        }
      }
      return iCount;

    } else {
      for (int i = 0; i < mat.cols; ++i) {
        if (mat.data[i] > iValue) {
          iCount += 1.0;
        }
      }

      return iCount;
    }
  }

 Mat ProjectedHistogram(Mat img, int t, int threshold = 20) {
    int sz = (t) ? img.rows : img.cols;
    Mat mhist = Mat::zeros(1, sz, CV_32F);

    for (int j = 0; j < sz; j++) {
      Mat data = (t) ? img.row(j) : img.col(j);

      mhist.at<float>(j) = countOfBigValue(data, threshold);
    }

    // Normalize histogram
    double min, max;
    minMaxLoc(mhist, &min, &max);

    if (max > 0)
      mhist.convertTo(mhist, -1, 1.0f / max, 0);

    return mhist;
}


Mat getHistogram(Mat in) {
  const int VERTICAL = 0;
  const int HORIZONTAL = 1;

  // Histogram features
  Mat vhist = ProjectedHistogram(in, VERTICAL);
  Mat hhist = ProjectedHistogram(in, HORIZONTAL);

  // Last 10 is the number of moments components
  int numCols = vhist.cols + hhist.cols;

  Mat out = Mat::zeros(1, numCols, CV_32F);

  int j = 0;
  for (int i = 0; i < vhist.cols; i++) {
    out.at<float>(j) = vhist.at<float>(i);
    j++;
  }
  for (int i = 0; i < hhist.cols; i++) {
    out.at<float>(j) = hhist.at<float>(i);
    j++;
  }

  return out;
}

void getHistogramFeatures(const Mat& image, Mat& features) {
  Mat grayImage;
  cvtColor(image, grayImage, CV_RGB2GRAY);

  //grayImage = histeq(grayImage);

  Mat img_threshold;
  threshold(grayImage, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
  //Mat img_threshold = grayImage.clone();
  //spatial_ostu(img_threshold, 8, 2, getPlateType(image, false));

  features = getHistogram(img_threshold);
}

// compute color histom
void getColorFeatures(const Mat& src, Mat& features) {
  Mat src_hsv;

  //grayImage = histeq(grayImage);
  cvtColor(src, src_hsv, CV_BGR2HSV);
  int channels = src_hsv.channels();
  int nRows = src_hsv.rows;

  // consider multi channel image
  int nCols = src_hsv.cols * channels;
  if (src_hsv.isContinuous()) {
    nCols *= nRows;
    nRows = 1;
  }

  const int sz = 180;
  int h[sz] = { 0 };

  uchar* p;
  //!把H限制在0-180
  for (int i = 0; i < nRows; ++i) {
    p = src_hsv.ptr<uchar>(i);
    for (int j = 0; j < nCols; j += 3) {
      int H = int(p[j]);      // 0-180
      if (H > sz - 1) H = sz - 1;
      if (H < 0) H = 0;
      h[H]++;
    }
  }

  Mat mhist = Mat::zeros(1, sz, CV_32F);
  for (int j = 0; j < sz; j++) {
    mhist.at<float>(j) = (float)h[j];
  }

  // Normalize histogram
  double min, max;
  minMaxLoc(mhist, &min, &max);

  if (max > 0)
    mhist.convertTo(mhist, -1, 1.0f / max, 0);

  features = mhist;
}


void getHistomPlusColoFeatures(const Mat& image, Mat& features) {
  // TODO
  Mat feature1, feature2;
  getHistogramFeatures(image, feature1);
  getColorFeatures(image, feature2);
  hconcat(feature1.reshape(1, 1), feature2.reshape(1, 1), features);
}

#endif // SVMFEATURE_H
