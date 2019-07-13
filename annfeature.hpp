#ifndef ANNFEATURE_H
#define ANNFEATURE_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/ml/ml.hpp>
#include "svmfeature.hpp"


Mat charFeatures(Mat in, int sizeData) {
  const int VERTICAL = 0;
  const int HORIZONTAL = 1;

  // cut the cetner, will afect 5% perices.
  Rect _rect = GetCenterRect(in);
  Mat tmpIn = CutTheRect(in, _rect);
  //Mat tmpIn = in.clone();

  // Low data feature
  Mat lowData;
  resize(tmpIn, lowData, Size(sizeData, sizeData));

  // Histogram features
  Mat vhist = ProjectedHistogram(lowData, VERTICAL);
  Mat hhist = ProjectedHistogram(lowData, HORIZONTAL);

  // Last 10 is the number of moments components
  int numCols = vhist.cols + hhist.cols + lowData.cols * lowData.cols;

  Mat out = Mat::zeros(1, numCols, CV_32F);
  // Asign values to

  int j = 0;
  for (int i = 0; i < vhist.cols; i++) {
    out.at<float>(j) = vhist.at<float>(i);
    j++;
  }
  for (int i = 0; i < hhist.cols; i++) {
    out.at<float>(j) = hhist.at<float>(i);
    j++;
  }
  for (int x = 0; x < lowData.cols; x++) {
    for (int y = 0; y < lowData.rows; y++) {
      out.at<float>(j) += (float)lowData.at <unsigned char>(x, y);
      j++;
    }
  }

  //std::cout << out << std::endl;

  return out;
}


Mat charFeatures2(Mat in, int sizeData) {
  const int VERTICAL = 0;
  const int HORIZONTAL = 1;

  // cut the cetner, will afect 5% perices.
  Rect _rect = GetCenterRect(in);
  Mat tmpIn = CutTheRect(in, _rect);
  //Mat tmpIn = in.clone();

  // Low data feature
  Mat lowData;
  resize(tmpIn, lowData, Size(sizeData, sizeData));

  // Histogram features
  Mat vhist = ProjectedHistogram(lowData, VERTICAL);
  Mat hhist = ProjectedHistogram(lowData, HORIZONTAL);

  // Last 10 is the number of moments components
  int numCols = vhist.cols + hhist.cols + lowData.cols * lowData.cols;

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
  for (int x = 0; x < lowData.cols; x++) {
    for (int y = 0; y < lowData.rows; y++) {
      out.at<float>(j) += (float)lowData.at <unsigned char>(x, y);
      j++;
    }
  }

  //std::cout << out << std::endl;

  return out;
}

Mat charProjectFeatures(const Mat& in, int sizeData) {
  const int VERTICAL = 0;
  const int HORIZONTAL = 1;

  SHOW_IMAGE(in, 0);
  // cut the cetner, will afect 5% perices.

  Mat lowData;
  resize(in, lowData, Size(sizeData, sizeData));

  SHOW_IMAGE(lowData, 0);
  // Histogram features
  Mat vhist = ProjectedHistogram(lowData, VERTICAL);
  Mat hhist = ProjectedHistogram(lowData, HORIZONTAL);

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
  //std::cout << out << std::endl;

  return out;
}

void getGrayCharFeatures(const Mat& grayChar, Mat& features) {
  // TODO: check channnels == 1
  SHOW_IMAGE(grayChar, 0);
  SHOW_IMAGE(255 - grayChar, 0);

  // resize to uniform size, like 20x32
  bool useResize = false;
  bool useConvert = true;
  bool useMean = true;
  bool useLBP = false;

  Mat char_mat;
  if (useResize) {
    char_mat.create(kGrayCharHeight, kGrayCharWidth, CV_8UC1);
    resize(grayChar, char_mat, char_mat.size(), 0, 0, INTER_LINEAR);
  } else {
    char_mat = grayChar;
  }
  SHOW_IMAGE(char_mat, 0);

  // convert to float
  Mat float_img;
  if (useConvert) {
    float scale = 1.f / 255;
    char_mat.convertTo(float_img, CV_32FC1, scale, 0);
  } else {
    float_img = char_mat;
  }
  SHOW_IMAGE(float_img, 0);

  // cut from mean, it can be optional

  Mat mean_img;
  if (useMean) {
    float_img -= mean(float_img);
    mean_img = float_img;
  } else {
    mean_img = float_img;
  }
  SHOW_IMAGE(mean_img, 0);

  // use lbp to get features, it can be changed to other
  Mat feautreImg;
  if (useLBP) {
    Mat lbpimage = libfacerec::olbp(char_mat);
    SHOW_IMAGE(lbpimage, 0);
    feautreImg = libfacerec::spatial_histogram(lbpimage, kCharLBPPatterns, kCharLBPGridX, kCharLBPGridY);
  } else {
    feautreImg = mean_img.reshape(1, 1);
  }

  // return back
  features = feautreImg;
}


void getGrayPlusProject(const Mat& grayChar, Mat& features)
{
  // TODO: check channnels == 1
  SHOW_IMAGE(grayChar, 0);
  SHOW_IMAGE(255 - grayChar, 0);

  // resize to uniform size, like 20x32
  bool useResize = false;
  bool useConvert = true;
  bool useMean = true;
  bool useLBP = false;

  Mat char_mat;
  if (useResize) {
    char_mat.create(kGrayCharHeight, kGrayCharWidth, CV_8UC1);
    resize(grayChar, char_mat, char_mat.size(), 0, 0, INTER_LINEAR);
  }
  else {
    char_mat = grayChar;
  }
  SHOW_IMAGE(char_mat, 0);

  // convert to float
  Mat float_img;
  if (useConvert) {
    float scale = 1.f / 255;
    char_mat.convertTo(float_img, CV_32FC1, scale, 0);
  }
  else {
    float_img = char_mat;
  }
  SHOW_IMAGE(float_img, 0);

  // cut from mean, it can be optional

  Mat mean_img;
  if (useMean) {
    float_img -= mean(float_img);
    mean_img = float_img;
  }
  else {
    mean_img = float_img;
  }
  SHOW_IMAGE(mean_img, 0);

  // use lbp to get features, it can be changed to other
  Mat feautreImg;
  if (useLBP) {
    Mat lbpimage = libfacerec::olbp(char_mat);
    SHOW_IMAGE(lbpimage, 0);
    feautreImg = libfacerec::spatial_histogram(lbpimage, kCharLBPPatterns, kCharLBPGridX, kCharLBPGridY);
  }
  else {
    feautreImg = mean_img.reshape(1, 1);
  }
  SHOW_IMAGE(grayChar, 0);
  Mat binaryChar;
  threshold(grayChar, binaryChar, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
  SHOW_IMAGE(binaryChar, 0);
  Mat projectFeature = charProjectFeatures(binaryChar, 32);

  hconcat(feautreImg.reshape(1, 1), projectFeature.reshape(1, 1), features);
}


void getGrayPlusLBP(const Mat& grayChar, Mat& features)
{
  // TODO: check channnels == 1
  SHOW_IMAGE(grayChar, 0);
  SHOW_IMAGE(255 - grayChar, 0);

  // resize to uniform size, like 20x32
  bool useResize = false;
  bool useConvert = true;
  bool useMean = true;
  bool useLBP = true;

  Mat char_mat;
  if (useResize) {
    char_mat.create(kGrayCharHeight, kGrayCharWidth, CV_8UC1);
    resize(grayChar, char_mat, char_mat.size(), 0, 0, INTER_LINEAR);
  }
  else {
    char_mat = grayChar;
  }
  SHOW_IMAGE(char_mat, 0);

  // convert to float
  Mat float_img;
  if (useConvert) {
    float scale = 1.f / 255;
    char_mat.convertTo(float_img, CV_32FC1, scale, 0);
  }
  else {
    float_img = char_mat;
  }
  SHOW_IMAGE(float_img, 0);

  // cut from mean, it can be optional

  Mat mean_img;
  if (useMean) {
    float_img -= mean(float_img);
    mean_img = float_img;
  }
  else {
    mean_img = float_img;
  }
  SHOW_IMAGE(mean_img, 0);

  // use lbp to get features, it can be changed to other
  Mat originImage = mean_img.clone();
  Mat lbpimage = libfacerec::olbp(mean_img);
  SHOW_IMAGE(lbpimage, 0);
  lbpimage = libfacerec::spatial_histogram(lbpimage, kCharLBPPatterns, kCharLBPGridX, kCharLBPGridY);

  // 32x20 + 16x16
  hconcat(mean_img.reshape(1, 1), lbpimage.reshape(1, 1), features);
}

void getLBPplusHistFeatures(const Mat& image, Mat& features) {
  Mat grayImage;
  cvtColor(image, grayImage, CV_RGB2GRAY);

  Mat lbpimage;
  lbpimage = libfacerec::olbp(grayImage);
  Mat lbp_hist = libfacerec::spatial_histogram(lbpimage, 64, 8, 4);
  //features = lbp_hist.reshape(1, 1);

  Mat greyImage;
  cvtColor(image, greyImage, CV_RGB2GRAY);

  //grayImage = histeq(grayImage);
  Mat img_threshold;
  threshold(greyImage, img_threshold, 0, 255,
    CV_THRESH_OTSU + CV_THRESH_BINARY);
  Mat histomFeatures = getHistogram(img_threshold);

  hconcat(lbp_hist.reshape(1, 1), histomFeatures.reshape(1, 1), features);
  //std::cout << features << std::endl;
  //features = histomFeatures;
}

#endif // ANNFEATURE_H
