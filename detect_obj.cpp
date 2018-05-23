#include <stdio.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;
#define rangie 5

Mat hsv_img;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

void mouseCB(int event, int x, int y, int flags, void* userdata)
{
 if  ( event == EVENT_LBUTTONDOWN )
 {
  cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
  cout << (int)hsv_img.at<Vec3b>(y,x)[0] << " " << (int)hsv_img.at<Vec3b>(y,x)[1] << " " << (int)hsv_img.at<Vec3b>(y,x)[2] << endl;
}
}


int main(int argc, char** argv)
{
  char a;
  cout<<"image(i) or video(v)";
  cin>>a;

  Mat grid_thr, grid_thr_l, no_grid, rb_thr, gb_thr, gb, rb, allb;
  int h=177,s=54,v=200,ht=56,st=55;
  int h_g=85,s_g=95,v_g=93,ht_g=38,st_g=48;
  int h_r=9,s_r=186,v_r=212,ht_r=6,st_r=103;

  namedWindow("Sliders");
  namedWindow("MainIM", CV_WINDOW_NORMAL);
  setMouseCallback("MainIM", mouseCB, NULL);
  createTrackbar("H","Sliders",&h,255);
  createTrackbar("S","Sliders",&s,255);
  createTrackbar("V","Sliders",&v,255);
  createTrackbar("HT","Sliders",&ht,255);
  createTrackbar("ST","Sliders",&st,255);

  createTrackbar("H_G","Sliders",&h_g,255);
  createTrackbar("S_G","Sliders",&s_g,255);
  createTrackbar("V_G","Sliders",&v_g,255);
  createTrackbar("HT_G","Sliders",&ht_g,255);
  createTrackbar("ST_G","Sliders",&st_g,255);

  createTrackbar("H_R","Sliders",&h_r,255);
  createTrackbar("S_R","Sliders",&s_r,255);
  createTrackbar("V_R","Sliders",&v_r,255);
  createTrackbar("HT_R","Sliders",&ht_r,255);
  createTrackbar("ST_R","Sliders",&st_r,255);

  if (a=='i')
  {
    Mat img=imread(argv[1],CV_LOAD_IMAGE_COLOR);

    Mat src, src_gray;
    cvtColor(img, hsv_img, CV_BGR2HSV_FULL);


    imshow("MainIM",img);
    while(1)
    {
      inRange(hsv_img,Scalar(h-ht,s-st,0), Scalar(h+ht,s+st,255),grid_thr);
      inRange(hsv_img,Scalar(h_g-ht_g,s_g-st_g,0), Scalar(h_g+ht_g,s_g+st_g,255),gb_thr);
      inRange(hsv_img,Scalar(h_r-ht_r,s_r-st_r,0), Scalar(h_r+ht_r,s_r+st_r,255),rb_thr);

      vector < vector<Point2i > > blobs;
      Mat grid_thr_s;
      bitwise_not(grid_thr, no_grid);
      bitwise_and(no_grid, gb_thr, gb);
      bitwise_and(no_grid, rb_thr, rb);
      bitwise_or(rb, gb, allb);
      namedWindow("gthr", CV_WINDOW_NORMAL );
      namedWindow("rb", CV_WINDOW_NORMAL );
      namedWindow("gb", CV_WINDOW_NORMAL );
      namedWindow("allb", CV_WINDOW_NORMAL );

      GaussianBlur( allb, allb, Size(9, 9), 2, 2 );

      imshow("gthr", grid_thr);
      imshow("rb", rb);
      imshow("gb", gb);
      imshow("allb", allb);
      ofstream outfile;
      outfile.open("mat.dat");
      for(int i=0;i<allb.rows;i++)
      {
        for(int j=0;j<allb.cols;j++)
        {
          if(allb.at<uchar>(i,j)>50 )
            outfile<<i<<" "<<j<<endl;
        }
      }
    //outfile<< rb<<endl;
      imwrite("../obj_det/grid_thr.jpg",grid_thr);
      imwrite("../obj_det/rb.jpg",rb);
      imwrite("../obj_det/gb.jpg",gb);
      imwrite("../obj_det/allb.jpg",allb);
      waitKey(33);
    }
  }
  else
  {
    VideoCapture cap(argv[1]);
    namedWindow("gthr", CV_WINDOW_NORMAL );
    namedWindow("rb", CV_WINDOW_NORMAL );
    namedWindow("gb", CV_WINDOW_NORMAL );
    namedWindow("allb", CV_WINDOW_NORMAL );
    namedWindow("detect", CV_WINDOW_NORMAL );
    int cont=0;
    float stored[10][2][3];
    int count[rangie]={0,0,0,0,0};
    while(1)
    {

      Mat img;
      cap >> img;
      
      cvtColor(img, hsv_img, CV_BGR2HSV_FULL);
      if (img.empty())
        break;
      imshow( "MainIM", img );

      inRange(hsv_img,Scalar(h-ht,s-st,0), Scalar(h+ht,s+st,255),grid_thr);
      inRange(hsv_img,Scalar(h_g-ht_g,s_g-st_g,0), Scalar(h_g+ht_g,s_g+st_g,255),gb_thr);
      inRange(hsv_img,Scalar(h_r-ht_r,s_r-st_r,0), Scalar(h_r+ht_r,s_r+st_r,255),rb_thr);

      vector < vector<Point2i > > blobs;
      Mat grid_thr_s,detect,src_gray;
      
      bitwise_not(grid_thr, no_grid);
      bitwise_and(no_grid, gb_thr, gb);
      bitwise_and(no_grid, rb_thr, rb);
      bitwise_or(rb, gb, allb);

      GaussianBlur( allb, allb, Size(9, 9), 0, 0 );
      allb.copyTo(detect);
      allb.copyTo(src_gray);

      Mat canny_output;
      vector<vector<Point> > contours;
      vector<Vec4i> hierarchy;

  /// Detect edges using canny
      Canny( src_gray, canny_output, thresh, thresh*2, 3 );
  /// Find contours
      findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
      Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
  /// Draw contours
      int at[250][2]={0};
      int ct=0;
      cout<<"\nSize:"<<contours.size()<<endl;
      for(int i=0;i<contours.size();i++)
      {
        int sumx=0;
        int sumy=0;
        if(contours[i].size()>15)
        {
          for(int j=0;j<contours[i].size();j++)
          {
            sumx+=contours[i][j].x;
            sumy+=contours[i][j].y;
          }
          int b=sumx/contours[i].size();
          int c=sumy/contours[i].size();
          int k;
          for(k=0;k<contours.size();k++)
          {
            if(abs(b-at[k][0])+abs(c-at[k][1])<20)
              break;
          }
          if(k==contours.size())
          {
           
            at[ct][0]=b;
            at[ct++][1]=c;
          }
        }
      }
      for(int i=0;i<ct;i++)
      {
        cout<<at[i][0]<<","<<at[i][1]<<endl;
        circle(detect, Point(at[i][0],at[i][1]),50, Scalar(255,255,255),3, 8,0);
      }
      for( int i = 0; i< contours.size(); i++)
      {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
      }

      namedWindow( "Contours", CV_WINDOW_NORMAL );
      imshow( "Contours", drawing );
      cout<<endl;      
      imshow("no_grid", no_grid);
      imshow("grid_thr", grid_thr);

      imshow("rb", rb);
      imshow("gb", gb);
      imshow("allb", allb);
      imshow("detect",detect);
      imwrite("../obj_det/grid_thr.jpg",grid_thr);
      imwrite("../obj_det/MainIM.jpg",img);
      imwrite("../obj_det/rb.jpg",rb);
      imwrite("../obj_det/gb.jpg",gb);
      imwrite("../obj_det/allb.jpg",allb);
      cont++;
      waitKey(100);
    }

    cap.release();
    
    destroyAllWindows();

  }
  return 0;
}
//Feedback:- /mavros/local_postition/pose (postitons) /odem (with orientation)
// pose.pose.position.x