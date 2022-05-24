#include "brisque.h"
#define CATEGORIES 5
#define IMAGENUM   982
#define JP2KNUM    227
#define JPEGNUM    233
#define WNNUM      174
#define GBLURNUM   174
#define FFNUM      174


void trainModel(){

    cout<<"retraining..."<<endl;
    
    FILE* fid;
    const char* foldername = "databaserelease2";
    //---------------------------------------------------- 
    // class is the distortion category, there are 982 images in LIVE database
    vector<string> distortionlabels;
    distortionlabels.push_back("jp2k");
    distortionlabels.push_back("jpeg");
    distortionlabels.push_back("wn");
    distortionlabels.push_back("gblur");
    distortionlabels.push_back("fastfading");

    int imnumber[5] = {0,227,460,634,808};
 
    vector<int>categorylabels;
    categorylabels.insert(categorylabels.end(),JP2KNUM,0);
    categorylabels.insert(categorylabels.end(),JPEGNUM,1);
    categorylabels.insert(categorylabels.end(),WNNUM,2);
    categorylabels.insert(categorylabels.end(),GBLURNUM,3);
    categorylabels.insert(categorylabels.end(),FFNUM,4);
    
    int  iforg[IMAGENUM];
    fid = fopen("livedbfiles/orgs.txt","r");
    for(int itr = 0; itr<IMAGENUM;itr++)
    fscanf(fid,"%d",iforg+itr);
    fclose(fid);
    
    float dmosscores[IMAGENUM];
    fid = fopen("livedbfiles/dmos.txt","r");
    for(int itr =0; itr<IMAGENUM;itr++)
    fscanf(fid,"%f",dmosscores+itr);
    fclose(fid);
 
    const char* filename = "train.txt";
    fid = fopen(filename,"w");
    fclose(fid);
    for(int itr = 0; itr < IMAGENUM; itr++)
    {
      // cout<<itr<<":"<<iforg[itr]<<endl;
      //Dont compute features for original images
      if(iforg[itr])
      continue;

      float score = dmosscores[itr];

      string imname ="";
      imname.append(foldername);
      imname.append("/");
      imname.append(distortionlabels[categorylabels[itr]].c_str());
      imname.append("/img");
      imname.append(static_cast<ostringstream*>( &(ostringstream() <<(itr-imnumber[categorylabels[itr]]+1)))->str());
      imname.append(".bmp");
      cout<<imname<<"\n";
   
      Mat orig = imread(imname.c_str());
      vector<double> brisqueFeatures;
      ComputeBrisqueFeature(orig, brisqueFeatures);
      //printVector(brisqueFeatures);
      printVectortoFile(filename,brisqueFeatures,score);

    }
    
    system("libsvm/svm-scale -l -1 -u 1 -s allrange train.txt > train_scale");
    system("libsvm/svm-train  -s 3 -g 0.05 -c 1024 -b 1 -q train_scale allmodel");
   
    remove("train.txt");
    remove("train_scale");
    
    //return 0;

}

