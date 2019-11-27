package projekt.simsearch;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.features2d.Features2d;
//import org.opencv.features2d.ORB;
//import org.opencv.features2d.BRISK;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.*;
//import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.highgui.Highgui;
import org.opencv.features2d.FeatureDetector;


public class FeatureDetection {
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		//imports images, converts to matrix
		String img1 = "C:/Users/saray/Desktop/3. Semester M.Sc/SimSearch/images/redroses.jpg";
		String img2 = "C:/Users/saray/Desktop/3. Semester M.Sc/SimSearch/images/redroses.jpg";
		Mat mat1 = Highgui.imread(img1);
		Mat mat2 = Highgui.imread(img2);
		
		//detects features, stores them in MatOfKeyPoint
		//Keypoints unter drawKeypoints sehen mit SURF besser aus als mit ORB etc.
		FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.ORB); 

		MatOfKeyPoint keysImg1 = new MatOfKeyPoint();
		MatOfKeyPoint keysImg2 = new MatOfKeyPoint();
		
		featureDetector.detect(mat1, keysImg1);
		featureDetector.detect(mat2, keysImg2);
		
		//shows img1 with keypoints
		//Keypoints mit ORB sind zu wenig und komisch verteilt, WARUM?? 
		/**
		Scalar color = new Scalar(255, 0, 0);
		Mat outImg = new Mat();
		Features2d.drawKeypoints(mat1, keysImg1, outImg, color, 0); //statt 0: Features2d.DRAW_RICH_KEYPOINTS oder Features2d.NOT_DRAW_SINGLE_POINTS
		Highgui.imwrite("C:/Sara/images/TRY2.jpg", outImg);
		**/
		
		//extracts descriptors for features, stores them in Mat
		//brauchen wir das schon? oder erst nach dem Clustern der Keypoints?
		DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
		
		Mat descriptorsImg1 = new Mat();
		Mat descriptorsImg2 = new Mat();
		
		descriptorExtractor.compute(mat1, keysImg1, descriptorsImg1);
		descriptorExtractor.compute(mat2, keysImg2, descriptorsImg2);	
		//System.out.println(descriptorsImg1.dump());
		
		
		//clustering, kmeans returns the compactness measure, i.e. how good the labeling was done
		// Compactness(CP) measures the average distance between every pair of data
		// points in a cluster. If there are multiple clusters, Compactness is the
		// average of all clusters.
		// A low value of CP indicates better and more compact clusters.
		int k = 20;
		Mat labels1 = new Mat();
		TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER,100,0.1);
		int attempts = 10;
		Mat centers1 = new Mat();		
		Core.kmeans(descriptorsImg1, k, labels1, criteria, attempts, Core.KMEANS_PP_CENTERS, centers1);
		//centers ist k x 64 Matrix, also 20x64 hier
		//System.out.println(labels1.dump()); 
		
		Mat labels2 = new Mat();
		Mat centers2 = new Mat();		
		Core.kmeans(descriptorsImg2, k, labels2, criteria, attempts, Core.KMEANS_PP_CENTERS, centers2);
		//System.out.println(centers2.dump());
		
		//System.out.println(descriptorsImg2.rows());
		//System.out.println(labels2.rows());
		//Einträge/Zeilen in labels insgesamt muss gleich Zeilen in descriptors sein, passt

		//k = 2 --> labels = 0 oder 1,
		//k = 5 --> lables = 0,1,2,3 oder 4
		//Anzahl unique labels = k
		//wenn k = 2: bei Änderung von attempts ändert sich auch Reihenfolge von labels
	  
		
		//MATCHING
		//matches the features of two images
		DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE); //BF, BFL1, BFSL2, FLANNBASED
		
		MatOfDMatch matches = new MatOfDMatch();		
	    matcher.match(centers1, centers2, matches);
	    //descriptor matching images look way better
		//matcher.match(descriptorsImg1, descriptorsImg2, matches);
	    //System.out.println(matches.dump());
		
		//doesn't work because drawMatches doesn't take list of matofdmatch
		//List<MatOfDMatch> matches = new LinkedList<MatOfDMatch>();
		//matcher.knnMatch(descriptorsImg1, descriptorsImg2, matches, 2);
		
	    
	    //DRAW MATCHES
		Mat newImg = new Mat();
		Scalar matchColor = new Scalar(0,255,255);
		Scalar keyColor = new Scalar(255,255,255);
		MatOfByte matchesMask = new MatOfByte();
		Features2d.drawMatches(mat1, keysImg1, mat2, keysImg2, matches, newImg, matchColor, keyColor, matchesMask, 0);
		//Highgui.imwrite("C:/Users/saray/Desktop/3. Semester M.Sc/SimSearch/center20_matchersBF_ORB_SURF.jpg", newImg); 
		
		 //TODO knnMatch and Ratio-Test for goodMatches
	    /**
	    LinkedList<MatOfDMatch> matchesList = new LinkedList<>();
	    matcher.knnMatch(centers1, centers2, matchesList, 2);
	    
	    LinkedList<DMatch> goodMatchesList = new LinkedList<>();
	    for (int i = 0; i < matchesList.size() ; i++) {
	        double ratio = 0.8;
	        if (matchesList.get(i).toArray()[0].distance < ratio * matchesList.get(i).toArray()[1].distance) {
	            goodMatchesList.addLast(matchesList.get(i).toArray()[0]);
	        }
	    }
	    MatOfDMatch goodMatches = new MatOfDMatch();
	    goodMatches.fromList(goodMatchesList);
	    **/
	    
	    
	    
		/**
		//clustering Versuche, ALT
		Mat src = imread( argv[1], 1 );
		Mat samples = new Mat(src.rows() * src.cols(), 3, CV_32F);
		  for( int y = 0; y < src.rows(); y++ )
		    for( int x = 0; x < src.cols(); x++ )
		      for( int z = 0; z < 3; z++)
		        samples.at<float>(y + x*src.rows(), z) = src.at<Vec3b>(y,x)[z];


		  int clusterCount = 15;
		  Mat labels;
		  int attempts = 5;
		  Mat centers;
		  kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), 
				  attempts, KMEANS_PP_CENTERS, centers );


		  Mat new_image = new Mat( src.size(), src.type() );
		  for( int y = 0; y < src.rows; y++ )
		    for( int x = 0; x < src.cols; x++ )
		    { 
		      int cluster_idx = labels.at<int>(y + x*src.rows,0);
		      new_image.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
		      new_image.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
		      new_image.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
		    }
		  imshow( "clustered image", new_image );
		  waitKey( 0 );
		  **/
		
		
	    
		
		//FUNKTIONIERT, aus Cluster Klasse, gibt {0=1013571, 1=246995, 2=794506, 3=211283, 4=733645}
		//{0=29861, 1=39980, 2=28539, 3=46867, 4=14753} zurück
	    /**
		int i = 0;
	    int y = 0;
		List<Mat> clusters1 = Cluster.cluster(mat1, 5); //keysImg1 und keysImg2 klappt nicht
		List<Mat> clusters2 = Cluster.cluster(mat2, 5);
		**/
		
		//generates images of the cluster matrices
		/**
		for(Mat clusteredMat : clusters1) {
	    	Highgui.imwrite("C:/Sara/images/cluster1_" + i++ + ".jpg", clusteredMat);
		}
	    
	    for(Mat clusteredMat : clusters2) {
	    	Highgui.imwrite("C:/Sara/images/cluster2_" + y++ + ".jpg", clusteredMat);	
	    }**/
	    
	    
	    /**
	    //print cluster matrices
		//get(y).dump() druckt Matrix-Inhalt aus, 5x 2000x1500
	    for(int z = 0; z < clusters1.size(); z++) {
	    	System.out.println(clusters1.get(z)); 
	    }
	    
	    for(int z = 0; z < clusters2.size(); z++) {
	    	System.out.println(clusters2.get(z)); 
	    }
	    **/
	    
	    

	    
		
		
		//-------------------------------------------------------------------------------------------
	    /** ALT

		Mat mask1 = new Mat();
		Mat mask2 = new Mat();
		Mat maskOut = new Mat();
		
		//ORB or BRISK detector
		ORB detector = ORB.create();
		//BRISK detector = BRISK.create();
		detector.detect(mat1, keypoints1, mask1); 
		detector.detect(mat2, keypoints2, mask2);
		//draws the matches between two images, creates new image
		
		
		Mat newImg = new Mat();
		Features2d.drawMatches(mat1, keysImg1, mat2, keysImg2, matofdmatch, newImg);
		Highgui.imwrite("C:/Sara/matches.jpg", newImg);
		**/
		
		
		//k-medoids descriptor von JavaML angucken
		//bruteforce-hamming
		//maxDisOfMatch, D1LTD2 (1-100), minNumOfGoodMatchesToDraw, homography and ransac test
		//lieber KNN statt kmeans
		//earth movers, JFast
		
		//KMeans kmeans = new KMeans(2,5, new EuclideanDistance());
		//Dataset[] var = kmeans.cluster();
		
		//von OpenCV angebotene Matching Distanzen probieren, Bruteforce-Hamming geht nicht mit SURF, andere probieren
		//Earth Mover's Distance
		//Keypoint ranking, wie viele gleiche keypoints?
		
		//RANSAC und RATIO Test ausprobieren, sortieren um nur beste Matches zu finden
	    //knnMatch probieren, siehe Lesezeichen
		//kmeans bestimmt wie viele Cluster, das bestimmt wie viele Matches, wie optimieren wir k? 
		//auf Bildern: die gematchten gelben Keypoints sind Centroids? 
	}

}
