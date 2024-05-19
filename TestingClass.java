import java.awt.Dimension;
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.image.BufferedImageOp;
import java.io.File;
import java.lang.reflect.Array;
import java.util.Arrays;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;
import org.opencv.objdetect.*;
import org.opencv.core.MatOfFloat;


/**
 * This Class is made for debugging/testing of the functionality of other classes and is not going to be used in the final version
 *
 *
 */


public class TestingClass {

	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		/**
		 * first thing to be tested is the image capturing functionality
		 */
		
		System.out.println("TestingClass.java started");
		
		
		//for now, we are going to visually test the image capturing functionality using HighGui 
		//I haven't found a way to avoid using waitKey(1)
		/**
		//turn to true to get one capture of web cam
		boolean getimagefromcam = false;
		if (getimagefromcam) {
		HighGui.imshow("test image", CamData.getMat());
		HighGui.waitKey(1);
		}
		
		//Testing the continuous image displaying functionality
		
		//turn to true to get video (continuous stram of images)
		boolean video = false;

		while (video) {
		HighGui.imshow("test image", CamData.getMat());
		HighGui.waitKey(1);
		}
		*/
		
		//TODO find out what waitKey does and how to avoid using it. 
		//
		
		//because opencv is c++, native library name has to be loaded before using any native coe
	    //System.loadLibrary(Core.NATIVE_LIBRARY_NAME); // load opencv_java
	    
		/**
		 * In this part, an attempt is going to be conducted on face detection and display, in the following order
		 * 1. Detect Face and get the MatOfRect of the face
		 * 2. Overaly and get an overlayed Mat
		 * 3. Display the overlayed mat
		 */
	    
	    
		
		//String filename = "haarcascade_frontalcatface.xml"; //all files are hidden in RemEye folder, among the src and bin folders
		//String filename2 = "lbpcascade_frontalface_improved.xml"; 
		
		
		
		//CascadeClassifier faceCC = new CascadeClassifier(filename2);
		//System.out.println(faceCC.empty()); //check for CascadeClassifier being loaded
		//System.out.println(CamData.getMat().dump());
		//MatOfRect rects = new MatOfRect(); //MatOfRect is the type var which will contain the rectangle with that face
		//System.out.println(rects.dump());
		
		/* TESTING AREA  works 
		// * 
		faceCC.detectMultiScale(CamData.getMat(), rects);
		Rect[] arrayrects = rects.toArray();
		
		while (arrayrects.length < 1) {
		faceCC.detectMultiScale(CamData.getMat(), rects);
		arrayrects = rects.toArray();
		System.out.println("Tryig to get face rect");
		}
		Mat nextimage = CamData.getMat();
		
		
		
		System.out.println(arrayrects[0]);
		System.out.println("x: " + arrayrects[0].x + "y: " + arrayrects[0].y + " heights: " + arrayrects[0].height + " width: " + arrayrects[0].width);
		Imgproc.rectangle(nextimage, arrayrects[0], new Scalar(255));
		HighGui.imshow("square", nextimage);
		HighGui.waitKey(1);
		
		*/
		
		//Testing the continuous image displaying functionality
		
		//turn to true to get video (continuous stram of images)
		
		///*
		/*
		boolean videodetect = false;      // set to true to test the detection on a continuous stream
		MatOfRect videorects = new MatOfRect();
		Rect[] arrayvideorects;
		Rect something = null;
		Mat videomat = null;
//		Mat grayvid = CamData.getMat();		
		Mat face = null;
		Size dsize = new Size(50, 50);
		while (videodetect) {
			videomat = CamData.getMat();
		
				
				Imgproc.cvtColor(videomat, videomat, Imgproc.COLOR_BGR2GRAY);
				faceCC.detectMultiScale(videomat, videorects, 2, 2, 0, dsize); //this is where the hiccup occurs
				arrayvideorects = videorects.toArray();
				
				
				System.out.println(arrayvideorects.length); // has to be at least 1 for a face
				if (arrayvideorects.length > 0) {
					something = arrayvideorects[0];
				}
				if (something != null) {
					videomat = new Mat(videomat, something);
					HighGui.imshow("test image", videomat);
					HighGui.waitKey(20);
				} else {
	//				Imgproc.rectangle(videomat, something, new Scalar(255));
				HighGui.imshow("test image", videomat);
				HighGui.waitKey(20);
				}
				something = null;
			
		}
		
		*/
		/**
		 * 
		 */
		
		//System.out.println(arrayrects.length);
		//System.out.println(arrayrects);
		//System.out.println(rects);

		
		//HighGui.imshow("test image", testface);
		//HighGui.waitKey(1);
		
		//HighGui.imshow("testface", CamData.getFace());
		//HighGui.waitKey(50);
		
		/**
		 * Testing eye detection on a face
		 */
		/*
		
		String eyefilet = "haarcascade_eye.xml"; 
		Mat facemat = CamData.getFace();
		Mat eyemat1 = new Mat();
		Mat eyemat2 = new Mat();
		MatOfRect eyerect = new MatOfRect();
		CascadeClassifier eyeCC = new CascadeClassifier(eyefilet);
		System.out.println(eyeCC.empty());
		
		
		eyeCC.detectMultiScale(facemat, eyerect);
		
		
		Rect[] eyearray = eyerect.toArray();
		if (eyearray.length > 0) {
		System.out.print(eyearray.length);
		/*Imgproc.rectangle(facemat, eyearray[0], new Scalar(255));
		Imgproc.rectangle(facemat, eyearray[1], new Scalar(255));
		eyemat1 = new Mat(facemat, eyearray[0]);
		eyemat2 = new Mat(facemat, eyearray[1]);
		HighGui.imshow("testface1", eyemat1);
		HighGui.waitKey(50);
		HighGui.imshow("testface2", eyemat2);
		HighGui.waitKey(50);
		
		 */
		/*
		} else
			System.out.println("no eyes found");
		*/
		
		
		
		//Testing the CamData.save and retrieve methods
		
		/**
		
		System.out.println("Testing .save method with size constructor");
		CamData a = new CamData(1);
		int[] testCoords = new int[]{453, 234};
		System.out.println("saved coords? " + a.save(testCoords));
		System.out.println("getCoords test " + a.getCoords(0)[1]);
		System.out.println("getX test " + a.getX(0));
		System.out.println("get eye test " + a.getEye0(0).toString());
		
		System.out.println("test saving one more than given size");
		int[] testCoords1 = new int[]{99999, 7777722};
		System.out.println("saved coords? " + a.save(testCoords1));
		System.out.println("getCoords test " + a.getCoords(1)[1]);
		System.out.println("getX test " + a.getX(1));
		System.out.println("get eye test " + a.getEye0(1).toString());
		

		System.out.println("test testing everything without size");
		CamData b = new CamData();
		int[] testCoords2 = new int[]{1414414141, 999};
		System.out.println("saved coords? " + b.save(testCoords2));
		System.out.println("getCoords test y " + b.getCoords(0)[1]);
		System.out.println("getX test " + b.getX(0));
		System.out.println("get eye test " + b.getEye0(0).toString());
		*/
		
		/**
		 * Testing Debug mode vatiations
		 */
		
		//CamData.debugCamView();
		//CamData.debugFaceView();
		//CamData.debugFaceEyesView();
		//CamData.debugFaceEyesImprovedView();
		/*
		while (true) {
			HighGui.imshow("winname", CamData.getEyesImproved()[0]);
			HighGui.imshow("winname", CamData.getEyesImproved()[1]);
			HighGui.waitKey(20);
		}
		*/
		//HighGui.imshow("winname", CamData.getEyesImproved()[0]);
		//HighGui.waitKey(20);
		
		/**
		/**
		 * Testing sample eye comparison methods and deciding on the best one
		 */
		//CamData.debugFaceEyesImprovedView();
		/*
		CamData test = new CamData();
		int[] ndas = {1,2};
		if (test.save(ndas))
			System.out.println(test.getSize());
		if (test.save(ndas))
			System.out.println(test.getSize());
		test.debugExistingEyes();
		*/
		/*
		System.out.println("adsas");
		
		
		
		
		
		// testing that knn
		
		
		Mat[] testEyes = CamData.getEyesImproved();							// 1. GET EYES FROM CAM
		Mat testEye = testEyes[0];
		Mat testEye2 = testEyes[1];
		System.out.println(testEyes[0].toString() + " " + testEyes[1].toString());
		//HighGui.imshow("winname", testEye);
		//HighGui.waitKey(20);
		Imgproc.resize(testEye, testEye, (new Size(160, 160)));				// 2. RESIZE EYES || REQUIRED because eyes are different
		Imgproc.resize(testEye2, testEye2, (new Size(160, 160)));
		//Mat[] testEyes2 = CamData.getEyesImproved();
		//Mat testEye2 = testEyes2[0];
		
		MatOfFloat cv32 = new MatOfFloat();
		MatOfFloat cv322 = new MatOfFloat();
		
		KNearest knn = KNearest.create();
		HOGDescriptor hogdescriptor = new HOGDescriptor();
		System.out.println("before " + cv32);
		
		hogdescriptor.compute(testEye, cv32);								// 3. COMPUTE HOGD FOR EYES
		hogdescriptor.compute(testEye2, cv322);
		
		
		
		Mat cv32mat = new Mat();
		Mat cv322mat = new Mat();
		
		
		//cv32.convertTo(cv32mat, CvType.CV_32F);								//-- 4. CONVERT TO CV_32F  skip
		//cv322.convertTo(cv322mat, CvType.CV_32F);
		


		
		System.out.println(testEye);
		System.out.println("after  " + cv32mat);
		
		//System.out.println(cv32.dump());
		//System.out.println(testEye.dump());
		
		//Mat responses = new Mat(1, 2, CvType.CV_32F);
		//responses.put(0, 0, 5);
		//responses.put(0, 1, 8);
		// 
		Mat responses = new Mat(1, 8, CvType.CV_32F);					// 5. CREATE RESPONSES
		responses.put(0, 0, 5);
		responses.put(0, 1, 6);
		responses.put(0, 2, 7);
		responses.put(0, 3, 8);
		responses.put(0, 4, 9);
		responses.put(0, 5, 1);
		responses.put(0, 6, 2);
		responses.put(0, 7, 3);

		System.out.println("responses " + responses.dump());
		
		
		//long startTime = System.nanoTime();
		//cv32.convertTo(cv32mat, CvType.CV_32F);
		//cv322.convertTo(cv322mat, CvType.CV_32F);
		//long endTime = System.nanoTime();
		//System.out.println("That took " + (endTime - startTime) + " milliseconds");

		long startTime = System.nanoTime();
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		long endTime = System.nanoTime();
		System.out.println("That took " + (endTime - startTime) + " milliseconds");
		
		//Core.hconcat(cv32mat, cv322mat);
		
		
		System.out.println("before transpose " + cv32.toString());
		//long startTime = System.nanoTime();
		Core.transpose(cv32, cv32);								// 6. TRANSPOSE HOGDs -FORCE
		Core.transpose(cv322, cv322);							// we transpose to use push_back because converting to a format that hconcat would accept takes 3 times more time than transposing
		//long endTime = System.nanoTime();						// hconcat does not support MatOfFloat
		//System.out.println("That took " + (endTime - startTime) + " milliseconds");
		System.out.println("after transpose " + cv32.toString());
		
		cv32.push_back(cv322);									// 7. MAKE INPUT FROM HOGDs
		cv32.push_back(cv322);
		cv32.push_back(cv322);
		cv32.push_back(cv322);
		cv32.push_back(cv322);
		cv32.push_back(cv322);
		cv32.push_back(cv322);
		System.out.println(cv32.toString());
		
		//Core.transpose(responses, responses);
		//System.out.println(responses.dump());
		
		System.out.println(cv32.type());
		
		knn.train(cv32, Ml.ROW_SAMPLE, responses);					// 8. MAKE KNN

		System.out.println(knn.isTrained());
		//System.out.println(knn.);
		
		
		
		Mat results = new Mat();
		Mat neighbours = new Mat();
		knn.findNearest(cv322, 3, results, neighbours);				// 9. DETECT
		System.out.println(neighbours.dump());
		System.out.println(neighbours.toString());
		//neighbours.convertTo(neighbours, );
		
		System.out.println((int)neighbours.get(0, 0)[0] + " " + (int)neighbours.get(0, 1)[0] + " " + (int)neighbours.get(0, 2)[0]);
	
		*/
		
		//CamData.debugScaledOnlyEyesImprovedView(480);
		//
		
		//CamData cddd = new CamData();
		//cddd.startRunning();
		//cddd.debugFaceEyesImprovedView();
		//HighGui.imshow("test", CamData.getFace());
		//HighGui.waitKey(0);
		//Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		
		//System.out.println(screenSize.height + " " + screenSize.width);
		

		//CamData.debugScaledOnlyEyesImprovedView(480);
		//System.out.println("rob");
		CamData robcd = new CamData();
		robcd.startRunning();
		robcd.debugFaceEyesImprovedViewRun();
		//robcd.debugGuessFaceEyesImprovedViewRun();
		//robcd.debugScaledOnlyEyesImprovedView(160);
		//CamData.debugFaceEyesImprovedView();
		//Mat rob = CamData.getMat();
		//Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		//System.out.println(screenSize.height + " " + screenSize.width);
		//Mat[] eyetestsize = CamData.getEyesImproved();
		//Imgproc.equalizeHist(eyetestsize[0], eyetestsize[0]);
		//HighGui.imshow("rob", eyetestsize[0]);
		//HighGui.waitKey(0);
		//robcd.startRunning();
		
		//while (true) {
			
		//	System.out.println(robcd.checkForEyes());
			//robcd.checkForEyes();
			
		//}
		
		//gproc.resize(rob, rob, new Size (screenSize.width, screenSize.height));
		
		
		
		//Imgproc.circle(rob, new Point(300, 300), 50, new Scalar(0, 0, 255));

		//HighGui.imshow("rob", rob);
		//HighGui.waitKey(0);
		
	}
	

	
	public static void main2(CamData cd) {
		System.out.println("successful tranbbbsefr");
		System.out.println(cd.getSize());
		
		Mat classes = new Mat(1, cd.getSize(), CvType.CV_32F);
		Mat workingeye = new Mat();
		MatOfFloat cv32 = new MatOfFloat();
		Mat eyemat = new Mat();
		HOGDescriptor hog = new HOGDescriptor();
		Mat samples = new Mat();
		
		
		for (int i = 0; i < cd.getSize(); i++) {
			System.out.println("working eye " + cd.getEye0(i));
			workingeye = cd.getEye0(i);
			Imgproc.resize(workingeye, workingeye, new Size(160,160));
			hog.compute(cd.getEye0(i), cv32);
			cv32.convertTo(eyemat, CvType.CV_32F);
			Core.transpose(eyemat, eyemat);
			//System.out.println("before adding to smples: " + samples.toString());
			samples.push_back(eyemat);
			//System.out.println("after adding to smples: " +samples.toString());
			classes.put(0, i, i);
			
		}
		
		System.out.println(samples.toString());
		System.out.println(classes.dump());
		
		KNearest knn = KNearest.create();
		knn.train(samples, Ml.ROW_SAMPLE, classes);
		System.out.println(knn.isTrained());
		
		//testing now
		Mat testeye = new Mat();
		Mat newEye = new Mat();
		MatOfFloat testingsample = new MatOfFloat();
		testeye = CamData.getEyesImproved()[0];
		Imgproc.resize(testeye, testeye, new Size(160,160));
		hog.compute(testeye, testingsample);
		testingsample.convertTo(newEye, CvType.CV_32F);
		Core.transpose(newEye, newEye);
		
		Mat useless = new Mat();
		Mat neighbours = new Mat();
		
		
		knn.findNearest(newEye, 3, useless, neighbours);
		System.out.println(neighbours.dump());

		
		
		/**
		
		System.out.println("adsas");
		CamData roger = new CamData();
		roger.startRunning();
		
		
		int i = 1;
		while (i < 50) {
		Mat[] rogereyes = roger.getEyesImprovedRun();
		Imgproc.resize(rogereyes[0], rogereyes[0], new Size(480, 480));
		Imgproc.resize(rogereyes[1], rogereyes[1], new Size(480, 480));
		rogereyes[0].push_back(rogereyes[1]);
		

		HighGui.imshow("eye1", rogereyes[0]);
		HighGui.waitKey(20);
		i++;
		}
		
		roger.stopRunning();
		
		*///
		
	}
	public static void main3(CamData robcd) {
		System.out.println("successful transfer to main3");
		System.out.println(robcd.initializeKNearest(160));
		System.out.println(robcd.getSize());
		System.out.println("rob");
		//CamData robcd = new CamData();
		robcd.startRunning();
		robcd.debugGuessFaceEyesImprovedViewRun();
		//while (true) {
		//System.out.println(Arrays.toString(cd.getGuessRun(160)));
		//HighGui.imshow("rob", CamData.debugMatFaceEyesImprovedView());
		//HighGui.waitKey(20);
		//}
		//cd.debugGuessFaceEyesImprovedView();
		/*
		cd.startRunning();
		for (int i = 0; i < cd.getSize(); i++) { 
		Arrays.toString(cd.getCoords(i));
		}
		System.out.println(Arrays.toString(cd.getGuessKnn3(160)));
		//cd.startRunning();
		 */
		//cd.debugFaceEyesImprovedView();
		
		//CamData.debugScaledOnlyEyesImprovedView(480);
		
		/*
		
		System.out.println("rob");
		Mat rob = CamData.getMat();
		Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		
		Imgproc.resize(rob, rob, new Size (screenSize.width, screenSize.height));
		
		
		
		Imgproc.circle(rob, new Point(300, 300), 50, new Scalar(0, 0, 255));

		HighGui.imshow("rob", rob);
		
		
		HighGui.waitKey(0);
		
		//System.out.println("starting to guess");
		
		//while (true) {
		//System.out.println("X COORD:   " + cd.getGuessRun(160)[0]);
		//cd.debugGuessFaceEyesImprovedView();
		//System.out.println("Y COORD:   " + cd.getGuessRun(160)[1]);
		//}
		
		
		//cd.stopRunning();
		*/
		
	}

}
