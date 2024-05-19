import java.awt.Dimension;
import java.awt.Toolkit;
import java.util.Arrays;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.videoio.VideoCapture;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;
import org.opencv.core.MatOfFloat;

/**
 * This class contains most of the Image Extraction files. 
 * 
 */

public class CamData {
	
	
	/**
	 * knn instance of the CamData instance
	 * used to Estimate the Gaze 
	 * after training it with Calibration Data
	 * One instance for each eye – knn0 and knn1
	 * for eyes 0 and 1
	 */
	public KNearest knn0;
	public KNearest knn1;
	
	/**
	 * caprun is the videocapture for the dynamic class
	 */
	public VideoCapture caprun;
	
	/**
	 * eyeSamples contains an array with our sample eye set from calibration
	 * the Coordinates can be accessed using the array id for each eyes
	 * contains eye 0
	 */
	public Mat[] eye0Samples;
	
	/**
	 * eyeSamples contains an array with our sample eye set from calibration
	 * the Coordinates can be accessed using the array id for each eyes
	 * contains eye 1
	 */
	
	public Mat[] eye1Samples;
	
	/**
	 * The xCoords array has the x Coords for each eye with the same array id
	 * 
	 */
	public int[] xCoords;
	
	/**
	 * The yCoords array has the y Coords for each eye with the same array id
	 */
	public int[] yCoords;
	
	/**
	 * Counts the number (length) of entries in our database. 
	 * Empty = 0
	 */
	private int pCounter; 
	
	/**
	 * CamData class instance constructor
	 * The sample size will be increased with each successful .save() call
	 * Initialized with size 0
	 */	
	public CamData() {
		eye0Samples = new Mat[0];
		eye1Samples = new Mat[0];
		xCoords = new int[0];
		yCoords = new int[0];
		pCounter = 0;
	}
	
	/**
	 * CamData class instance constructor with size (preferred)
	 * @param size of the data set (number of points)
	 * If sample size will be more than data set it will be increased with each successful .save() call
	 */
	public CamData(int size) {
		eye0Samples = new Mat[size];
		eye1Samples = new Mat[size];
		xCoords = new int[size];
		yCoords = new int[size];
		pCounter = 0;
	}
	
	//save given coords into sample with captured eye image 
	
	/**
	 * Shows size (LENGTH) of sample dataset
	 * @return int size – number of samples so far
	 */
	
	public int getSize() {
		return pCounter;
	}
	
	
	/**
	 * saves the given coordinates with an eye .Mat into the "database" together
	 * the x coord is in the 0th element of the array
	 * the y coord is in the 1st element of the array
	 */
	boolean save(int[] coords) {
		
		if ((pCounter+1) > xCoords.length)			//can be replaced with while or while((pCounter == length))
			enlSize();								//make space for one more record
		
		Mat[] temp = new Mat[2];					//declare new temporary variable
		temp = getEyesImproved();							//save eyes into temporary variable
		if (temp == null)							//check if eyes can be detected
			return false;							//feedback if no eyes for too long
		
		//save all data into the instance variables after eye detection, to avoid problems in case of false
		eye0Samples[pCounter] = temp[0];			//save eye 0
		eye1Samples[pCounter] = temp[1];			//save eye 1
		xCoords[pCounter] = coords[0];				//save x coord
		yCoords[pCounter] = coords[1];				//save y coord

		
		pCounter++;									//increase the Sample Size Counter
		
		return true;								//feedback – saving is a success
		
	}
	
	/**
	 * saves the given coordinates with a captured eye pair
	 * Same as save but with the running eye saving system which is much faster
	 * requires startRunning before and stopRunning after 
	 * @param coords
	 * @return returns true if success, false if something went wrong
	 */
	
	boolean saveRun(int[] coords) {
		
		if ((pCounter+1) > xCoords.length)			//can be replaced with while or while((pCounter == length))
			enlSize();								//make space for one more record
		
		Mat[] temp = new Mat[2];					//declare new temporary variable
		temp = this.getEyesImprovedRun();							//save eyes into temporary variable
		if (temp == null)							//check if eyes can be detected
			return false;							//feedback if no eyes for too long
		
		//save all data into the instance variables after eye detection, to avoid problems in case of false
		eye0Samples[pCounter] = temp[0];			//save eye 0
		eye1Samples[pCounter] = temp[1];			//save eye 1
		xCoords[pCounter] = coords[0];				//save x coord
		yCoords[pCounter] = coords[1];				//save y coord

		
		pCounter++;									//increase the Sample Size Counter
		
		return true;								//feedback – saving is a success
		
	}
	
	//enlarge arrays == sample size by one
	
	/**
	 * increases the size of our x,y,eye arrays by one 
	 */
	private void enlSize() {
		
		int[] xNewArray = new int[xCoords.length+1];			//new x array, one element longer
		int[] yNewArray = new int[yCoords.length+1];			//new y array, one element longer
		Mat[] eye0NewArray = new Mat[eye0Samples.length+1];		//new Mat 0 array, one element longer
		Mat[] eye1NewArray = new Mat[eye1Samples.length+1];		//new Mat 1 array, one element longer
		for (int i = 0; i < xCoords.length; i++ ) {				//for loop to copy all elements from old array to new
			xNewArray[i] = xCoords[i]; 							//copy old array x coord to new x array 
			yNewArray[i] = yCoords[i];							//copy old array y coord to new y array 
			eye0NewArray[i] = eye0Samples[i];						//copy old array eye 0 Mat to new eye array 
			eye1NewArray[i] = eye1Samples[i];						//copy old array eye 1 Mat to new eye array
		}
		
		xCoords = xNewArray;
		yCoords = yNewArray;
		eye0Samples = eye0NewArray;
		eye1Samples = eye1NewArray;
	}
	
	
//instance get methods:
	
	
	//get x coordinate
	/**
	 * returns the x coordinate value of a given sample point ID
	 * The point IDs are the IDs of the calibration points saved
	 * @param sample ID
	 * @return x value coordinate
	 */
	int getX(int id) {
		if (id < pCounter)
			return xCoords[id];
		return 0;
	}
	
	//get y coordinate
	/**
	 * returns the y coordinate value of a given sample ID
	 * the sample ID are the IDs of the calibration points
	 * @param sample ID
	 * @return y value coordinate
	 */
	int getY(int id) {
		if (id < pCounter)
			return yCoords[id];
		return 0;
	}
	
	//get eye 0
	/**
	 * returns the eye0 Mat variable of a given sample ID
	 * @param sample ID
	 * @return eye0 Mat value
	 */
	Mat getEye0(int id) {
		if (id < pCounter)
			return eye0Samples[id];
		return null;
	}
	
	//get eye 1
	/**
	 * returns the eye1 Mat variable of a given sample ID
	 * @param sample ID
	 * @return eye1 Mat value
	 */
	Mat getEye1(int id) {
		if (id < pCounter)
			return eye1Samples[id];
		return null;
	}
	
	
	/**
	 * Opens VideoCapture device for capture
	 * Usually is required for non-static methods with "Run"
	 * improves capturing speed immensely
	 * Required for getEyesImprovedRun() 
	 * Recommended for multiple eye captures
	 * Needs to be closed after capture with .stopRunnint()
	 */
	public void startRunning() {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
		caprun = new VideoCapture(0);
		
	}
	
	
	/**
	 * Closes VideoCapture device for capture
	 * Does not really have to be used but can close the capture device
	 * Usually used after getEyesImprovedRun() loop done
	 */
	public void stopRunning() {
		caprun.release();
	}
	
	
	
	
	//get eye 0 and eye 1 in an Mat[2] array
	/**
	 * returns the Mat[] eyes of a given sample ID
	 * non-static method that works only with the instance
	 * don't mistake with getEyes()
	 * @param sample ID
	 * @return eyes Mat[] value (eye 0 in [0], eye 1 in [1])
	 * returns null if ID not present
	 */
	Mat[] getEyes(int id) {
		if (id < pCounter) 
			return new Mat[]{eye0Samples[id], eye1Samples[id]};
		return null;
	}
	
	//get x coordinate and y coordinate in an int[2] array
	/**
	 * returns the int[] array with the x and y coords of given sample ID
	 * @param sample ID
	 * @return coordinates, x in [0], y in [1]; 
	 * returns null if ID not present
	 */
	int[] getCoords(int id) {
//		System.out.println(id + ":id and pCounter:" + pCounter);    //debug
		if (id < pCounter)
			return new int[]{xCoords[id], yCoords[id]};
		return null;
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

/**
 * The static getMat method is supposed to return the Mat variable with the WebCam image data.	
 * Could be made private in the future.
 * Could be made more functional with the addition of another overloading method with a param, in order to, for example, show the image or send image information.
 * @return a Mat type variable with the webcam data
 */
	
	public static Mat getMat() {
		
		VideoCapture cap;
		Mat image;
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
		
		cap = new VideoCapture(0);
		
		if (!cap.isOpened())
            System.out.println("Camera Error");
		
		image = new Mat();
		cap.grab();
		cap.retrieve(image);
		
		return image;
		
	}
	
	
	
	
	
/**
 * Returns a face if detected immediately, otherwise returns null
 * This static function gets an image from the web cam, and returns a face in grayscale. 
 * This function captures an image from the web cam
 * @return one detected Mat::face from web cam in grayscale if detected, otherwise null
 */
	
	public static Mat getFaceNow() {
		
		/*
		 * This part is literally copied from getMat function, it gets the cam Mat data type variable in Mat::image
		 */
		
		VideoCapture cap;
		Mat image;
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
		
		cap = new VideoCapture(0);
		
		if (!cap.isOpened())
            System.out.println("Camera Error");
		
		image = new Mat();
		cap.grab();
		cap.retrieve(image);
		
		/*
		 * This part detects the face
		 */
		
		String filename = "lbpcascade_frontalface_improved.xml"; 
		Size dsize = new Size(50,50);
		MatOfRect rect = new MatOfRect();
		Rect[] arrayrects = null;
		
		
		
		
		CascadeClassifier faceCC = new CascadeClassifier(filename);				//Load CascadeClassifier
		if (faceCC.empty()) 													//Check for errors
			System.out.println("CascadeClassifier didn't load");
		
		Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2GRAY);					//Convert our Cam Mat in GrayScale
		
		faceCC.detectMultiScale(image, rect, 2, 2, 0, dsize);					//Detect the face and save the area in MatOfRect::rect
		arrayrects = rect.toArray();
		
		if (arrayrects.length == 0)
			return null;
		
		System.out.println("not null");
		
		image = new Mat(image, arrayrects[0]);
		return image;
			
		
		
		
	}
	
	
	/**
	 * Returns a Mat::face WHEN it is detected, otherwise loops until there is a face;
	 * This static function gets an image from the web cam, and returns a face in grayscale. 
	 * This function captures an image from the web cam
	 * @return one detected Mat::face from web cam in grayscale when detected.
	 */
	
	public static Mat getFace() {
		
		
		VideoCapture cap;
		Mat image;
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
		
		cap = new VideoCapture(0);
		
		if (!cap.isOpened())
            System.out.println("Camera Error");
		
		image = new Mat();
		
		
		/*
		 * This part detects the face
		 */
		
		String filename = "lbpcascade_frontalface_improved.xml"; 
		Size dsize = new Size(50,50);
		MatOfRect rect = new MatOfRect();
		Rect[] arrayrects = null;
		boolean detected = false;
		CascadeClassifier faceCC = new CascadeClassifier(filename);				//Load CascadeClassifier
		if (faceCC.empty()) 													//Check for errors
			System.out.println("CascadeClassifier didn't load");
		
		
		while(!detected) {
		
		cap.grab();
		cap.retrieve(image);

		Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2GRAY);					//Convert our Cam Mat in GrayScale
		
		faceCC.detectMultiScale(image, rect, 2, 2, 0, dsize);					//Detect the face and save the area in MatOfRect::rect
		arrayrects = rect.toArray();
		
		if (arrayrects.length != 0)
			detected = true;
		}
		
		
		return new Mat(image, arrayrects[0]);
			
		
		
		
	}
	
	
	/**
	 * TODO: make this more usable
	 * This static method returns two eyes from the cam from the face in a Mat array of size 2. It keeps capturing until it gets two eyes;
	 * This method keeps capturing cam pictures until it detects a face, then it tries to find 2 or more eyes, if it doesn't, it asks for the next face. 
	 * When it finds the two or more eyes on the n-th face Mat it returns the eyes in an Array
	 * @return Mat[2]::eyes
	 */
	
	public static Mat[] getEyes() {
		
		
		VideoCapture cap;
		Mat image;
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
		
		/*
		 * This part captures an image from the web cam
		 */
		cap = new VideoCapture(0);
		if (!cap.isOpened())
            System.out.println("Camera Error");
		image = new Mat();

		
		
		/*
		 * This part detects the face
		 */
		
		String filename = "lbpcascade_frontalface_improved.xml";			//lbp cascade file location
		String filenameeye = "haarcascade_eye.xml";							//haar cascade file location
		Size dsize = new Size(50,50);										
		MatOfRect rect = new MatOfRect();
		Mat face = new Mat();
		Rect[] arrayrects = null;
		Rect temp = null;
		Mat[] returneyes = new Mat[2];
		boolean detected = false;
		int iterationCounter = 0;
		
		CascadeClassifier faceCC = new CascadeClassifier(filename);				//Load CascadeClassifier
		if (faceCC.empty()) 													//Check for errors
			System.out.println("CascadeClassifier FACE didn't load");
		CascadeClassifier eyeCC = new CascadeClassifier(filenameeye);
		if (eyeCC.empty()) 													//Check for errors
			System.out.println("CascadeClassifier EYE didn't load");
			
		
		while(!detected) {
		
		if (iterationCounter > 500) 
			return null;
			
		cap.grab();
		cap.retrieve(image);

		Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2GRAY);					//Convert our Cam Mat in GrayScale
		
		faceCC.detectMultiScale(image, rect, 2, 2, 0, dsize);					//Detect the face and save the area in MatOfRect::rect
		arrayrects = rect.toArray();
		iterationCounter++;
		

		
		/*
		 * This part detects eyes
		 */
		if (arrayrects.length != 0)
			face = new Mat(image, arrayrects[0]);									//Crop the ROI
			eyeCC.detectMultiScale(face, rect, 1.1, 11);
			arrayrects = rect.toArray();
			if (	arrayrects.length > 1 && 
					((arrayrects[0].y + 10) > arrayrects[1].y) &&
					((arrayrects[1].y + 10) > arrayrects[0].y) ) { 		//make sure the eyes are different (%) and that eye 0 is always on one side
					if 	(arrayrects[0].x > arrayrects[1].x) {	//make sure we always have the same left and right eye
						temp = arrayrects[0];
						arrayrects[0] = arrayrects[1]; 
						arrayrects[1] = temp;
					}
					detected = true;
					cap.release();
			}
			iterationCounter++;
		
		}
		
		/*
		 * This part saves detected eyes and returns them as Mat[2]
		 */
		returneyes[0] = new Mat(face, arrayrects[0]);
		returneyes[1] = new Mat(face, arrayrects[1]);

		
		return returneyes;
		
		
	}
	
	
	
	
	
	
	
	
	
	/**
	 * this is the currently used method for extracting eyes
	 * This static method returns two eyes from the cam from the face in a Mat array of size 2. It keeps capturing until it gets two eyes;
	 * This method keeps capturing cam pictures until it detects a face, then it removes the lower half of the picture, divides the upper two parts in two, 
	 * searches each half of the upper half for one eye, then returns two eyes if two eyes are found in a Mat[] array
	 * 
	 * IF IT DOES NOT FIND EYES IN 500 ITERATIONS IT RETURNS NULL because of requirement
	 * @return Mat[2]::eyes
	 */
	
	public static Mat[] getEyesImproved() {
		
		
		VideoCapture cap;
		Mat image;
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
		
		/*
		 * This part captures an image from the web cam
		 */
		cap = new VideoCapture(0);
		if (!cap.isOpened())
            System.out.println("Camera Error");
		image = new Mat();
		
		
		/*
		 * This part detects the face
		 */
		
		String filename = "lbpcascade_frontalface_improved.xml";			//lbp cascade file location
		String filenameeye = "haarcascade_eye.xml";							//haar cascade file location
		Size dsize = new Size(50,50);										
		MatOfRect rect = new MatOfRect();
		Rect[] arrayrects = null;
		Mat[] returneyes = new Mat[2];
		boolean detected = false;
		int iterationCounter = 0;
		Rect eye0Rect = null;
		Rect eye1Rect = null;
		Mat eye0Mat = null;								//Crop the ROI
		Mat eye1Mat = null;
		MatOfRect eye0MatOfRect = new MatOfRect();
		MatOfRect eye1MatOfRect = new MatOfRect();
		Rect[] eye0RectsArray = null;
		Rect[] eye1RectsArray = null;

		
		CascadeClassifier faceCC = new CascadeClassifier(filename);				//Load CascadeClassifier
		if (faceCC.empty()) 													//Check for errors
			System.out.println("CascadeClassifier FACE didn't load");
		CascadeClassifier eyeCC = new CascadeClassifier(filenameeye);
		if (eyeCC.empty()) 													//Check for errors
			System.out.println("CascadeClassifier EYE didn't load");
			
		
		while(!detected) {
		
		if (iterationCounter > 500) 
			return null;
			
		cap.grab();
		cap.retrieve(image);

		Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2GRAY);					//Convert our Cam Mat in GrayScale
		
		faceCC.detectMultiScale(image, rect, 2, 2, 0, dsize);					//Detect the face and save the area in MatOfRect::rect
		arrayrects = rect.toArray();
		iterationCounter++;
		

		
		/*
		 * This part detects eyes
		 */
		if (arrayrects.length != 0)  {
			
			eye0Rect = new Rect(arrayrects[0].x, arrayrects[0].y, (arrayrects[0].width / 2), (arrayrects[0].height / 2));
			eye1Rect = new Rect(arrayrects[0].x + (arrayrects[0].width / 2), arrayrects[0].y, (arrayrects[0].width / 2), (arrayrects[0].height / 2));

			eye0Mat = new Mat(image, eye0Rect);									//Crop the ROI
			eye1Mat = new Mat(image, eye1Rect);

			eyeCC.detectMultiScale(eye0Mat, eye0MatOfRect, 1.1, 8);
			eyeCC.detectMultiScale(eye1Mat, eye1MatOfRect, 1.1, 8);
			eye0RectsArray = eye0MatOfRect.toArray();
			eye1RectsArray = eye1MatOfRect.toArray();
			System.out.println(eye0RectsArray.length + " " + eye1RectsArray.length);
			if ((eye0RectsArray.length > 0) && (eye1RectsArray.length > 0)) 
				detected = true;
			}
			iterationCounter++;
		
		}
		
		/*
		 * This part saves detected eyes and returns them as Mat[2]
		 */
		returneyes[0] = new Mat(eye0Mat, eye0RectsArray[0]);
		returneyes[1] = new Mat(eye1Mat, eye1RectsArray[0]);
		cap.release();
		
		return returneyes;
		
		
	}


	
	
	/**
	 * same as the static method getEyesImproved but has to be run from a non-static instance 
	 * IF IT DOES NOT FIND EYES IN 50000 ITERATIONS IT RETURNS NULL because of requirement
	 * @return returns a mat array of eyes MIGHT RETURN NULL BE CAREFUL
	 */
	
	public Mat[] getEyesImprovedRun() {
		
		
		
		Mat image;
		//System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
		
		/*
		 * This part captures an image from the web cam
		 */
		
		if (!caprun.isOpened())
            System.out.println("Camera Error");
		image = new Mat();
		
		
		/*
		 * This part detects the face
		 */
		
		String filename = "lbpcascade_frontalface_improved.xml";			//lbp cascade file location
		String filenameeye = "haarcascade_eye.xml";							//haar cascade file location
		Size dsize = new Size(50,50);										
		MatOfRect rect = new MatOfRect();
		Rect[] arrayrects = null;
		Mat[] returneyes = new Mat[2];
		boolean detected = false;
		int iterationCounter = 0;
		Rect eye0Rect = null;
		Rect eye1Rect = null;
		Mat eye0Mat = null;								//Crop the ROI
		Mat eye1Mat = null;
		MatOfRect eye0MatOfRect = new MatOfRect();
		MatOfRect eye1MatOfRect = new MatOfRect();
		Rect[] eye0RectsArray = null;
		Rect[] eye1RectsArray = null;

		
		CascadeClassifier faceCC = new CascadeClassifier(filename);				//Load CascadeClassifier
		if (faceCC.empty()) 													//Check for errors
			System.out.println("CascadeClassifier FACE didn't load");
		CascadeClassifier eyeCC = new CascadeClassifier(filenameeye);
		if (eyeCC.empty()) 													//Check for errors
			System.out.println("CascadeClassifier EYE didn't load");
			
		
		while(!detected) {
		
		if (iterationCounter > 50000) 
			return null;
			
		caprun.grab();
		caprun.retrieve(image);

		Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2GRAY);					//Convert our Cam Mat in GrayScale
		
		faceCC.detectMultiScale(image, rect, 2, 2, 0, dsize);					//Detect the face and save the area in MatOfRect::rect
		arrayrects = rect.toArray();
		iterationCounter++;
		

		
		/*
		 * This part detects eyes
		 */
		if (arrayrects.length != 0)  {
			
			eye0Rect = new Rect(arrayrects[0].x, arrayrects[0].y, (arrayrects[0].width / 2), (arrayrects[0].height / 2));
			eye1Rect = new Rect(arrayrects[0].x + (arrayrects[0].width / 2), arrayrects[0].y, (arrayrects[0].width / 2), (arrayrects[0].height / 2));

			eye0Mat = new Mat(image, eye0Rect);									//Crop the ROI
			eye1Mat = new Mat(image, eye1Rect);

			eyeCC.detectMultiScale(eye0Mat, eye0MatOfRect, 1.1, 8);
			eyeCC.detectMultiScale(eye1Mat, eye1MatOfRect, 1.1, 8);
			eye0RectsArray = eye0MatOfRect.toArray();
			eye1RectsArray = eye1MatOfRect.toArray();
			//System.out.println(eye0RectsArray.length + " " + eye1RectsArray.length);
			if ((eye0RectsArray.length > 0) && (eye1RectsArray.length > 0)) 
				detected = true;
			}
			iterationCounter++;
		
		}
		
		/*
		 * This part saves detected eyes and returns them as Mat[2]
		 */
		returneyes[0] = new Mat(eye0Mat, eye0RectsArray[0]);
		returneyes[1] = new Mat(eye1Mat, eye1RectsArray[0]);
		
		Imgproc.equalizeHist(returneyes[0], returneyes[0]);
		Imgproc.equalizeHist(returneyes[1], returneyes[1]);
		
		
		return returneyes;
	
		
		
	}

	
	
	
	
	
	
	
	
	
	
	
	/**
	 * This shows only the web cam image in a window until something happens
	 */
	public static void debugCamView() {
		
		VideoCapture cap;
		Mat image;
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
		
		while (true) {
		cap = new VideoCapture(0);
		
		if (!cap.isOpened())
            System.out.println("Camera Error");
		
		image = new Mat();
		cap.grab();
		cap.retrieve(image);
		
		HighGui.imshow("CamView",image);
		HighGui.waitKey(20);
		}
		
	}
	
	
	/**
	 * Shows the Camera View with the Detected face marked with a square, if detected
	 */
	public static void debugFaceView() {
		

		VideoCapture cap;
		Mat image;
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
		
		cap = new VideoCapture(0);
		
		if (!cap.isOpened())
            System.out.println("Camera Error");
		
		image = new Mat();
		Mat imageGray = new Mat();
		
		
		/*
		 * This part detects the face
		 */
		
		String filename = "lbpcascade_frontalface_improved.xml"; 
		Size dsize = new Size(50,50);
		MatOfRect rect = new MatOfRect();
		Rect[] arrayrects = null;
		CascadeClassifier faceCC = new CascadeClassifier(filename);				//Load CascadeClassifier
		if (faceCC.empty()) 													//Check for errors
			System.out.println("CascadeClassifier didn't load");
		
		
		while(true) {
		
		cap.grab();
		cap.retrieve(image);

		Imgproc.cvtColor(image, imageGray, Imgproc.COLOR_BGR2GRAY);					//Convert our Cam Mat in GrayScale
		
		faceCC.detectMultiScale(imageGray, rect, 2, 3, 0, dsize);					//Detect the face and save the area in MatOfRect::rect
		arrayrects = rect.toArray();
		
		if (arrayrects.length != 0)
			Imgproc.rectangle(image, arrayrects[0], new Scalar(255));
		
		
		HighGui.imshow("CamView",image);
		HighGui.waitKey(20);
		}	
		
	}

	
	
	
	
	
	/**
	 * Shows the Camera view with detected face if detected and detected eyes if detected
	 * uses the same parameters as .getEyes, made for debugging and etc
	 */
	public static void debugFaceEyesView() {
		

		VideoCapture cap;
		Mat image;
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
		
		cap = new VideoCapture(0);
		
		if (!cap.isOpened())
            System.out.println("Camera Error");
		
		image = new Mat();
		Mat imageGray = new Mat();
		Mat faceMat = new Mat();
		MatOfRect eyesMatRect = new MatOfRect();
		
		/*
		 * This part detects the face
		 */
		
		String filename = "lbpcascade_frontalface_improved.xml"; 
		String eyesfilename = "haarcascade_eye.xml";
		Size dsize = new Size(50,50);
		MatOfRect rect = new MatOfRect();
		Rect[] arrayrects = null;
		Rect[] eyesrects = null;
		Rect temp = null;
		CascadeClassifier faceCC = new CascadeClassifier(filename);				//Load CascadeClassifier
		CascadeClassifier eyesCC = new CascadeClassifier(eyesfilename);
		if (faceCC.empty()) 													//Check for errors
			System.out.println("CascadeClassifier didn't load");
		
		
		while(true) {
		
		cap.grab();
		cap.retrieve(image);

		Imgproc.cvtColor(image, imageGray, Imgproc.COLOR_BGR2GRAY);					//Convert our Cam Mat in GrayScale
		
		faceCC.detectMultiScale(imageGray, rect, 2, 2, 0, dsize);					//Detect the face and save the area in MatOfRect::rect
		arrayrects = rect.toArray();
		
		if (arrayrects.length != 0) {
			Imgproc.rectangle(image, arrayrects[0], new Scalar(255));
			faceMat = new Mat(imageGray, arrayrects[0]);
			eyesCC.detectMultiScale(faceMat, eyesMatRect, 1.1, 11);
			eyesrects = eyesMatRect.toArray();
			System.out.println(eyesrects.length);
			if (	eyesrects.length > 1 &&
					((eyesrects[0].y + 10) > eyesrects[1].y) &&
					((eyesrects[1].y + 10) > eyesrects[0].y) 
					) { 		//make sure the eyes are different (%) and that eye 0 is always on one side
				if 	(eyesrects[0].x > eyesrects[1].x) {	//make sure we always have the same left and right eye
					temp = eyesrects[0];
					eyesrects[0] = eyesrects[1]; 
					eyesrects[1] = temp;
				}
				eyesrects[0].x = eyesrects[0].x + arrayrects[0].x;
				eyesrects[0].y = eyesrects[0].y + arrayrects[0].y;
				eyesrects[1].x = eyesrects[1].x + arrayrects[0].x;
				eyesrects[1].y = eyesrects[1].y + arrayrects[0].y;
				Imgproc.rectangle(image, eyesrects[0], new Scalar(255));
				Imgproc.rectangle(image, eyesrects[1], new Scalar(255));
			}
			
			
			
		}
		
		HighGui.imshow("CamView",image);
		HighGui.waitKey(20);
		}	
		
	}
	
	
	/**
	 * Shows the Camera view with detected face if detected and detected eyes (improved) if detected
	 * uses the same parameters as .getEyesImproved, made for debugging and etc
	 */
	public static void debugFaceEyesImprovedView() {
		

		VideoCapture cap1; 
		//cap1 = new VideoCapture(0);
		
		
		Mat image;
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
		
		cap1 = new VideoCapture(0);
		
		if (!cap1.isOpened())
            System.out.println("Camera Error");
		
		image = new Mat();
		Mat imageGray = new Mat();
		Mat faceMat = new Mat();
		MatOfRect eyesMatRect = new MatOfRect();
		
		/*
		 * This part detects the face
		 */
		
		String filename = "lbpcascade_frontalface_improved.xml"; 
		String eyesfilename = "haarcascade_eye.xml";
		Size dsize = new Size(50,50);
		MatOfRect rect = new MatOfRect();
		Rect[] arrayrects = null;
		Rect eye0Rect = new Rect();
		Rect eye1Rect = new Rect();
		Mat eye0Mat = new Mat();
		Mat eye1Mat = new Mat();
		MatOfRect eye0MatOfRect = new MatOfRect();
		MatOfRect eye1MatOfRect = new MatOfRect();
		Rect[] eye0RectsArray = null;
		Rect[] eye1RectsArray = null;
 		
		CascadeClassifier faceCC = new CascadeClassifier(filename);				//Load CascadeClassifier
		CascadeClassifier eyesCC = new CascadeClassifier(eyesfilename);
		if (faceCC.empty()) 													//Check for errors
			System.out.println("CascadeClassifier didn't load");
		
		
		while(true) {
		
		cap1.grab();
		cap1.retrieve(image);

		Imgproc.cvtColor(image, imageGray, Imgproc.COLOR_BGR2GRAY);					//Convert our Cam Mat in GrayScale
		
		faceCC.detectMultiScale(imageGray, rect, 2, 2, 0, dsize);					//Detect the face and save the area in MatOfRect::rect
		arrayrects = rect.toArray();
		System.out.println("Faces found: " + arrayrects.length);
		if (arrayrects.length != 0)  {
			Imgproc.rectangle(image, arrayrects[0], new Scalar(255));
			eye0Rect = new Rect(arrayrects[0].x, arrayrects[0].y, (arrayrects[0].width / 2), (arrayrects[0].height / 2));
			eye1Rect = new Rect(arrayrects[0].x + (arrayrects[0].width / 2), arrayrects[0].y, (arrayrects[0].width / 2), (arrayrects[0].height / 2));

			eye0Mat = new Mat(image, eye0Rect);									//Crop the ROI
			eye1Mat = new Mat(image, eye1Rect);
			
			eyesCC.detectMultiScale(eye0Mat, eye0MatOfRect, 1.1, 8);
			eyesCC.detectMultiScale(eye1Mat, eye1MatOfRect, 1.1, 8);
			eye0RectsArray = eye0MatOfRect.toArray();
			eye1RectsArray = eye1MatOfRect.toArray();
			System.out.println("eye 0: " + eye0RectsArray.length);
			System.out.println("eye 1; " + eye1RectsArray.length);
			
			if ((eye0RectsArray.length > 0) && (eye1RectsArray.length > 0)) {				
				eye0RectsArray[0].x = eye0RectsArray[0].x + arrayrects[0].x;
				eye0RectsArray[0].y = eye0RectsArray[0].y + arrayrects[0].y;
				eye1RectsArray[0].x = eye1RectsArray[0].x + eye1Rect.x;
				eye1RectsArray[0].y = eye1RectsArray[0].y + arrayrects[0].y;
				Imgproc.rectangle(image, eye0RectsArray[0], new Scalar(255));
				Imgproc.rectangle(image, eye1RectsArray[0], new Scalar(255));
			}
			
			
			
		}
		
		HighGui.imshow("CamView",image);
		HighGui.waitKey(20);
		}	
		
	}
	
	
	/**
	 * same as static debugFaceEyesImprovedView but with run (requires startRunning()) and is non-static
	 * Shows the Camera view with detected face if detected and detected eyes (improved) if detected
	 * uses the same parameters as .getEyesImproved, made for debugging and etc
	 */
	public void debugFaceEyesImprovedViewRun() {
		

		//VideoCapture cap1; 
		//cap1 = new VideoCapture(0);
		
		
		Mat image;
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
		
		caprun = new VideoCapture(0);
		
		if (!caprun.isOpened())
            System.out.println("Camera Error");
		
		image = new Mat();
		Mat imageGray = new Mat();
		Mat faceMat = new Mat();
		MatOfRect eyesMatRect = new MatOfRect();
		
		/*
		 * This part detects the face
		 */
		
		String filename = "lbpcascade_frontalface_improved.xml"; 
		String eyesfilename = "haarcascade_eye.xml";
		Size dsize = new Size(50,50);
		MatOfRect rect = new MatOfRect();
		Rect[] arrayrects = null;
		Rect eye0Rect = new Rect();
		Rect eye1Rect = new Rect();
		Mat eye0Mat = new Mat();
		Mat eye1Mat = new Mat();
		MatOfRect eye0MatOfRect = new MatOfRect();
		MatOfRect eye1MatOfRect = new MatOfRect();
		Rect[] eye0RectsArray = null;
		Rect[] eye1RectsArray = null;
 		
		CascadeClassifier faceCC = new CascadeClassifier(filename);				//Load CascadeClassifier
		CascadeClassifier eyesCC = new CascadeClassifier(eyesfilename);
		if (faceCC.empty()) 													//Check for errors
			System.out.println("CascadeClassifier didn't load");
		
		
		while(true) {
		
		caprun.grab();
		caprun.retrieve(image);

		Imgproc.cvtColor(image, imageGray, Imgproc.COLOR_BGR2GRAY);					//Convert our Cam Mat in GrayScale
		
		faceCC.detectMultiScale(imageGray, rect, 2, 2, 0, dsize);					//Detect the face and save the area in MatOfRect::rect
		arrayrects = rect.toArray();
		System.out.println("Faces found: " + arrayrects.length);
		if (arrayrects.length != 0)  {
			Imgproc.rectangle(image, arrayrects[0], new Scalar(255));
			eye0Rect = new Rect(arrayrects[0].x, arrayrects[0].y, (arrayrects[0].width / 2), (arrayrects[0].height / 2));
			eye1Rect = new Rect(arrayrects[0].x + (arrayrects[0].width / 2), arrayrects[0].y, (arrayrects[0].width / 2), (arrayrects[0].height / 2));

			eye0Mat = new Mat(image, eye0Rect);									//Crop the ROI
			eye1Mat = new Mat(image, eye1Rect);
			
			eyesCC.detectMultiScale(eye0Mat, eye0MatOfRect, 1.1, 8);
			eyesCC.detectMultiScale(eye1Mat, eye1MatOfRect, 1.1, 8);
			eye0RectsArray = eye0MatOfRect.toArray();
			eye1RectsArray = eye1MatOfRect.toArray();
			System.out.println("eye 0: " + eye0RectsArray.length);
			System.out.println("eye 1; " + eye1RectsArray.length);
			
			if ((eye0RectsArray.length > 0) && (eye1RectsArray.length > 0)) {				
				eye0RectsArray[0].x = eye0RectsArray[0].x + arrayrects[0].x;
				eye0RectsArray[0].y = eye0RectsArray[0].y + arrayrects[0].y;
				eye1RectsArray[0].x = eye1RectsArray[0].x + eye1Rect.x;
				eye1RectsArray[0].y = eye1RectsArray[0].y + arrayrects[0].y;
				Imgproc.rectangle(image, eye0RectsArray[0], new Scalar(255));
				Imgproc.rectangle(image, eye1RectsArray[0], new Scalar(255));
			}
			
			
			
		}
		
		//System.out.println(Arrays.toString(this.getGuessKnn3(160)));
		
		HighGui.imshow("CamView",image);
		HighGui.waitKey(20);
		}	
		
	}
	
	
	
	/**
	 * Same as debugFaceEyesImprovedView but returns mat only once
	 * Shows the Camera view with detected face if detected and detected eyes (improved) if detected
	 * uses the same parameters as .getEyesImproved, made for debugging and etc
	 */
	public static Mat debugMatFaceEyesImprovedView() {
		

		VideoCapture cap1; 
		//cap1 = new VideoCapture(0);
		
		
		Mat image;
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
		
		cap1 = new VideoCapture(0);
		
		if (!cap1.isOpened())
            System.out.println("Camera Error");
		
		image = new Mat();
		Mat imageGray = new Mat();
		Mat faceMat = new Mat();
		MatOfRect eyesMatRect = new MatOfRect();
		
		/*
		 * This part detects the face
		 */
		
		String filename = "lbpcascade_frontalface_improved.xml"; 
		String eyesfilename = "haarcascade_eye.xml";
		Size dsize = new Size(50,50);
		MatOfRect rect = new MatOfRect();
		Rect[] arrayrects = null;
		Rect eye0Rect = new Rect();
		Rect eye1Rect = new Rect();
		Mat eye0Mat = new Mat();
		Mat eye1Mat = new Mat();
		MatOfRect eye0MatOfRect = new MatOfRect();
		MatOfRect eye1MatOfRect = new MatOfRect();
		Rect[] eye0RectsArray = null;
		Rect[] eye1RectsArray = null;
 		
		CascadeClassifier faceCC = new CascadeClassifier(filename);				//Load CascadeClassifier
		CascadeClassifier eyesCC = new CascadeClassifier(eyesfilename);
		if (faceCC.empty()) 													//Check for errors
			System.out.println("CascadeClassifier didn't load");
		
		
		//while(true) {
		
		cap1.grab();
		cap1.retrieve(image);

		Imgproc.cvtColor(image, imageGray, Imgproc.COLOR_BGR2GRAY);					//Convert our Cam Mat in GrayScale
		
		faceCC.detectMultiScale(imageGray, rect, 2, 2, 0, dsize);					//Detect the face and save the area in MatOfRect::rect
		arrayrects = rect.toArray();
		System.out.println("Faces found: " + arrayrects.length);
		if (arrayrects.length != 0)  {
			Imgproc.rectangle(image, arrayrects[0], new Scalar(255));
			eye0Rect = new Rect(arrayrects[0].x, arrayrects[0].y, (arrayrects[0].width / 2), (arrayrects[0].height / 2));
			eye1Rect = new Rect(arrayrects[0].x + (arrayrects[0].width / 2), arrayrects[0].y, (arrayrects[0].width / 2), (arrayrects[0].height / 2));

			eye0Mat = new Mat(image, eye0Rect);									//Crop the ROI
			eye1Mat = new Mat(image, eye1Rect);
			
			eyesCC.detectMultiScale(eye0Mat, eye0MatOfRect, 1.1, 8);
			eyesCC.detectMultiScale(eye1Mat, eye1MatOfRect, 1.1, 8);
			eye0RectsArray = eye0MatOfRect.toArray();
			eye1RectsArray = eye1MatOfRect.toArray();
			System.out.println("eye 0: " + eye0RectsArray.length);
			System.out.println("eye 1; " + eye1RectsArray.length);
			
			if ((eye0RectsArray.length > 0) && (eye1RectsArray.length > 0)) {				
				eye0RectsArray[0].x = eye0RectsArray[0].x + arrayrects[0].x;
				eye0RectsArray[0].y = eye0RectsArray[0].y + arrayrects[0].y;
				eye1RectsArray[0].x = eye1RectsArray[0].x + eye1Rect.x;
				eye1RectsArray[0].y = eye1RectsArray[0].y + arrayrects[0].y;
				Imgproc.rectangle(image, eye0RectsArray[0], new Scalar(255));
				Imgproc.rectangle(image, eye1RectsArray[0], new Scalar(255));
			}
			
			
			
		}
		
		return image;
		
	}
	
	
	/**
	 * This function displays only eyes from getEyesImprovedRun() continuously scaled in one window
	 * @param scale is the size of one eye (window will have two)
	 */
	
	public static void debugScaledOnlyEyesImprovedView(int scale) {
		CamData roger = new CamData();
		roger.startRunning();
		
		
		int i = 1;
		while (i < 1500) {
		Mat[] rogereyes = roger.getEyesImprovedRun();
		if (rogereyes != null) {
		Imgproc.resize(rogereyes[0], rogereyes[0], new Size(scale, scale));
		Imgproc.resize(rogereyes[1], rogereyes[1], new Size(scale, scale));
		rogereyes[0].push_back(rogereyes[1]);
		HighGui.imshow("eye1", rogereyes[0]);
		HighGui.waitKey(20);
		}
		i++;
		}
		
		roger.stopRunning();
	}
	
	/**
	 * Creates windows with all eyes from calibration sample
	 * This instance method is for debugging purposes and creates many eye windows and lists their coords
	 */
	
	public void debugExistingEyes() {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
		System.out.println("raw");
		if (this.getSize() > 0) {
		for (int i = 0; i < this.getSize(); i++) {
			HighGui.imshow("CamView" + i,this.getEye0(i));
			System.out.println("coords for " + i + " eye " + this.getX(i) + " " + this.getY(i));
			//HighGui.waitKey(20);
		}
		HighGui.waitKey(20);
		}
		
	}
	
	
	
	
	
	//KNN-PART
	
	/**
	 * This method initializes the K-Nearest Neighbour ML instance
	 * It adds the calibration samples from the CamData instance to the KNN-Network
	 * needs to be called AFTER calibration BEFORE tracking
	 * 
	 * @return true when finished (knn0 and knn1 successfully trained), false if 0 elements in the instance or if at least one knn not trained
	 */
	public boolean initializeKNearest(int size) {
		//REFERENCE - THE DEAFULTIZED EYE SIZE IS 120
		if (this.pCounter == 0) 
			return false;
		else {
			HOGDescriptor hogDescriptor = new HOGDescriptor();						//instance of hog descriptor

			/**
			 * This is the eye 0 knn0 training part
			 */
			Mat samples0 = new Mat();												//samples Mat array (empty at initialization)
			Mat responses0 = new Mat(1, this.pCounter, CvType.CV_32F);				//responses Mat array
			//Size winStride = new Size(strS, strS);
			for (int i = 0; i < this.pCounter; i++) {
				Mat workingSample = new Mat();
				Imgproc.resize(eye0Samples[i], workingSample, new Size(size, size));	//resizing eye to a standard size, since all eyes different
				
				MatOfFloat eyehog = new MatOfFloat();		
				//HOGD PARAMS: IMAGE, HOGD_DST, WINDOW STRIDE
				//hogDescriptor
				hogDescriptor.compute(workingSample, eyehog);	//computes hogdescriptor from workingSamples, adds to eyehog
				
				Core.transpose(eyehog, eyehog);					//has to be transposed to be accepted by push_back, hconcat doesn't work with MatOfFloat, conversion takes longer than transposing
				samples0.push_back(eyehog);						//adds eyehog into a new row of samples
				
				responses0.put(0, i, i);							//adds the eye ID into responses
			}
			//System.out.println("samples " + samples0.toString() + " responses " + responses0.toString() + " dump " + responses0.dump());
			knn0 = KNearest.create();
			knn0.train(samples0, Ml.ROW_SAMPLE, responses0);  	//trains the knn, Row_Sample means that each row of samples is one sample
			//now knn0 should be trained 
			
			/**
			 * This is the eye 1 knn1 training part
			 */
			Mat samples1 = new Mat();												//samples Mat array (empty at initialization)
			Mat responses1 = new Mat(1, this.pCounter, CvType.CV_32F);				//responses Mat array
			for (int i = 0; i < this.pCounter; i++) {
				Mat workingSample = new Mat();
				Imgproc.resize(eye1Samples[i], workingSample, new Size(size, size));	//resizing eye to a standard size, since all eyes different
				
				MatOfFloat eyehog = new MatOfFloat();		
				hogDescriptor.compute(workingSample, eyehog);	//computes hogdescriptor from workingSamples, adds to eyehog
				
				Core.transpose(eyehog, eyehog);					//has to be transposed to be accepted by push_back, hconcat doesn't work with MatOfFloat, conversion takes longer than transposing
				samples1.push_back(eyehog);						//adds eyehog into a new row of samples
				
				responses1.put(0, i, i);							//adds the eye ID into responses
			}
			knn1 = KNearest.create();
			knn1.train(samples1, Ml.ROW_SAMPLE, responses1);  	//trains the knn, Row_Sample means that each row of samples is one sample
			//now knn1 should be trained
			
			
			return (knn0.isTrained() && knn1.isTrained());
		}
	}
	
	/**
	 * This method has to be called before actual tracking but after calibration
	 * Summarizes and feeds the saved eyes of the instance to the ML Algorithm 
	 * InitializeKNearest with the default size of 160
	 * @return the same as initializeKNearest(size)
	 */
	public boolean initializeKNearest() {
		return this.initializeKNearest(160);
	}
	
	/**
	 * Interpolates the calibration coordinates of guessed 3 neighbours, that are given as ID
	 * @param firstID	the first neighbour ID that was returned by the knn
	 * @param secondID	the second neighbour ID that was returned by the knn
	 * @param thirdID	the third neighbour ID that was returned by the knn
	 * @return the interpolated guessed coordinates as an int[]{xCoord,yCoord}
	 */
	
	public int[] interP(int firstID, int secondID, int thirdID) {
		
		int x = (int)(this.getX(firstID) + this.getX(secondID) + this.getX(thirdID))/3;
		int y = (int)(this.getY(firstID) + this.getY(secondID) + this.getY(thirdID))/3;
		
		//System.out.println("interP now" + this.getX(firstID));
		//System.out.println("x " + x);
		
		int[] returned = new int[2];
		returned[0] = x;
		returned[1] = y;
		
		//System.out.println("returned " + returned);
		
		return returned;
		
		
	}
	
	
	/**
	 * interpolates the coorinates of two point IDs and returns an array of [x, y]
	 * @param firstID first point ID
	 * @param secondID second point ID
	 * @return the coorinates between them in an int[] of size 2
	 */
	public int[] interP2(int firstID, int secondID) {
		
		int x = (int)(this.getX(firstID) + this.getX(secondID))/2;
		int y = (int)(this.getY(firstID) + this.getY(secondID))/2;
		
		//System.out.println("interP now" + this.getX(firstID));
		//System.out.println("x " + x);
		
		int[] returned = new int[2];
		returned[0] = x;
		returned[1] = y;
		
		//System.out.println("returned " + returned);
		
		return returned;
		
		
	}
	
	
	/**
	 * Returns the prediction for the current view position
	 * this method is the default prediction coordiantes method that returns coordinates
	 * @return coordinates
	 */
	
	public int[] getGuessRun() {
		return getGuessRun(160);
	}
	
	/**
	 * this instance method is used to get the guessed coordinates of the immediate eye capture
	 * requires startRunning() to be executed before starting this method
	 * usually combines three coords but sometimes rejects one result if too far away
	 * never rejects thefirst guess tho because first
	 * @return an int[] array of guessed interpolated coordinates of both eyes [xCoord, yCoord] (might be changed)
	 */
	
	public int[] getGuessRun(int size) {
		Mat[] eyes = this.getEyesImprovedRun();
		Mat results = new Mat();
		//Size winStride = new Size(strS, strS);
		
		//first reform eyes for knn
		if (eyes == null) 
			System.out.println("EYES ARE NULL");
		Imgproc.resize(eyes[0], eyes[0], new Size(size, size));
		Imgproc.resize(eyes[1], eyes[1], new Size(size, size));
		
		MatOfFloat eyehog0 = new MatOfFloat();		
		MatOfFloat eyehog1 = new MatOfFloat();	
		
		HOGDescriptor hogDescriptor = new HOGDescriptor();		//unify (maybe)
		//hog compute here
		hogDescriptor.compute(eyes[0], eyehog0);	
		hogDescriptor.compute(eyes[1], eyehog1);
		
		Core.transpose(eyehog0, eyehog0);					
		Core.transpose(eyehog1, eyehog1);	
		
		//first knn0 guess
		Mat neighbourResponses0 = new Mat();
		this.knn0.findNearest(eyehog0, 3, results, neighbourResponses0);
		
		int fst = (int)neighbourResponses0.get(0, 0)[0];
		int snd = (int)neighbourResponses0.get(0, 1)[0];
		int thd = (int)neighbourResponses0.get(0, 2)[0];
		//System.out.println("fist " + fst + " snd " + snd + " thd " + thd);		
		
		
		//get dimensions
		Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		//Imgproc.resize(image, image, new Size (screenSize.width, screenSize.height));
		int[] eye0guess;
		
		if (distance(fst, snd) > screenSize.height) {
			if (distance(fst, thd) > screenSize.height) {
				if (distance(fst,snd) >= distance(fst,thd)) {
					//if distance of 2 is bigger than distance of 3 remove 2
					eye0guess = this.interP2(fst, thd);
				}	else {
					//if distance of 3 is bigger than distance of 2 remove 3
					eye0guess = this.interP2(fst, snd);
				}
			} else {
				//if distance snd is bigger but distanceof thd is NOT bigger remove snd
				eye0guess = this.interP2(fst, thd);
			}
		} else if (distance(fst,thd) > screenSize.height) {
			//distance of snd is not BIG but distance of thd is BIG remove thd
			eye0guess = this.interP2(fst, snd);
			}
		else {
			//distance of neither is big
		
		
		
		eye0guess = this.interP(fst, snd, thd);
		//System.out.println("eye0guess" + eye0guess);
		}
		
		
		
		//now knn1 guess
		Mat neighbourResponses1 = new Mat();
		this.knn1.findNearest(eyehog1, 3, results, neighbourResponses1);
		
		fst = (int)neighbourResponses1.get(0, 0)[0];
		snd = (int)neighbourResponses1.get(0, 1)[0];
		thd = (int)neighbourResponses1.get(0, 2)[0];
		
		
		
		int[] eye1guess;
		if (distance(fst, snd) > screenSize.height) {
			if (distance(fst, thd) > screenSize.height) {
				if (distance(fst,snd) >= distance(fst,thd)) {
					//if distance of 2 is bigger than distance of 3 remove 2
					eye1guess = this.interP2(fst, thd);
				}	else {
					//if distance of 3 is bigger than distance of 2 remove 3
					eye1guess = this.interP2(fst, snd);
				}
			} else {
				//if distance snd is bigger but distanceof thd is NOT bigger remove snd
				eye1guess = this.interP2(fst, thd);
			}
		} else if (distance(fst,thd) > screenSize.height) {
			//distance of snd is not BIG but distance of thd is BIG remove thd
			eye1guess = this.interP2(fst, snd);
			}
		else {
			//distance of neither is big
		
		
		
		//System.out.println("fist " + fst + " snd " + snd + " thd " + thd);
		eye1guess = this.interP(fst, snd, thd);
		//System.out.println("eye1guess " + eye1guess);
		}
		//now combine and return
		
		int[] finalGuess = combine2(eye0guess, eye1guess);
		
		//int[] finalGuess1 = {finalGuess[0], finalGuess[1], finalGuess[0], finalGuess[1], finalGuess[0], finalGuess[1]};
		
		return finalGuess;
	}
	
	/**
	 * combine two coordinate array [x,y] and [x,y] into one by interpolating them 
	 * @param eye0	the first coordinate array
	 * @param eye1	the second coordinate array
	 * @return	the combined interpolated coordinate array [x,y]
	 */
	
	public static int[] combine2(int[] eye0, int[] eye1) {
		
		int x = ((eye0[0] + eye1[0])/2);
		int y = ((eye0[1] + eye1[1])/2);
		
		return new int[]{x,y};
		
	}
	
	/**
	 * same as debugFaceEyesImprovedVIew() but with the guess being drawn on top
	 * considered a failure since imshow can't cooperate with swing without 
	 * 
	 */
	
	public void debugGuessFaceEyesImprovedViewRun() {
		

		//VideoCapture cap1; 
		//cap1 = new VideoCapture(0);
		
		
		Mat image;
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
		
		//cap = new VideoCapture(0);
		
		if (!caprun.isOpened())
            System.out.println("Camera Error");
		
		image = new Mat();
		Mat imageGray = new Mat();
		Mat faceMat = new Mat();
		MatOfRect eyesMatRect = new MatOfRect();
		
		/*
		 * This part detects the face
		 */
		
		String filename = "lbpcascade_frontalface_improved.xml"; 
		String eyesfilename = "haarcascade_eye.xml";
		Size dsize = new Size(50,50);
		MatOfRect rect = new MatOfRect();
		Rect[] arrayrects = null;
		Rect eye0Rect = new Rect();
		Rect eye1Rect = new Rect();
		Mat eye0Mat = new Mat();
		Mat eye1Mat = new Mat();
		MatOfRect eye0MatOfRect = new MatOfRect();
		MatOfRect eye1MatOfRect = new MatOfRect();
		Rect[] eye0RectsArray = null;
		Rect[] eye1RectsArray = null;
		Mat[] returneyes = new Mat[2];//added for knn
		
		CascadeClassifier faceCC = new CascadeClassifier(filename);				//Load CascadeClassifier
		CascadeClassifier eyesCC = new CascadeClassifier(eyesfilename);
		if (faceCC.empty()) 													//Check for errors
			System.out.println("CascadeClassifier didn't load");
		
		
		//this.startRunning(); //can be removed
		
		while(true) {
		
		caprun.grab();
		caprun.retrieve(image);
		Mat original = new Mat();
		
		Imgproc.cvtColor(image, imageGray, Imgproc.COLOR_BGR2GRAY);					//Convert our Cam Mat in GrayScale
		image.copyTo(original);
		
		faceCC.detectMultiScale(imageGray, rect, 2, 2, 0, dsize);					//Detect the face and save the area in MatOfRect::rect
		arrayrects = rect.toArray();
		System.out.println("Faces found: " + arrayrects.length);
		if (arrayrects.length != 0)  {
			Imgproc.rectangle(image, arrayrects[0], new Scalar(255));
			eye0Rect = new Rect(arrayrects[0].x, arrayrects[0].y, (arrayrects[0].width / 2), (arrayrects[0].height / 2));
			eye1Rect = new Rect(arrayrects[0].x + (arrayrects[0].width / 2), arrayrects[0].y, (arrayrects[0].width / 2), (arrayrects[0].height / 2));

			eye0Mat = new Mat(image, eye0Rect);									//Crop the ROI
			eye1Mat = new Mat(image, eye1Rect);
			
			eyesCC.detectMultiScale(eye0Mat, eye0MatOfRect, 1.1, 8);	//eye0Mat the picture with eye0
			eyesCC.detectMultiScale(eye1Mat, eye1MatOfRect, 1.1, 8);	//eye1Mat the picture with eye1
			eye0RectsArray = eye0MatOfRect.toArray();					//array for eye0
			eye1RectsArray = eye1MatOfRect.toArray();					//array for eye1
			System.out.println("eye 0: " + eye0RectsArray.length);
			System.out.println("eye 1; " + eye1RectsArray.length);
			
			if ((eye0RectsArray.length > 0) && (eye1RectsArray.length > 0)) {				
				
				returneyes[0] = new Mat(eye0Mat, eye0RectsArray[0]);
				returneyes[1] = new Mat(eye1Mat, eye1RectsArray[0]);
				
				eye0RectsArray[0].x = eye0RectsArray[0].x + arrayrects[0].x;
				eye0RectsArray[0].y = eye0RectsArray[0].y + arrayrects[0].y;
				eye1RectsArray[0].x = eye1RectsArray[0].x + eye1Rect.x;
				eye1RectsArray[0].y = eye1RectsArray[0].y + arrayrects[0].y;
				Imgproc.rectangle(image, eye0RectsArray[0], new Scalar(255));
				Imgproc.rectangle(image, eye1RectsArray[0], new Scalar(255));
				
				
				//part with guessing
				
				

				
				//now the eyes are in returneyes 0
				
			}
			
			
			
		}
		
		//Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		//System.out.println(screenSize.height + " " + screenSize.width);
		
		//Imgproc.resize(image, image, new Size (screenSize.width, screenSize.height));
		
		//int[] coordsarray = this.getGuessRun(160);

		//Imgproc.circle(image, new Point(3000, 3000), 50, new Scalar(0, 0, 255));
		
		//System.out.println("after point drawin");

		

		HighGui.imshow("CamView",image);
		HighGui.waitKey(20);
		}	
		
	}

	
	/**
	 * 
	 * @return a Mat of a picture and overlayed detected viewpoint.
	 */
	
	public int[] getGuessMat(int size, Mat[] eyes) {
		//Mat[] eyes = this.getEyesImprovedRun();
		Mat results = new Mat();
		
		//first reform eyes for knn
		Imgproc.resize(eyes[0], eyes[0], new Size(size, size));
		Imgproc.resize(eyes[1], eyes[1], new Size(size, size));
		
		MatOfFloat eyehog0 = new MatOfFloat();		
		MatOfFloat eyehog1 = new MatOfFloat();	
		
		HOGDescriptor hogDescriptor = new HOGDescriptor();		//unify (maybe)
		
		hogDescriptor.compute(eyes[0], eyehog0);	
		hogDescriptor.compute(eyes[1], eyehog1);
		
		Core.transpose(eyehog0, eyehog0);					
		Core.transpose(eyehog1, eyehog1);	
		
		//first knn0 guess
		Mat neighbourResponses0 = new Mat();
		this.knn0.findNearest(eyehog0, 3, results, neighbourResponses0);
		
		int fst = (int)neighbourResponses0.get(0, 0)[0];
		int snd = (int)neighbourResponses0.get(0, 1)[0];
		int thd = (int)neighbourResponses0.get(0, 2)[0];
		//System.out.println("fist " + fst + " snd " + snd + " thd " + thd);		
		int[] eye0guess = this.interP(fst, snd, thd);
		//System.out.println("eye0guess" + eye0guess);
		
		//now knn1 guess
		Mat neighbourResponses1 = new Mat();
		this.knn1.findNearest(eyehog1, 3, results, neighbourResponses1);
		
		fst = (int)neighbourResponses1.get(0, 0)[0];
		snd = (int)neighbourResponses1.get(0, 1)[0];
		thd = (int)neighbourResponses1.get(0, 2)[0];
		//System.out.println("fist " + fst + " snd " + snd + " thd " + thd);
		int[] eye1guess = this.interP(fst, snd, thd);
		//System.out.println("eye1guess " + eye1guess);
		
		//now combine and return
		
		int[] finalGuess = combine2(eye0guess, eye1guess);
		return finalGuess;
	}
	
	/**
	 * calls getGuessKnn3 with default values 160 and returns the 3 closest neighbours
	 * @return returns the same as getGuessKnn3(160);
	 */
	public int[] getGuessKnn3() {
		return getGuessKnn3(160);
	}
	

	/**
	 * returns the 3 nearest neighbours coordinates as [x1,y1, x2,y2, x3,y3]
	 * the first neighbours are the closest
	 */
	public int[] getGuessKnn3(int size) {
		
		Mat[] eyes = null;
		//boolean wagon = true;
		//while (wagon) {
			eyes = this.getEyesImprovedRun();
		//	if (eyes != null)
		//		wagon = false;
		//}
		
		Mat results = new Mat();
		
		//first reform eyes for knn
		
		Imgproc.resize(eyes[0], eyes[0], new Size(size, size));
		Imgproc.resize(eyes[1], eyes[1], new Size(size, size));
		
		MatOfFloat eyehog0 = new MatOfFloat();		
		MatOfFloat eyehog1 = new MatOfFloat();	
		
		HOGDescriptor hogDescriptor = new HOGDescriptor();		//unify (maybe)
		
		hogDescriptor.compute(eyes[0], eyehog0);	
		hogDescriptor.compute(eyes[1], eyehog1);
		
		Core.transpose(eyehog0, eyehog0);					
		Core.transpose(eyehog1, eyehog1);	
		
		int[] returnInt0 = new int[6];
		int[] returnInt1 = new int[6];
		
		//first knn0 guess
		Mat neighbourResponses0 = new Mat();
		System.out.println(knn0.isTrained());
		this.knn0.findNearest(eyehog0, 3, results, neighbourResponses0);
		
		int fst = (int)neighbourResponses0.get(0, 0)[0];
		int snd = (int)neighbourResponses0.get(0, 1)[0];
		int thd = (int)neighbourResponses0.get(0, 2)[0];
		//System.out.println("fist " + fst + " snd " + snd + " thd " + thd);		
		//int[] eye0guess = this.interP(fst, snd, thd);
		//System.out.println("eye0guess" + eye0guess);
		
		
		returnInt0[0] = this.getX(fst);
		returnInt0[2] = this.getX(snd);
		returnInt0[4] = this.getX(thd);
		
		returnInt0[1] = this.getY(fst);
		returnInt0[3] = this.getY(snd);
		returnInt0[5] = this.getY(thd);
		
		
		//now knn1 guess
		Mat neighbourResponses1 = new Mat();
		this.knn1.findNearest(eyehog1, 3, results, neighbourResponses1);
		
		fst = (int)neighbourResponses1.get(0, 0)[0];
		snd = (int)neighbourResponses1.get(0, 1)[0];
		thd = (int)neighbourResponses1.get(0, 2)[0];
		//System.out.println("fist " + fst + " snd " + snd + " thd " + thd);
		//int[] eye1guess = this.interP(fst, snd, thd);
		//System.out.println("eye1guess " + eye1guess);
		
		returnInt1[0] = this.getX(fst);
		returnInt1[2] = this.getX(snd);
		returnInt1[4] = this.getX(thd);
		
		returnInt1[1] = this.getY(fst);
		returnInt1[3] = this.getY(snd);
		returnInt1[5] = this.getY(thd);
		
		//for (int i = 0; i < 6; i++) {
		//	returnInt0[i] = (returnInt0[i] + returnInt1[i])/2;
		//}
		
		//now combine and return
		
		//int[] finalGuess = combine2(eye0guess, eye1guess);
		
		//System.out.println(Arrays.toString(returnIntX));
		return returnInt0;
		
	}
	
	
	/**
	 * special function required for looping inside GUI and exiting by closing eyes (doesn't work)
	 * returns true if eyes have been detected or returns false if eyes haven't been detected for more than 15 (default) iteration
	 * @return
	 */
	public boolean checkForEyes() {
		
		if (getEyesImprovedRunWithCounter() == null) {
			return false;
		}
		return true;
	}
	
	
	/**
	 * special method for checkForEyes() that counts the number of failed attempts of capturing eyes
	 * @return false as soon as there are at least 15 times without two captured eyes
	 */
	public Mat[] getEyesImprovedRunWithCounter() {
		
		
		
		Mat image;
		//System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
		
		/*
		 * This part captures an image from the web cam
		 */
		
		if (!caprun.isOpened())
            System.out.println("Camera Error");
		image = new Mat();
		
		
		/*
		 * This part detects the face
		 */
		
		String filename = "lbpcascade_frontalface_improved.xml";			//lbp cascade file location
		String filenameeye = "haarcascade_eye.xml";							//haar cascade file location
		Size dsize = new Size(50,50);										
		MatOfRect rect = new MatOfRect();
		Rect[] arrayrects = null;
		Mat[] returneyes = new Mat[2];
		boolean detected = false;
		int iterationCounter = 0;
		Rect eye0Rect = null;
		Rect eye1Rect = null;
		Mat eye0Mat = null;								//Crop the ROI
		Mat eye1Mat = null;
		MatOfRect eye0MatOfRect = new MatOfRect();
		MatOfRect eye1MatOfRect = new MatOfRect();
		Rect[] eye0RectsArray = null;
		Rect[] eye1RectsArray = null;

		
		CascadeClassifier faceCC = new CascadeClassifier(filename);				//Load CascadeClassifier
		if (faceCC.empty()) 													//Check for errors
			System.out.println("CascadeClassifier FACE didn't load");
		CascadeClassifier eyeCC = new CascadeClassifier(filenameeye);
		if (eyeCC.empty()) 													//Check for errors
			System.out.println("CascadeClassifier EYE didn't load");
			
		
		while(!detected) {
		
		if (iterationCounter > 15) 
			return null;
			
		caprun.grab();
		caprun.retrieve(image);

		Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2GRAY);					//Convert our Cam Mat in GrayScale
		
		faceCC.detectMultiScale(image, rect, 2, 2, 0, dsize);					//Detect the face and save the area in MatOfRect::rect
		arrayrects = rect.toArray();
		iterationCounter++;
		

		
		/*
		 * This part detects eyes
		 */
		if (arrayrects.length != 0)  {
			
			eye0Rect = new Rect(arrayrects[0].x, arrayrects[0].y, (arrayrects[0].width / 2), (arrayrects[0].height / 2));
			eye1Rect = new Rect(arrayrects[0].x + (arrayrects[0].width / 2), arrayrects[0].y, (arrayrects[0].width / 2), (arrayrects[0].height / 2));

			eye0Mat = new Mat(image, eye0Rect);									//Crop the ROI
			eye1Mat = new Mat(image, eye1Rect);

			eyeCC.detectMultiScale(eye0Mat, eye0MatOfRect, 1.1, 8);
			eyeCC.detectMultiScale(eye1Mat, eye1MatOfRect, 1.1, 8);
			eye0RectsArray = eye0MatOfRect.toArray();
			eye1RectsArray = eye1MatOfRect.toArray();
			//System.out.println(eye0RectsArray.length + " " + eye1RectsArray.length);
			if ((eye0RectsArray.length > 0) && (eye1RectsArray.length > 0)) 
				detected = true;
			}
			iterationCounter++;
		
		}
		
		/*
		 * This part saves detected eyes and returns them as Mat[2]
		 */
		returneyes[0] = new Mat(eye0Mat, eye0RectsArray[0]);
		returneyes[1] = new Mat(eye1Mat, eye1RectsArray[0]);
		
		Imgproc.equalizeHist(returneyes[0], returneyes[0]);
		Imgproc.equalizeHist(returneyes[1], returneyes[1]);
		
		
		return returneyes;
		
		
	}
	
	
	/**
	 * calculates the euclidian distance between the two 2d CamData point ID coordinates
	 * @param id1 first point ID
	 * @param id2 second point ID
	 * @return  distance 
	 */
	
	public int distance(int id1, int id2) {
		
		double distance = Math.sqrt(
				 ( this.getX(id1) - this.getX(id2) )^2 +
				 ( this.getY(id1) - this.getY(id2) )^2				
				);
		
		return (int)distance;		
		
	}

}
